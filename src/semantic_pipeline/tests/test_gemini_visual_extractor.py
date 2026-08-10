import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import semantic_pipeline.gemini.extractor as gemini_visual_extractor
from semantic_pipeline.core.compact_metadata import CompactVisualRecord
from semantic_pipeline.core.frame_id import frame_id_from_path, parse_frame_id
from semantic_pipeline.gemini.extractor import (
    GeminiDetection,
    GeminiFrameResult,
    GeminiVisualExtractor,
    build_visual_prompt,
    canonicalize_response_frame_ids,
    compact_record_from_gemini,
    load_gemini_api_key,
    load_gemini_visual_model,
    map_response_slots_to_frame_ids,
    run_extraction,
    sanitise_gemini_payload,
)
from semantic_pipeline.core.request_limits import (
    is_daily_quota_error,
    is_rate_limit_error,
    is_transient_service_error,
)
from semantic_pipeline.core.visual_profiles import VisualContextResolver


def make_image(tmp_path: Path, name: str = "001.jpg") -> Path:
    image_dir = tmp_path / "L21_V001"
    image_dir.mkdir()
    path = image_dir / name
    Image.new("RGB", (200, 100), "white").save(path)
    return path


def test_visual_prompt_requires_boxes_for_physical_scene_subjects():
    prompt = build_visual_prompt()
    assert "return at least one useful box" in prompt
    assert "damaged pavement, rubble, pipes" in prompt
    assert "Return an empty detections array only" in prompt
    assert "genuinely empty/blurred frame" in prompt


def test_parses_frame_id_from_keyframe_path(tmp_path):
    path = make_image(tmp_path, "017.jpg")
    assert frame_id_from_path(path) == "L21_V001_f0017"
    assert parse_frame_id("L21_V001_f0017").video_name == "L21_V001"
    assert parse_frame_id("L21_V001_f0017").frame_index == 17


def test_domain_profile_is_added_as_ephemeral_prompt_context():
    context = VisualContextResolver().for_frame_id("L26_V001_f0001")
    prompt = build_visual_prompt(context)

    assert context.profile.profile_id == "cooking"
    assert "food and cooking" in prompt
    assert "ingredient" in prompt
    assert "trust visible pixels" in prompt


def test_news_profile_requires_a_separate_bottom_ticker_transcript():
    prompt = build_visual_prompt(VisualContextResolver().for_frame_id("L21_V001_f0001"))

    assert "news_ticker_text" in prompt
    assert "scrolling news ticker/crawl" in prompt


def test_prompt_requests_direct_vietnamese_visual_captions():
    prompt = build_visual_prompt()

    assert "caption_vi" in prompt
    assert "generated directly from visual" in prompt
    assert "Do not translate or paraphrase `caption`" in prompt


def test_prompt_limits_optional_object_enrichment():
    prompt = build_visual_prompt()
    normalized_prompt = " ".join(prompt.split())

    assert "no more than five" in normalized_prompt
    assert "description_vi" in prompt
    assert "at most six short lowercase English visual attributes" in normalized_prompt
    assert "ordinary, background, partly hidden, or non-distinctive object" in normalized_prompt


def test_maps_gemini_boxes_to_compact_detections_and_relations(tmp_path):
    path = make_image(tmp_path)
    result = GeminiFrameResult(
        frame_id="L21_V001_f0001",
        caption=" A person next to a car. ",
        detailed_caption=" A person stands beside a parked car on a street. ",
        caption_vi=" Một người đứng cạnh ô tô. ",
        detailed_caption_vi=" Một người đứng bên cạnh chiếc ô tô đỗ trên đường. ",
        ocr_text="  BAI  XE ",
        news_ticker_text="  Tin tuc moi nhat ",
        detections=[
            GeminiDetection(
                label="person",
                box_2d=[100, 100, 900, 400],
                description=" a person wearing a bright red coat ",
                description_vi=" một người mặc áo khoác đỏ nổi bật ",
                attributes=[" Red Coat ", "standing", "red coat"],
                action=" Standing ",
            ),
            GeminiDetection(
                label="car",
                box_2d=[100, 600, 900, 950],
                description="a blue parked car",
                description_vi="một chiếc ô tô màu xanh đang đỗ",
                attributes=["blue", "parked"],
            ),
        ],
    )

    record = compact_record_from_gemini(result, path)

    assert record.caption == "A person next to a car."
    assert record.detailed_caption == "A person stands beside a parked car on a street."
    assert record.caption_vi == "Một người đứng cạnh ô tô."
    assert record.detailed_caption_vi == "Một người đứng bên cạnh chiếc ô tô đỗ trên đường."
    assert record.ocr_text == "BAI XE"
    assert record.news_ticker_text == "Tin tuc moi nhat"
    assert [item.object_id for item in record.detections] == ["car_0", "person_0"]
    assert record.detections[0].bbox == (0.6, 0.1, 0.95, 0.9)
    assert record.detections[0].description == "a blue parked car"
    assert record.detections[0].attributes == ["blue", "parked"]
    assert record.detections[1].description == "a person wearing a bright red coat"
    assert record.detections[1].description_vi == "một người mặc áo khoác đỏ nổi bật"
    assert record.detections[1].attributes == ["red coat", "standing"]
    assert record.detections[1].action == "standing"
    triples = {
        (item.subject_id, item.predicate, item.object_id)
        for item in record.spatial_relations
    }
    assert ("person_0", "left_of", "car_0") in triples
    assert ("car_0", "right_of", "person_0") in triples


def test_keeps_boxes_but_enriches_at_most_five_objects(tmp_path):
    path = make_image(tmp_path)
    result = GeminiFrameResult(
        frame_id="L21_V001_f0001",
        caption="Six objects.",
        detailed_caption="Six separate objects appear across the image.",
        detections=[
            GeminiDetection(
                label=f"object {index}",
                box_2d=[100, index * 160, 900, index * 160 + 120],
                description=f"distinctive object {index}",
                description_vi=f"vật thể nổi bật {index}",
                attributes=[f"attribute {index}"],
            )
            for index in range(6)
        ],
    )

    record = compact_record_from_gemini(result, path)

    assert len(record.detections) == 6
    assert sum(bool(item.description) for item in record.detections) == 5
    assert next(item for item in record.detections if item.label == "object 5").description == ""


def test_drops_editorial_overlay_detection_labels(tmp_path):
    path = make_image(tmp_path)
    record = compact_record_from_gemini(
        GeminiFrameResult(
            frame_id="L21_V001_f0001",
            caption="A program title card.",
            detailed_caption="A red program title card appears on a white background.",
            detections=[
                GeminiDetection(label="logo", box_2d=[100, 100, 900, 900]),
                GeminiDetection(label="watermark", box_2d=[0, 0, 100, 100]),
            ],
        ),
        path,
    )

    assert record.detections == []


def test_drops_only_truncated_boxes_and_keeps_the_frame_payload():
    payload, malformed = sanitise_gemini_payload(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_id": "L21_V001_f0001",
                        "caption": "A news anchor.",
                        "detailed_caption": "A news anchor appears in a studio.",
                        "detections": [
                            {"label": "anchor", "box_2d": [10, 20, 900, 800]},
                            {"label": "bad box", "box_2d": [0, 0, 1000]},
                        ],
                    }
                ]
            }
        )
    )

    assert len(malformed) == 1
    assert payload["frames"][0]["caption"] == "A news anchor."
    assert payload["frames"][0]["detections"] == [
        {"label": "anchor", "box_2d": [10, 20, 900, 800]}
    ]


def test_rejects_response_for_another_frame(tmp_path):
    path = make_image(tmp_path)
    result = GeminiFrameResult(
        frame_id="L21_V001_f0002",
        caption="wrong frame",
        detailed_caption="A test image for the wrong frame identifier.",
    )
    with pytest.raises(ValueError, match="expected"):
        compact_record_from_gemini(result, path)


def test_canonicalizes_harmless_gemini_frame_padding_difference():
    frames = canonicalize_response_frame_ids(
        [
            GeminiFrameResult(
                frame_id="L26_V001_f00010",
                caption="A test frame.",
                detailed_caption="A test frame used to normalize Gemini frame padding.",
            )
        ],
        {"L26_V001_f0010"},
    )

    assert frames[0].frame_id == "L26_V001_f0010"


def test_valid_slots_override_swapped_gemini_frame_ids(tmp_path):
    first_path = make_image(tmp_path, "001.jpg")
    second_path = first_path.parent / "002.jpg"
    Image.new("RGB", (200, 100), "white").save(second_path)

    frames = map_response_slots_to_frame_ids(
        [
            GeminiFrameResult(
                slot=1,
                frame_id="L21_V001_f0002",
                caption="First image caption.",
                detailed_caption="Caption for the first image in input order.",
            ),
            GeminiFrameResult(
                slot=2,
                frame_id="L21_V001_f0001",
                caption="Second image caption.",
                detailed_caption="Caption for the second image in input order.",
            ),
        ],
        [first_path, second_path],
    )

    assert [frame.frame_id for frame in frames] == [
        "L21_V001_f0001",
        "L21_V001_f0002",
    ]
    assert frames[0].caption == "First image caption."


def test_retries_only_the_frame_omitted_from_a_batch(tmp_path):
    first_path = make_image(tmp_path, "001.jpg")
    second_path = first_path.parent / "002.jpg"
    Image.new("RGB", (200, 100), "white").save(second_path)

    class FakeModels:
        def __init__(self):
            self.responses = [
                {
                    "frames": [
                        {
                            "frame_id": "L21_V001_f0001",
                            "caption": "First frame.",
                            "detailed_caption": "The first test frame is white.",
                            "ocr_text": "",
                            "detections": [],
                        }
                    ]
                },
                {
                    "frames": [
                        {
                            "frame_id": "L21_V001_f0002",
                            "caption": "Second frame.",
                            "detailed_caption": "The second test frame is white.",
                            "ocr_text": "",
                            "detections": [],
                        }
                    ]
                },
            ]

        def generate_content(self, **_kwargs):
            return SimpleNamespace(text=json.dumps(self.responses.pop(0)))

    extractor = object.__new__(GeminiVisualExtractor)
    extractor.client = SimpleNamespace(models=FakeModels())
    extractor.model_name = "test-model"
    extractor.context_resolver = VisualContextResolver()
    extractor.use_video_context = True
    extractor.max_output_tokens = 65_536

    records = extractor.extract_batch([first_path, second_path])

    assert [record.frame_id for record in records] == ["L21_V001_f0001", "L21_V001_f0002"]


def test_discards_an_unknown_frame_id_and_retries_the_missing_frame(tmp_path):
    first_path = make_image(tmp_path, "001.jpg")
    second_path = first_path.parent / "002.jpg"
    Image.new("RGB", (200, 100), "white").save(second_path)

    class FakeModels:
        def __init__(self):
            self.responses = [
                {
                    "frames": [
                        {
                            "frame_id": "L21_V001_f0001",
                            "caption": "First frame.",
                            "detailed_caption": "The first test frame is white.",
                            "ocr_text": "",
                            "detections": [],
                        },
                        {
                            "frame_id": "L21_V001_f0003",
                            "caption": "Wrong adjacent frame.",
                            "detailed_caption": "This ID was hallucinated by Gemini.",
                            "ocr_text": "",
                            "detections": [],
                        },
                    ]
                },
                {
                    "frames": [
                        {
                            "frame_id": "L21_V001_f0002",
                            "caption": "Second frame.",
                            "detailed_caption": "The second test frame is white.",
                            "ocr_text": "",
                            "detections": [],
                        }
                    ]
                },
            ]

        def generate_content(self, **_kwargs):
            return SimpleNamespace(text=json.dumps(self.responses.pop(0)))

    extractor = object.__new__(GeminiVisualExtractor)
    extractor.client = SimpleNamespace(models=FakeModels())
    extractor.model_name = "test-model"
    extractor.context_resolver = VisualContextResolver()
    extractor.use_video_context = True
    extractor.max_output_tokens = 65_536

    records = extractor.extract_batch([first_path, second_path])

    assert [record.frame_id for record in records] == ["L21_V001_f0001", "L21_V001_f0002"]


def test_compact_schema_omits_derived_fields(tmp_path):
    path = make_image(tmp_path)
    record = compact_record_from_gemini(
        GeminiFrameResult(
            frame_id="L21_V001_f0001",
            caption="A white image.",
            detailed_caption="A uniformly white test image with no visible objects.",
        ),
        path,
    )
    dumped = record.model_dump(mode="json")
    assert set(dumped) == {
        "frame_id",
        "caption",
        "detailed_caption",
        "caption_vi",
        "detailed_caption_vi",
        "ocr_text",
        "news_ticker_text",
        "detections",
        "spatial_relations",
    }
    assert CompactVisualRecord.model_validate(dumped) == record


def test_batches_respect_count_and_byte_budget(tmp_path):
    paths = []
    for number, size in enumerate((5, 5, 7), start=1):
        path = tmp_path / f"L21_V001_f{number:04d}.jpg"
        path.write_bytes(b"x" * size)
        paths.append(path)

    batches = GeminiVisualExtractor.batch_paths(
        paths,
        batch_size=2,
        max_inline_bytes=10,
    )
    assert [[path.name for path in batch] for batch in batches] == [
        ["L21_V001_f0001.jpg", "L21_V001_f0002.jpg"],
        ["L21_V001_f0003.jpg"],
    ]


def test_load_api_key_prefers_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")
    assert load_gemini_api_key() == "from-environment"


def test_load_visual_model_prefers_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_VISUAL_MODEL", "model-from-environment")
    assert load_gemini_visual_model() == "model-from-environment"


def test_distinguishes_daily_quota_errors_from_rpm_errors():
    assert is_daily_quota_error(
        RuntimeError("429 quota_exceeded GenerateRequestsPerDayPerProject")
    )
    assert not is_daily_quota_error(
        RuntimeError("429 quota_exceeded GenerateRequestsPerMinutePerProject")
    )
    assert not is_daily_quota_error(RuntimeError("429 rate_limit_exceeded RPM"))
    assert is_rate_limit_error(
        RuntimeError("GenerateRequestsPerMinutePerProject-FreeTier")
    )
    assert not is_rate_limit_error(RuntimeError("GenerateRequestsPerDayPerProject"))


def test_classifies_temporary_model_overload_as_retryable():
    assert is_transient_service_error(
        RuntimeError("503 UNAVAILABLE model is currently experiencing high demand")
    )
    assert not is_transient_service_error(
        RuntimeError("429 RESOURCE_EXHAUSTED daily quota")
    )
    assert is_transient_service_error(
        RuntimeError("Gemini returned empty response text; the batch can be retried")
    )


def test_writes_a_separate_caption_artifact_for_each_video(tmp_path, monkeypatch):
    input_dir = tmp_path / "keyframes"
    first_path = input_dir / "L21_V001" / "001.jpg"
    second_path = input_dir / "L22_V002" / "001.jpg"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(first_path)
    Image.new("RGB", (200, 100), "white").save(second_path)

    class FakeExtractor:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def batch_paths(image_paths, **_kwargs):
            return [list(image_paths)]

        def extract_batch(self, image_paths, with_spatial=True):
            assert with_spatial is True
            return [
                CompactVisualRecord(
                    frame_id=frame_id_from_path(path),
                    caption="A white test frame.",
                    detailed_caption="A uniformly white test image.",
                )
                for path in image_paths
            ]

    monkeypatch.setattr(gemini_visual_extractor, "GeminiVisualExtractor", FakeExtractor)
    output_dir = tmp_path / "metadata" / "caption"

    summary = run_extraction(
        input_dir,
        output_dir,
        api_key="test-key",
        global_filter_results_path=None,
        request_budget_state_path=tmp_path / "request_budget.json",
    )

    assert summary == {
        "processed": 2,
        "api_frames": 2,
        "total": 2,
        "batches": 2,
        "daily_limit_reached": False,
        "daily_quota_reported": False,
    }
    assert (output_dir / "L21" / "L21_V001.json").is_file()
    assert (output_dir / "L22" / "L22_V002.json").is_file()
    assert not list(output_dir.glob("*.json"))


def test_global_filter_sends_only_representative_and_copies_its_metadata(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "keyframes" / "L99_V001"
    input_dir.mkdir(parents=True)
    for frame_name in ("001.jpg", "002.jpg"):
        Image.new("RGB", (200, 100), "white").save(input_dir / frame_name)

    global_filter = tmp_path / "global_filter_results.csv"
    global_filter.write_text(
        "video_name,frame_name,is_representative,representative_frame,similarity\n"
        "L99_V001,001.jpg,True,001.jpg,1.0\n"
        "L99_V001,002.jpg,False,001.jpg,0.99\n",
        encoding="utf-8",
    )
    requested_frame_ids = []

    class FakeExtractor:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def batch_paths(image_paths, **_kwargs):
            return [list(image_paths)]

        def extract_batch(self, image_paths, **_kwargs):
            requested_frame_ids.extend(frame_id_from_path(path) for path in image_paths)
            return [
                CompactVisualRecord(
                    frame_id=frame_id_from_path(path),
                    caption="Representative frame.",
                    detailed_caption="Metadata produced once for the representative.",
                )
                for path in image_paths
            ]

    monkeypatch.setattr(gemini_visual_extractor, "GeminiVisualExtractor", FakeExtractor)
    output_dir = tmp_path / "metadata" / "caption"

    summary = run_extraction(
        input_dir.parent,
        output_dir,
        api_key="test-key",
        global_filter_results_path=global_filter,
        request_budget_state_path=tmp_path / "request_budget.json",
    )

    assert requested_frame_ids == ["L99_V001_f0001"]
    assert summary == {
        "processed": 2,
        "api_frames": 1,
        "total": 2,
        "batches": 1,
        "daily_limit_reached": False,
        "daily_quota_reported": False,
    }
    records = json.loads((output_dir / "L99" / "L99_V001.json").read_text())
    assert [record["frame_id"] for record in records] == [
        "L99_V001_f0001",
        "L99_V001_f0002",
    ]
    assert records[0]["caption"] == records[1]["caption"] == "Representative frame."


def test_daily_request_arguments_do_not_stop_local_extraction(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "keyframes" / "L88_V001"
    input_dir.mkdir(parents=True)
    for frame_name in ("001.jpg", "002.jpg"):
        Image.new("RGB", (200, 100), "white").save(input_dir / frame_name)
    requested_frame_ids = []

    class FakeExtractor:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def batch_paths(image_paths, batch_size, **_kwargs):
            return [[path] for path in image_paths] if batch_size == 1 else [list(image_paths)]

        def extract_batch(self, image_paths, **_kwargs):
            requested_frame_ids.extend(frame_id_from_path(path) for path in image_paths)
            return [
                CompactVisualRecord(
                    frame_id=frame_id_from_path(path),
                    caption="A test frame.",
                    detailed_caption="A test image used to verify the daily budget.",
                )
                for path in image_paths
            ]

    monkeypatch.setattr(gemini_visual_extractor, "GeminiVisualExtractor", FakeExtractor)
    summary = run_extraction(
        input_dir.parent,
        tmp_path / "metadata" / "caption",
        api_key="test-key",
        batch_size=1,
        daily_request_limit=1,
        global_filter_results_path=None,
        request_budget_state_path=tmp_path / "request_budget.json",
    )

    assert requested_frame_ids == ["L88_V001_f0001", "L88_V001_f0002"]
    assert summary["daily_limit_reached"] is False
    assert summary["api_frames"] == 2
