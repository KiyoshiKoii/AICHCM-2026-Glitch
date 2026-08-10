from semantic_pipeline.gemini.backfill_descriptions import apply_updates, build_prompt


def _record() -> dict:
    return {
        "frame_id": "L21_V001_f0039",
        "caption": "A person rides a blue scooter.",
        "detailed_caption": "A man wearing a light blue shirt and helmet rides a blue scooter.",
        "detailed_caption_vi": "Một người đàn ông mặc áo xanh nhạt và đội mũ bảo hiểm đi xe máy xanh.",
        "detections": [
            {
                "object_id": "man_0",
                "label": "man",
                "description": "",
                "description_vi": "",
                "attributes": [],
                "action": "",
            },
            {
                "object_id": "scooter_0",
                "label": "scooter",
                "description": "",
                "description_vi": "",
                "attributes": [],
                "action": "",
            },
        ],
    }


def test_build_prompt_includes_caption_and_only_missing_objects():
    prompt = build_prompt([_record()])

    assert "A person rides a blue scooter." in prompt
    assert "light blue shirt" in prompt
    assert '"object_id": "scooter_0"' in prompt


def test_apply_updates_fills_only_empty_detection_fields():
    records = [_record()]
    updated = apply_updates(
        records,
        {
            "frames": [
                {
                    "frame_id": "L21_V001_f0039",
                    "objects": [
                        {
                            "object_id": "man_0",
                            "description": "a man wearing a light blue shirt and helmet",
                            "description_vi": "người đàn ông mặc áo xanh nhạt và đội mũ bảo hiểm",
                            "attributes": ["light blue shirt", "helmet"],
                            "action": "riding a scooter",
                        },
                        {
                            "object_id": "scooter_0",
                            "description": "a blue scooter",
                            "description_vi": "xe máy màu xanh",
                            "attributes": ["blue"],
                            "action": "",
                        },
                    ],
                }
            ]
        },
    )

    assert updated == 2
    assert records[0]["detections"][0]["attributes"] == ["light blue shirt", "helmet"]
    assert records[0]["detections"][1]["description"] == "a blue scooter"
