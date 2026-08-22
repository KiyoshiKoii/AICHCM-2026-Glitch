from types import SimpleNamespace

from visual_pipeline.text_processing import (
    DEFAULT_CLIP_CONTEXT_LENGTH,
    DEFAULT_CLIP_TOKEN_OVERLAP,
    clip_context_length,
    prepare_clip_text_inputs,
)


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(overflow_to_sample_mapping=[0, 0])
        self.device = None

    def to(self, device: str):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self, tokenizer_limit: int = 77) -> None:
        self.tokenizer = SimpleNamespace(model_max_length=tokenizer_limit)
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return FakeInputs()


def test_prepare_clip_text_inputs_truncates_to_model_context_length() -> None:
    processor = FakeProcessor(tokenizer_limit=512)
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=77))
    )

    inputs = prepare_clip_text_inputs(processor, model, ["very long prompt"], "cuda")

    assert processor.kwargs == {
        "text": ["very long prompt"],
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": 77,
        "return_overflowing_tokens": True,
        "stride": DEFAULT_CLIP_TOKEN_OVERLAP,
    }
    assert inputs.device == "cuda"
    assert "overflow_to_sample_mapping" not in inputs


def test_clip_context_length_falls_back_to_tokenizer_then_default() -> None:
    model_without_limit = SimpleNamespace(config=SimpleNamespace(text_config=None))

    assert clip_context_length(model_without_limit, FakeProcessor(64)) == 64
    assert (
        clip_context_length(
            model_without_limit,
            SimpleNamespace(tokenizer=SimpleNamespace(model_max_length=10**30)),
        )
        == DEFAULT_CLIP_CONTEXT_LENGTH
    )
