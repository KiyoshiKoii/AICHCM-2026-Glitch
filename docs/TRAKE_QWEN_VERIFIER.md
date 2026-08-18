# Qwen temporal verifier for TRAKE

`QwenTemporalVerifier` is an optional second-stage verifier. It does **not**
embed every keyframe and it does not decide which video to retrieve.

1. Sparse summaries, captions and timeline events select a video and an event
   anchor.
2. The verifier samples only a bounded raw-video window around that anchor.
3. Qwen2.5-VL runs a coarse pass then a dense refine pass and returns one
   native-video timestamp as strict JSON.
4. The result replaces the sparse timestamp only when it respects the event
   boundary. For `first` / `last` events, Qwen cannot jump farther than five
   seconds from the retrieval anchor; otherwise the sparse answer is kept with
   a `verification.status: fallback` trace.

This makes the feature safe for actions, contact, state changes and completion
events while preserving existing text/visual retrieval behavior.

## Runtime profile

Default model: `Qwen/Qwen2.5-VL-3B-Instruct`, loaded in 4-bit.

- Coarse: at most 8 frames at 1 fps.
- Refine: at most 12 frames at 4 fps in a +/-1.5 second window.
- Input images are resized to a maximum side of 280 px. This is required on an
  8 GB GPU: raw HD frame sequences can allocate more than 13 GB of attention
  memory even when model weights are quantized.
- Results are cached at `data/processed/qwen_temporal_cache/` by model,
  candidate event, prompt and sampler settings.

Install the optional runtime once:

```powershell
python -m pip install accelerate bitsandbytes
```

## CLI pilot

```powershell
$env:PYTHONPATH = 'src'
python -m semantic_pipeline.retrieval.temporal_cli `
  "bike race on a city road with cyclists`nE1: first cyclists visible" `
  --batch-id L23 --video-id L23_V001 `
  --without-gemini-query-parser --qwen-verify --qwen-candidate-events 1
```

Use `verify_with_qwen: true` and `qwen_candidate_events` (1..5) in the
`/internal/search/temporal-events` request to enable the same behavior through
the internal API. The default remains off, so existing callers stay lightweight.

## Runtime-dense experiment

`DenseMotionTemporalVerifier` is the non-generative online alternative. It
sequentially decodes a selected raw-video event at 2 fps, batches frames through
FP16 CLIP, and uses adjacent-frame difference only to resolve a boundary inside
the high-semantic region. Vietnamese event text is translated locally with
`Helsinki-NLP/opus-mt-vi-en`; it is not added to the sparse retrieval query.

Enable it with `--dense-verify` in the CLI or `verify_with_dense: true` in the
internal API. Qwen and dense verification are mutually exclusive.

On the `L24_V002` four-event pilot, cold CLI latency was 42.6 seconds and warm
in-process latency was 13.75 seconds, compared with roughly 257 seconds for the
Qwen experiment. Dense refinement improved sub-keyframe resolution but did not
repair an incorrectly retrieved episode, and CLIP presence can lag the true
start of an action. Treat this mode as a fast boundary refiner, not a replacement
for candidate retrieval or a learned temporal action model.
