# Qwen3-VL Embed visual retrieval

Qwen3-VL-Embedding-2B is the active text-to-keyframe retrieval model. It runs
inside the shared `aichcm2026` Conda environment; a separate Qwen virtualenv is
no longer required. Legacy CLIP vectors and their `kis_images` collection are
preserved for reference only and are not queried by the Visual Pipeline.

## Runtime assets

- Model files: `data/models/Qwen3-VL-Embedding-2B/`
- Image embeddings: `data/npy_features_qwen3_vl_2b/`
- Qdrant collection: `kis_images_qwen3_vl_2b_test`
- Qdrant URL: `http://127.0.0.1:6333`

The extractor validates frame count, filename order, duplicate frame indices,
embedding dimension, finite values, and L2 normalization before writing an
artifact manifest.

## Install and run

Install the project dependencies into the existing Conda environment:

```bash
conda activate aichcm2026
python -m pip install --upgrade --force-reinstall \
  "transformers==4.57.3" \
  "sentence-transformers>=5.4.0" \
  "qwen-vl-utils==0.0.14" \
  "av>=14.0.0"
```

After downloading model files and artifacts from Kaggle, create the Qwen index:

```bash
export PYTHONPATH=src
export QDRANT_URL=http://127.0.0.1:6333
export QWEN3_VL_EMBED_MODEL=data/models/Qwen3-VL-Embedding-2B
export QWEN3_VL_EMBED_FEATURE_DIR=data/npy_features_qwen3_vl_2b

python -m visual_pipeline.qwen3_embedding.database --all-videos --recreate
```

Start the Qwen visual service:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m visual_pipeline.server
```

The first query loads Qwen once. On a CUDA machine the service defaults to
float16; set `QWEN3_VL_EMBED_DEVICE=cpu` to keep Qwen on CPU instead.

## Evaluation file format

After manual frame annotation, save ranked results as JSONL and score them
without loading the model:

```json
{"query_id":"q01","query":"...","relevant":[{"video_id":"L21_V001","frame_id":"137.jpg"}],"hits":[{"video_id":"L21_V001","frame_id":"137.jpg","score":0.91}]}
```

```bash
PYTHONPATH=src python -m visual_pipeline.qwen3_embedding.evaluation results.jsonl
```
