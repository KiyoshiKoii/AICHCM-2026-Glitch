# Internal API contract

Online Serving deliberately isolates upstream payload differences in
`src/backend/services/pipeline_clients.py`. The default contract is below.

## Dev 1 — visual text retrieval

`POST {DEV1_BASE_URL}{DEV1_TEXT_PATH}`

```json
{
  "query": "a person in a kitchen opening a refrigerator",
  "top_k": 100
}
```

## Dev 2 — semantic retrieval

`POST {DEV2_BASE_URL}{DEV2_TEXT_PATH}`

```json
{
  "query": "refrigerator fridge kitchen appliance opening",
  "keywords": ["refrigerator", "fridge", "kitchen", "appliance", "opening"],
  "top_k": 100
}
```

Both text APIs may return a bare array or an object under `results`, `items`,
`data`, or `hits`. Each result must have `frame_id` (or `id`), and may have
`score` (or `similarity`) plus `metadata`.

```json
{
  "results": [
    {
      "frame_id": "vid05_f1024",
      "score": 0.91,
      "metadata": {"video_id": "vid05"}
    }
  ]
}
```

Raw scores are retained for diagnostics, but ordering uses ranks only:

$$\operatorname{RRF}(d) = \sum_{s \in S_d}\frac{1}{k + \operatorname{rank}_s(d)}$$

The default is `k = 60`. Duplicate frame IDs within one source count once.

## Dev 1 — image retrieval

`POST {DEV1_BASE_URL}{DEV1_IMAGE_PATH}` receives the original upload as a
multipart field named `file`. Online Serving returns Dev 1's JSON response
without invoking the LLM or Dev 2.
