from typing import Any

import httpx

from backend.core.errors import UpstreamError
from backend.models.schemas import UpstreamResult


def normalize_upstream_results(payload: Any, source: str) -> list[UpstreamResult]:
    if isinstance(payload, dict):
        raw_results = next(
            (payload[key] for key in ("results", "items", "data", "hits") if key in payload),
            None,
        )
    else:
        raw_results = payload

    if not isinstance(raw_results, list):
        raise UpstreamError(f"{source} response must be a list or contain results/items/data/hits")

    normalized: list[UpstreamResult] = []
    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            raise UpstreamError(f"{source} result at index {index} is not an object")
        frame_id = item.get("frame_id") or item.get("id")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise UpstreamError(f"{source} result at index {index} has no valid frame_id")

        raw_score = item.get("score", item.get("similarity"))
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError) as exc:
            raise UpstreamError(f"{source} result at index {index} has an invalid score") from exc

        metadata = item.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise UpstreamError(f"{source} result at index {index} has invalid metadata")
        metadata = dict(metadata)
        for key, value in item.items():
            if key not in {"frame_id", "id", "score", "similarity", "metadata", "thumbnail_url"}:
                metadata.setdefault(key, value)

        normalized.append(UpstreamResult(frame_id=frame_id.strip(), score=score, metadata=metadata))
    return normalized


class InternalPipelineClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        source: str,
        base_url: str,
        text_path: str,
        image_path: str | None = None,
    ) -> None:
        self.source = source
        self._client = client
        self._text_url = f"{base_url.rstrip('/')}/{text_path.lstrip('/')}"
        self._image_url = f"{base_url.rstrip('/')}/{image_path.lstrip('/')}" if image_path else None

    async def search_text(self, payload: dict[str, Any]) -> Any:
        try:
            response = await self._client.post(self._text_url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UpstreamError(f"{self.source} text API failed: {exc}") from exc

    async def search_image(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> Any:
        if not self._image_url:
            raise UpstreamError(f"{self.source} does not expose an image API")
        try:
            response = await self._client.post(
                self._image_url,
                files={"file": (filename, content, content_type)},
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise UpstreamError(f"{self.source} image API failed: {exc}") from exc
