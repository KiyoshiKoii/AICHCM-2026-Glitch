from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.schemas.search import SearchHit, UpstreamResult
from backend.utils.thumbnail import build_thumbnail_url


@dataclass
class _Accumulator:
    frame_id: str
    first_seen: int
    rrf_score: float = 0.0
    source_ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[UpstreamResult]],
    *,
    k: int = 60,
    limit: int = 20,
    thumbnail_base_url: str,
    source_weights: Mapping[str, float] | None = None,
) -> list[SearchHit]:
    """Fuse ranked lists with score = sum(1 / (k + rank))."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if limit <= 0:
        return []
    if source_weights is not None and any(weight < 0 for weight in source_weights.values()):
        raise ValueError("source weights must be non-negative")

    accumulators: dict[str, _Accumulator] = {}
    first_seen_counter = 0

    for source, results in rankings.items():
        seen_in_source: set[str] = set()
        weight = (
            source_weights.get(source, 1.0)
            if source_weights is not None
            else (1.05 if source == "dev2" else 1.0)
        )
        if weight == 0:
            continue
        for rank, result in enumerate(results, start=1):
            if result.frame_id in seen_in_source:
                continue
            seen_in_source.add(result.frame_id)

            if result.frame_id not in accumulators:
                accumulators[result.frame_id] = _Accumulator(
                    frame_id=result.frame_id,
                    first_seen=first_seen_counter,
                )
                first_seen_counter += 1

            item = accumulators[result.frame_id]
            item.rrf_score += (1.0 / (k + rank)) * weight
            item.source_ranks[source] = rank
            item.source_scores[source] = result.score
            for key, value in result.metadata.items():
                item.metadata.setdefault(key, value)

    ordered = sorted(
        accumulators.values(),
        key=lambda item: (
            -item.rrf_score,
            min(item.source_ranks.values()),
            item.first_seen,
        ),
    )[:limit]

    return [
        SearchHit(
            frame_id=item.frame_id,
            score=item.rrf_score,
            thumbnail_url=build_thumbnail_url(item.frame_id, thumbnail_base_url),
            metadata=item.metadata,
        )
        for item in ordered
    ]
