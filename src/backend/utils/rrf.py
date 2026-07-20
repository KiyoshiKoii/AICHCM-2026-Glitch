from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.models.schemas import SearchHit, UpstreamResult
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
) -> list[SearchHit]:
    """Fuse ranked lists with score = sum(1 / (k + rank))."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if limit <= 0:
        return []

    accumulators: dict[str, _Accumulator] = {}
    first_seen_counter = 0

    for source, results in rankings.items():
        seen_in_source: set[str] = set()
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
            item.rrf_score += 1.0 / (k + rank)
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
            rrf_score=item.rrf_score,
            thumbnail_url=build_thumbnail_url(item.frame_id, thumbnail_base_url),
            source_ranks=item.source_ranks,
            source_scores=item.source_scores,
            metadata=item.metadata,
        )
        for item in ordered
    ]
