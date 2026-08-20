import re
from bisect import bisect_left

from backend.core.errors import FrameIdError
from backend.schemas.frames import FrameContextItem, FrameContextData, FrameContextResponse
from backend.utils.keyframe_mapper import get_keyframe_ordinals, get_keyframe_position
from backend.utils.thumbnail import build_thumbnail_url

FRAME_ID_PATTERN = re.compile(r"^(?P<prefix>.+_f)(?P<index>\d+)$")


def _context_item(
    *,
    prefix: str,
    width: int,
    ordinal: int,
    video_name: str,
    thumbnail_base_url: str,
) -> FrameContextItem:
    frame_id = f"{prefix}{ordinal:0{width}d}"
    position = get_keyframe_position(video_name, ordinal)
    return FrameContextItem(
        frame_id=frame_id,
        thumbnail_url=build_thumbnail_url(frame_id, thumbnail_base_url),
        frame_index=position["frame_index"] if position else None,
        timestamp_ms=position["timestamp_ms"] if position else None,
        fps=position["fps"] if position else None,
    )


def build_frame_context(
    frame_id: str,
    *,
    thumbnail_base_url: str,
    radius: int | None = 5,
) -> FrameContextResponse:
    match = FRAME_ID_PATTERN.fullmatch(frame_id)
    if not match:
        raise FrameIdError("frame_id must end with '_f' followed by digits")
    if radius is not None and radius < 0:
        raise ValueError("radius must not be negative")

    index_text = match.group("index")
    current_index = int(index_text)

    prefix = match.group("prefix")
    video_name = prefix[:-2]
    width = len(index_text)
    available_ordinals = get_keyframe_ordinals(video_name)

    if available_ordinals:
        center_position = bisect_left(available_ordinals, current_index)
        before_start = 0 if radius is None else max(0, center_position - radius)
        before_ordinals = available_ordinals[before_start:center_position]
        after_start = center_position + 1 if (
            center_position < len(available_ordinals)
            and available_ordinals[center_position] == current_index
        ) else center_position
        after_end = None if radius is None else after_start + radius
        after_ordinals = available_ordinals[after_start:after_end]
    else:
        fallback_radius = radius if radius is not None else 5
        min_index = max(0, current_index - fallback_radius)
        before_ordinals = tuple(range(min_index, current_index))
        after_ordinals = tuple(
            range(current_index + 1, current_index + fallback_radius + 1)
        )

    before_frames = [
        _context_item(
            prefix=prefix,
            width=width,
            ordinal=ordinal,
            video_name=video_name,
            thumbnail_base_url=thumbnail_base_url,
        )
        for ordinal in before_ordinals
    ]
    after_frames = [
        _context_item(
            prefix=prefix,
            width=width,
            ordinal=ordinal,
            video_name=video_name,
            thumbnail_base_url=thumbnail_base_url,
        )
        for ordinal in after_ordinals
    ]

    return FrameContextResponse(
        status="success",
        data=FrameContextData(
            center_frame=_context_item(
                prefix=prefix,
                width=width,
                ordinal=current_index,
                video_name=video_name,
                thumbnail_base_url=thumbnail_base_url,
            ),
            before_frames=before_frames,
            after_frames=after_frames
        )
    )
