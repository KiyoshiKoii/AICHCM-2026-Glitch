import re

from backend.core.errors import FrameIdError
from backend.schemas.frames import FrameContextItem, FrameContextData, FrameContextResponse
from backend.utils.thumbnail import build_thumbnail_url

FRAME_ID_PATTERN = re.compile(r"^(?P<prefix>.+_f)(?P<index>\d+)$")


def build_frame_context(
    frame_id: str,
    *,
    thumbnail_base_url: str,
    radius: int = 5,
) -> FrameContextResponse:
    match = FRAME_ID_PATTERN.fullmatch(frame_id)
    if not match:
        raise FrameIdError("frame_id must end with '_f' followed by digits")
    if radius < 0:
        raise ValueError("radius must not be negative")

    index_text = match.group("index")
    current_index = int(index_text)

    prefix = match.group("prefix")
    width = len(index_text)
    
    before_frames = []
    # Avoid negative indices for frames
    min_index = max(0, current_index - radius)
    for offset in range(min_index - current_index, 0):
        nearby_id = f"{prefix}{current_index + offset:0{width}d}"
        before_frames.append(
            FrameContextItem(
                frame_id=nearby_id,
                thumbnail_url=build_thumbnail_url(nearby_id, thumbnail_base_url),
            )
        )

    after_frames = []
    for offset in range(1, radius + 1):
        nearby_id = f"{prefix}{current_index + offset:0{width}d}"
        after_frames.append(
            FrameContextItem(
                frame_id=nearby_id,
                thumbnail_url=build_thumbnail_url(nearby_id, thumbnail_base_url),
            )
        )

    return FrameContextResponse(
        status="success",
        data=FrameContextData(
            center_frame=FrameContextItem(
                frame_id=frame_id,
                thumbnail_url=build_thumbnail_url(frame_id, thumbnail_base_url)
            ),
            before_frames=before_frames,
            after_frames=after_frames
        )
    )
