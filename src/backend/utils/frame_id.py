<<<<<<< HEAD
import re

from backend.core.errors import FrameIdError
from backend.models.schemas import FrameContextItem, FrameContextResponse
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
    if current_index < radius:
        raise FrameIdError(f"frame_id must have at least {radius} preceding frames")

    prefix = match.group("prefix")
    width = len(index_text)
    frames = []
    for offset in range(-radius, radius + 1):
        nearby_id = f"{prefix}{current_index + offset:0{width}d}"
        frames.append(
            FrameContextItem(
                frame_id=nearby_id,
                offset=offset,
                thumbnail_url=build_thumbnail_url(nearby_id, thumbnail_base_url),
            )
        )
    return FrameContextResponse(current_frame_id=frame_id, frames=frames)
=======
# Logic parse frame_id (ví dụ: vid05_f1024 -> video_name: vid05, frame_index: 1024)
>>>>>>> 700adb8d8c8d3eea7e3ce1e2131f80b56c6ab3c1
