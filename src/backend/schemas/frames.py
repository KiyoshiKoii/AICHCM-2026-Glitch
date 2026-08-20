from pydantic import BaseModel


class FrameContextItem(BaseModel):
    frame_id: str
    thumbnail_url: str
    frame_index: int | None = None
    timestamp_ms: int | None = None
    fps: float | None = None


class FrameContextData(BaseModel):
    center_frame: FrameContextItem
    before_frames: list[FrameContextItem]
    after_frames: list[FrameContextItem]


class FrameContextResponse(BaseModel):
    status: str = "success"
    data: FrameContextData
