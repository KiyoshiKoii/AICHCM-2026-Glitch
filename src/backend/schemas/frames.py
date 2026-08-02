from pydantic import BaseModel


class FrameContextItem(BaseModel):
    frame_id: str
    thumbnail_url: str


class FrameContextData(BaseModel):
    center_frame: FrameContextItem
    before_frames: list[FrameContextItem]
    after_frames: list[FrameContextItem]


class FrameContextResponse(BaseModel):
    status: str = "success"
    data: FrameContextData
