def build_thumbnail_url(frame_id: str, base_url: str) -> str:
    """
    Builds the URL for the frame thumbnail.
    Assumes all thumbnails are JPEG format.
    """
    return f"{base_url.rstrip('/')}/{frame_id}.jpg"
