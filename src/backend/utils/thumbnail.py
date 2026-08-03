def build_thumbnail_url(frame_id: str, base_url: str) -> str:
    """
    Builds the URL for the frame thumbnail.
    Assumes all thumbnails are JPEG format.
    """
    clean_id = frame_id[:-4] if frame_id.lower().endswith(".jpg") else frame_id
    return f"{base_url.rstrip('/')}/{clean_id}.jpg"
