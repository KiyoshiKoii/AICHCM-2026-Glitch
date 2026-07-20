from urllib.parse import quote


def build_thumbnail_url(frame_id: str, base_url: str) -> str:
    encoded_frame_id = quote(frame_id, safe="")
    return f"{base_url.rstrip('/')}/{encoded_frame_id}.jpg"
