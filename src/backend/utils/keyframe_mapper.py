import os
import csv
from functools import lru_cache

# Path to the data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAP_DIR = os.path.join(BASE_DIR, "data", "map-keyframes")

@lru_cache(maxsize=1024)
def _load_video_mapping(video_name: str) -> dict[int, int]:
    """
    Loads the keyframe mapping for a specific video.
    Returns a dict mapping `n` to `frame_idx`.
    """
    csv_path = os.path.join(MAP_DIR, f"{video_name}.csv")
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
        
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n = int(row["n"])
                frame_idx = int(row["frame_idx"])
                mapping[n] = frame_idx
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        
    return mapping

def get_true_frame_idx(video_name: str, n: int) -> int | None:
    """
    Gets the true original frame index for a given video and keyframe `n`.
    Returns None if mapping doesn't exist.
    """
    mapping = _load_video_mapping(video_name)
    return mapping.get(n)
