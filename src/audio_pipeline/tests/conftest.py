# conftest.py — pytest root configuration
# Đặt trong thư mục tests/ để pytest tự thêm src/audio_pipeline vào sys.path
import sys
import os

# Thêm thư mục cha (src/audio_pipeline) vào path để import config, extractor, v.v.
_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)
