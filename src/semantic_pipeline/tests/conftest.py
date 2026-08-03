"""Cấu hình chung cho test (Task 5).

Test chạy trên metadata.json thật (24 frame) — file này được commit lên git nên
ai clone repo về cũng chạy test được ngay, không cần chạy lại extractor.py.
"""

import sys
from pathlib import Path

import pytest

SEMANTIC_DIR = Path(__file__).resolve().parent.parent

# Cho phép `import database` / `import server` khi chạy pytest từ gốc repo.
sys.path.insert(0, str(SEMANTIC_DIR))


@pytest.fixture(scope="session")
def db():
    from database import TextDatabase

    return TextDatabase()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    import server

    with TestClient(server.app) as test_client:
        yield test_client
