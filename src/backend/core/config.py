from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
    )

    app_name: str = "AIC 2026 Online Serving"
    app_version: str = "0.1.0"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"

    dev1_base_url: str = "http://localhost:8001"
    dev1_text_path: str = "/search/text"
    dev1_image_path: str = "/search/image"
    dev2_base_url: str = "http://localhost:8002"
    dev2_text_path: str = "/search/text"

    upstream_top_k: int = Field(default=100, ge=20, le=1000)
    output_top_k: int = Field(default=20, ge=1, le=20)
    rrf_k: int = Field(default=60, gt=0)
    allow_partial_results: bool = True

    thumbnail_base_url: str = "http://localhost:8000/thumbnails"
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_image_bytes: int = Field(default=20 * 1024 * 1024, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
