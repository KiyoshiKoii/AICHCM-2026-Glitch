import pytest
import httpx
from unittest.mock import AsyncMock

from backend.services.query_analyzer import (
    GEMINI_QUERY_RESPONSE_SCHEMA,
    OllamaQueryParser,
    extract_model_text,
    parse_llm_json,
)
from backend.schemas.search import ParsedQuery
from backend.core.errors import LLMParserError

def test_parse_llm_json_valid():
    json_str = '''
    ```json
    {
        "visual_prompt": "a red car",
        "semantic_keywords": ["red", "car"]
    }
    ```
    '''
    parsed = parse_llm_json(json_str)
    assert parsed.visual_prompt == "a red car"
    assert parsed.semantic_keywords == ["red", "car"]

def test_parse_llm_json_invalid_schema():
    json_str = '{"visual_prompt": "a red car"}' # missing semantic_keywords
    with pytest.raises(LLMParserError):
        parse_llm_json(json_str)


def test_extract_model_text_reads_candidate_parts_when_text_is_empty():
    class Part:
        text = '{"visual_prompt":"a red car","semantic_keywords":["red","car"]}'

    class Content:
        parts = [Part()]

    class Candidate:
        content = Content()

    class Response:
        text = None
        candidates = [Candidate()]

    assert extract_model_text(Response()).startswith('{"visual_prompt"')


def test_extract_model_text_returns_empty_for_empty_response():
    class Response:
        text = None
        candidates = []

    assert extract_model_text(Response()) == ""


def test_gemini_response_schema_uses_only_supported_minimal_fields():
    assert GEMINI_QUERY_RESPONSE_SCHEMA == {
        "type": "OBJECT",
        "properties": {
            "visual_prompt": {"type": "STRING"},
            "semantic_keywords": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
        },
        "required": ["visual_prompt", "semantic_keywords"],
    }

@pytest.mark.asyncio
async def test_ollama_parser_fallback():
    # Mock a client that always raises an error
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.HTTPError("Network failure")
    
    parser = OllamaQueryParser(mock_client, "http://test", "model")
    
    # Even on HTTP error, it should fallback safely
    query = "xe màu đỏ"
    parsed = await parser.parse(query)
    
    assert parsed.visual_prompt == query
    assert parsed.semantic_keywords == [query]
