import json

import pytest
import httpx
from unittest.mock import AsyncMock

from backend.services.query_analyzer import (
    GEMINI_QUERY_RESPONSE_SCHEMA,
    OllamaQueryParser,
    SYSTEM_PROMPT,
    extract_model_text,
    normalise_scene_keywords,
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
    assert parsed.object_queries == []


def test_parse_llm_json_keeps_object_terms_bound_together():
    parsed = parse_llm_json(
        json.dumps(
            {
                "visual_prompt": "a bald man wearing a light blue shirt",
                "semantic_keywords": ["bald man", "light blue shirt"],
                "object_queries": [
                    {
                        "english_phrase": "bald man wearing a light blue shirt",
                        "vietnamese_phrase": "người đàn ông hói mặc áo sơ mi xanh nhạt",
                    }
                ],
                "ocr_queries": ["SẠT LỞ"],
                "program_queries": ["60 Giây Sáng"],
            },
            ensure_ascii=False,
        )
    )

    assert parsed.object_queries[0].english_phrase == "bald man wearing a light blue shirt"
    assert parsed.object_queries[0].vietnamese_phrase == "người đàn ông hói mặc áo sơ mi xanh nhạt"
    assert parsed.ocr_queries == ["SẠT LỞ"]
    assert parsed.program_queries == ["60 Giây Sáng"]

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
            "object_queries": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "english_phrase": {"type": "STRING"},
                        "vietnamese_phrase": {"type": "STRING"},
                    },
                    "required": ["english_phrase", "vietnamese_phrase"],
                },
            },
            "ocr_queries": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "program_queries": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
        },
        "required": [
            "visual_prompt",
            "semantic_keywords",
            "object_queries",
            "ocr_queries",
            "program_queries",
        ],
    }


def test_query_planner_prompt_forbids_guessed_context_and_keeps_object_actions():
    assert "Every keyword must be directly entailed by the user's query" in SYSTEM_PROMPT
    assert "Do not convert clothing into an occupation" in SYSTEM_PROMPT
    assert "Include every action explicitly requested for that object" in SYSTEM_PROMPT
    assert "never invent the head noun of an ambiguous Vietnamese phrase" in SYSTEM_PROMPT
    assert "object query for a group, count" in SYSTEM_PROMPT
    assert "Never produce \"collapsed roadside stall\"" in SYSTEM_PROMPT
    assert 'include the gender-neutral retrieval variant\n"group of people"' in SYSTEM_PROMPT
    assert 'Do NOT use the weaker,\noverlapping keyword "group of men"' in SYSTEM_PROMPT


def test_scene_keyword_normaliser_keeps_group_and_damaged_path_retrievable():
    parsed = ParsedQuery(
        visual_prompt="a group of men beside a collapsed roadside",
        semantic_keywords=[
            "group of men",
            "collapsed roadside",
            "broken path",
            "damaged embankment",
        ],
    )

    result = normalise_scene_keywords(
        "một nhóm người đàn ông đứng cạnh con đường bị sụp",
        parsed,
    )

    assert result.object_queries == []
    assert "group of men" not in result.semantic_keywords
    assert "group of people" in result.semantic_keywords
    assert "several men" in result.semantic_keywords
    assert "damaged pathway" in result.semantic_keywords
    assert "eroded pathway" in result.semantic_keywords
    assert "broken walkway" in result.semantic_keywords
    assert "erosion" in result.semantic_keywords
    assert "collapsed roadside" not in result.semantic_keywords

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
    assert parsed.object_queries == []
    assert parsed.ocr_queries == []
    assert parsed.program_queries == []
