import asyncio
import json
from json import JSONDecodeError
from typing import Any

import httpx
from pydantic import ValidationError

from backend.core.errors import LLMParserError
from backend.schemas.search import ParsedQuery

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

SYSTEM_PROMPT = """You are a query parser for an egocentric video retrieval system.
Convert a Vietnamese search query into exactly two English fields:
1. visual_prompt: a concise, natural description containing only visible people,
   objects, actions, setting, spatial relations, colors, and temporal cues.
2. semantic_keywords: 5-12 short English keywords or close synonyms useful for
   lexical/metadata retrieval.
Do not invent details. Follow the supplied JSON schema exactly."""

# Gemini's response-schema endpoint accepts a constrained Schema subset.  Do
# not pass Pydantic's generated schema here: ``extra='forbid'`` becomes
# ``additionalProperties: false``, which Gemini rejects with HTTP 400.
GEMINI_QUERY_RESPONSE_SCHEMA = {
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


def extract_model_text(response: Any) -> str:
    """Read text from Gemini's convenience property or raw candidate parts.

    Some Gemini responses expose ``response.text`` as ``None`` even though a
    candidate part contains text.  Treating that property as the only source
    caused valid parser responses to fall back to the original Vietnamese
    query, which CLIP cannot rank reliably.
    """

    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text.strip()
    return ""


def parse_llm_json(raw_content: str) -> ParsedQuery:
    """Extract and validate the first JSON object from an LLM response."""
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise LLMParserError("LLM returned an empty response")

    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    candidates: list[Any] = []
    try:
        candidates.append(json.loads(text))
    except JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                candidates.append(value)
                break
            except JSONDecodeError:
                continue

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            return ParsedQuery.model_validate(candidate)
        except ValidationError as exc:
            raise LLMParserError(f"LLM JSON does not match the required schema: {exc}") from exc

    raise LLMParserError("LLM response does not contain a valid JSON object")


class OllamaQueryParser:
    def __init__(self, client: httpx.AsyncClient, base_url: str, model: str) -> None:
        self._client = client
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model

    async def parse(self, query: str) -> ParsedQuery:
        schema = ParsedQuery.model_json_schema()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Parse this Vietnamese query. Return JSON only.\n"
                        f"JSON schema: {json.dumps(schema, ensure_ascii=False)}\n"
                        f"Query: {query}"
                    ),
                },
            ],
            "stream": False,
            "format": schema,
            "think": False,
            "options": {"temperature": 0},
        }
        try:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
            body = response.json()
            content = body["message"]["content"]
            return parse_llm_json(content)
        except Exception as exc:
            # Fallback on any error (network, parse, validation)
            print(f"[Warning] QueryParser fallback used due to: {exc}")
            return ParsedQuery(visual_prompt=query, semantic_keywords=[query])


class GeminiQueryParser:
    def __init__(self, api_key: str | None, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key) if api_key and genai else None

    async def parse(self, query: str) -> ParsedQuery:
        if not self.client:
            print("[Warning] GeminiQueryParser: No API key or genai lib found. Using fallback.")
            return ParsedQuery(visual_prompt=query, semantic_keywords=[query])

        schema = ParsedQuery.model_json_schema()
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Parse this Vietnamese query. Return JSON only.\n"
            f"JSON schema: {json.dumps(schema, ensure_ascii=False)}\n"
            f"Query: {query}"
        )
        try:
            for attempt in range(2):
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GEMINI_QUERY_RESPONSE_SCHEMA,
                        temperature=0.0,
                        max_output_tokens=512,
                    ),
                )
                response_text = extract_model_text(response)
                if response_text:
                    return parse_llm_json(response_text)
                if attempt == 0:
                    # Empty text is often a transient candidate-generation
                    # issue. Retry once, but do not create an unbounded request
                    # loop that could amplify quota usage.
                    await asyncio.sleep(0.25)
                    prompt += "\nReturn a non-empty JSON object now; do not return an empty response."
            raise LLMParserError("Gemini returned no text for query parsing")
        except Exception as exc:
            print(f"[Warning] GeminiQueryParser fallback used due to: {exc}")
            return ParsedQuery(visual_prompt=query, semantic_keywords=[query])

