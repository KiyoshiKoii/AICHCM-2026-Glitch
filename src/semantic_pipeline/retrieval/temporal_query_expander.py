"""Gemini query parsing for constraint-preserving temporal retrieval."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from semantic_pipeline.core.environment import get_env_value
from semantic_pipeline.retrieval.temporal_query_parser import (
    ParsedTemporalQuery,
    TemporalEventQuery,
    fallback_concept_groups,
    query_tokens,
)

try:  # Keep deterministic retrieval available without google-genai installed.
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - depends on optional local dependency.
    genai = None
    types = None


CONCEPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_terms": {"type": "array", "items": {"type": "string"}},
        "aliases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["source_terms", "aliases"],
}

TEMPORAL_QUERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "context_concepts": {"type": "array", "items": CONCEPT_SCHEMA},
        "event_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "event_index": {"type": "integer"},
                    "concepts": {"type": "array", "items": CONCEPT_SCHEMA},
                },
                "required": ["event_index", "concepts"],
            },
        },
    },
    "required": ["context_concepts", "event_concepts"],
}


SYSTEM_PROMPT = """You parse Vietnamese video-retrieval queries into non-negotiable concepts.
Every explicit entity, location, object, color, action, state, count, and relation must remain a
required constraint. Do not replace a specific requested object with a broader topic.

For each concept, return source_terms copied from the supplied query and aliases that preserve the
same meaning. You may add a well-known proper-name alias when it is unambiguous (for example,
the Paris 2024 mascot may include Phryge), but never invent a location, event, or object.
An alias must preserve the complete concept. For example, "con phố" and "đèn lồng" are not
complete aliases for "phố lồng đèn" because each drops a required part. Split source terms into
separate required concepts when each part has its own aliases. Keep generic news phrasing out of
concepts. Return JSON only."""


def _clean_strings(values: Any, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    return list(
        dict.fromkeys(
            " ".join(str(value).split())
            for value in values
            if isinstance(value, str) and " ".join(value.split())
        )
    )[:limit]


def _apply_concepts(
    source_text: str,
    fallback_groups: tuple[tuple[str, ...], ...],
    concepts: Any,
) -> tuple[tuple[str, ...], ...]:
    """Merge Gemini aliases while retaining every source-token fallback."""

    source_tokens = query_tokens(source_text)
    remaining = {
        folded: group[0]
        for group in fallback_groups
        if group
        for folded in query_tokens(group[0])
    }
    groups: list[tuple[str, ...]] = []
    if isinstance(concepts, list):
        for raw in concepts[:12]:
            if not isinstance(raw, dict):
                continue
            source_terms = _clean_strings(raw.get("source_terms"), limit=6)
            term_tokens = set().union(*(query_tokens(term) for term in source_terms)) if source_terms else set()
            # An expansion is valid only when it refers to actual user terms.
            if not term_tokens or not term_tokens <= source_tokens:
                continue
            aliases = [
                alias
                for alias in _clean_strings(raw.get("aliases"), limit=8)
                # A strict subset is only a partial mention, not an alias of
                # the complete concept. Zero-overlap translations and proper
                # names remain valid (for example linh vat -> mascot).
                if not ((query_tokens(alias) - {"con", "cai", "chiec"}) < term_tokens)
            ]
            values = list(dict.fromkeys([" ".join(source_terms), *aliases]))
            values = [value for value in values if query_tokens(value)]
            if not values:
                continue
            groups.append(tuple(values))
            for token in term_tokens:
                remaining.pop(token, None)
    groups.extend((surface,) for surface in remaining.values())
    return tuple(groups) if groups else fallback_groups


class GeminiTemporalQueryParser:
    """Expand strict retrieval concepts with at most one Gemini call/query."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else get_env_value("GEMINI_API_KEY")
        self.model = model or (
            get_env_value("GEMINI_TEMPORAL_QUERY_MODEL")
            or get_env_value("GEMINI_QUERY_MODEL")
            or get_env_value("GEMINI_VIDEO_SUMMARY_MODEL")
            or "gemini-3.5-flash-lite"
        )
        self.client = genai.Client(api_key=self.api_key) if self.api_key and genai else None
        self.mode = "gemini" if self.client and types else "deterministic_fallback"

    def parse(self, parsed: ParsedTemporalQuery) -> ParsedTemporalQuery:
        if self.client is None or types is None:
            return parsed
        payload = {
            "video_context": parsed.shared_context,
            "events": [
                {"event_index": event.event_index, "source_label": event.source_label, "text": event.text}
                for event in parsed.events
            ],
        }
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"{SYSTEM_PROMPT}\n\nInput:\n{json.dumps(payload, ensure_ascii=False)}",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_json_schema=TEMPORAL_QUERY_SCHEMA,
                ),
            )
            text = getattr(response, "text", None)
            plan = json.loads(text) if isinstance(text, str) and text.strip() else None
            if not isinstance(plan, dict):
                raise ValueError("Gemini temporal parser returned no JSON object")
            event_concepts = {
                int(item["event_index"]): item.get("concepts")
                for item in plan.get("event_concepts", [])
                if isinstance(item, dict) and isinstance(item.get("event_index"), int)
            }
            events = tuple(
                replace(
                    event,
                    required_concept_groups=_apply_concepts(
                        event.text,
                        event.required_concept_groups or fallback_concept_groups(event.text),
                        event_concepts.get(event.event_index),
                    ),
                )
                for event in parsed.events
            )
            return replace(
                parsed,
                events=events,
                context_concept_groups=_apply_concepts(
                    parsed.shared_context,
                    parsed.context_concept_groups or fallback_concept_groups(parsed.shared_context),
                    plan.get("context_concepts"),
                ),
            )
        except Exception:
            # Query-time LLM failure must not make existing deterministic
            # retrieval unavailable or multiply provider calls through retry.
            self.mode = "deterministic_fallback_after_llm_error"
            return parsed
