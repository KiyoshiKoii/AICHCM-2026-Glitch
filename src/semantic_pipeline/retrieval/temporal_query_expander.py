"""Gemini query parsing for constraint-preserving temporal retrieval."""

from __future__ import annotations

import json
import re
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

EVENT_TYPES = frozenset(
    {
        "first_appearance",
        "action_start",
        "first_contact",
        "state_attainment",
        "action_completion",
        "last_occurrence",
        "compound_event",
        "action_event",
    }
)
SELECTION_RULES = frozenset({"earliest_true", "latest_true", "representative"})
EVENT_ANCHORS = {
    "first_appearance": "first_visible",
    "action_start": "action_start",
    "first_contact": "first_contact",
    "state_attainment": "state_complete",
    "action_completion": "last_complete",
    "last_occurrence": "last_complete",
    "compound_event": "action_start",
    "action_event": "representative",
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
                    "event_type": {"type": "string", "enum": sorted(EVENT_TYPES)},
                    "selection_rule": {"type": "string", "enum": sorted(SELECTION_RULES)},
                    "retrieval_prompts": {"type": "array", "items": {"type": "string"}},
                    "target_predicates": {"type": "array", "items": {"type": "string"}},
                    "context_predicates": {"type": "array", "items": {"type": "string"}},
                    "transition_required": {"type": "boolean"},
                },
                "required": [
                    "event_index",
                    "concepts",
                    "event_type",
                    "selection_rule",
                    "retrieval_prompts",
                    "target_predicates",
                    "context_predicates",
                    "transition_required",
                ],
            },
        },
    },
    "required": ["context_concepts", "event_concepts"],
}


SYSTEM_PROMPT = """You parse Vietnamese TRAKE video queries into grounded visual event plans.
Every explicit entity, location, object, color, action, state, count, and relation must remain a
required constraint. Do not replace a specific requested object with a broader topic or invent
details that the query does not support.

For each concept, return source_terms copied from the supplied query and aliases that preserve the
same meaning. You may add a well-known proper-name alias when it is unambiguous (for example,
the Paris 2024 mascot may include Phryge), but never invent a location, event, or object.
An alias must preserve the complete concept. For example, "con phố" and "đèn lồng" are not
complete aliases for "phố lồng đèn" because each drops a required part. Split source terms into
separate required concepts when each part has its own aliases. Keep generic news phrasing out of
concepts.

For each event also produce a visual plan:
- event_type describes the boundary: first_appearance, action_start, first_contact,
  state_attainment, action_completion, last_occurrence, compound_event, or action_event.
- selection_rule is earliest_true, latest_true, or representative. Preserve explicit first/last
  wording from the user.
- retrieval_prompts are 1-3 short English descriptions for locating the broad event region. Include
  the distinctive requested entities and the observable action, but omit temporal words such as
  first, begins, earliest, last, and finally.
- target_predicates are 1-3 short English positive visual states that must be true at the answer
  frame. Every predicate must independently distinguish the event from nearby non-event frames;
  do not split an action into static entity-presence fragments. Describe visible evidence, not
  intent or an assumed prior scene. For action_start use an observable action in progress (for
  example, "performers walking while moving the dragon"), never a static state such as "performers
  holding dragon poles". For first_contact describe the objects touching; for completion describe
  the visibly completed state.
- context_predicates contain only additional positive visual constraints explicitly supported by
  the query, such as the worker wearing black or facing the camera. Use an empty array when none.
- transition_required is true only when adjacent frames are needed to distinguish the requested
  boundary. Never generate a hypothetical before-scene and never use negation as a visual prompt.
Return JSON only."""


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


def _clean_visual_prompts(values: Any, *, limit: int) -> tuple[str, ...]:
    prompts = _clean_strings(values, limit=limit)
    cleaned = []
    for prompt in prompts:
        prompt = re.sub(
            r"(?i)\b(first|earliest|latest|last|finally|begins?|start(?:s|ed|ing)?)\b",
            " ",
            prompt,
        )
        prompt = " ".join(prompt.split()).strip(" .,:;-")
        if prompt:
            cleaned.append(prompt)
    return tuple(dict.fromkeys(cleaned))


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


def _apply_event_plan(event: TemporalEventQuery, raw: Any) -> TemporalEventQuery:
    """Apply a grounded Gemini plan without weakening explicit temporal constraints."""

    if not isinstance(raw, dict):
        return event
    event_type = str(raw.get("event_type", "")).strip()
    if event_type not in EVENT_TYPES:
        event_type = event.event_type
    proposed_selection = str(raw.get("selection_rule", "")).strip()
    if proposed_selection not in SELECTION_RULES:
        proposed_selection = event.selection_rule
    # The deterministic parser owns explicit first/last wording. Gemini may
    # only choose a rule when the source event was temporally unspecified.
    selection_rule = (
        event.selection_rule
        if event.selection_rule != "representative"
        else proposed_selection
    )
    retrieval_prompts = _clean_visual_prompts(raw.get("retrieval_prompts"), limit=3)
    target_predicates = _clean_visual_prompts(raw.get("target_predicates"), limit=3)
    context_predicates = _clean_visual_prompts(raw.get("context_predicates"), limit=4)
    transition_required = raw.get("transition_required")
    if not isinstance(transition_required, bool):
        transition_required = event.transition_required
    transition_required = bool(transition_required or event.transition_required)
    return replace(
        event,
        event_type=event_type,
        selection_rule=selection_rule,
        required_anchor=EVENT_ANCHORS.get(event_type, event.required_anchor),
        retrieval_prompts=retrieval_prompts or event.retrieval_prompts or (event.text,),
        target_predicates=target_predicates or event.target_predicates or (event.text,),
        context_predicates=context_predicates,
        transition_required=transition_required,
    )


def temporal_query_to_retrieval_spec(parsed: ParsedTemporalQuery) -> dict[str, Any]:
    """Serialize a parsed query for temporal video encoders and verifiers."""

    return {
        "video_context": parsed.shared_context,
        "events": [
            {
                "event_id": event.source_label,
                "event_index": event.event_index,
                "description": event.text,
                "event_type": event.event_type,
                "selection_rule": event.selection_rule,
                "retrieval_prompts": list(event.retrieval_prompts or (event.text,)),
                "target_predicates": list(event.target_predicates or (event.text,)),
                "context_predicates": list(event.context_predicates),
                "transition_required": event.transition_required,
                "required_anchor": event.required_anchor,
            }
            for event in parsed.events
        ],
    }


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
        self.last_error: str | None = None

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
            event_plans = {
                int(item["event_index"]): item
                for item in plan.get("event_concepts", [])
                if isinstance(item, dict) and isinstance(item.get("event_index"), int)
            }
            events = tuple(
                _apply_event_plan(
                    replace(
                        event,
                        required_concept_groups=_apply_concepts(
                            event.text,
                            event.required_concept_groups or fallback_concept_groups(event.text),
                            event_plans.get(event.event_index, {}).get("concepts"),
                        ),
                    ),
                    event_plans.get(event.event_index),
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
        except Exception as exc:
            # Query-time LLM failure must not make existing deterministic
            # retrieval unavailable or multiply provider calls through retry.
            self.mode = "deterministic_fallback_after_llm_error"
            self.last_error = f"{type(exc).__name__}: {exc}"[:500]
            return parsed
