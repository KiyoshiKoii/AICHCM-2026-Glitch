import asyncio
import json
import re
import unicodedata
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

SYSTEM_PROMPT = """You are a query planner for a Vietnamese video retrieval system.
Convert the user's Vietnamese query into the JSON fields required by the supplied schema.

- visual_prompt: one concise, natural English visual description for CLIP. Keep it below 55
  English words and include only observable scene evidence requested by the user. Preserve
  important people, objects, actions, colors, count, and spatial arrangement. Never add a name,
  occupation, location, event, time of day, relationship, or other fact not explicitly requested
  or visually verifiable.
- semantic_keywords: 1-12 short English words or phrases for broad scene/caption retrieval.
  Prefer discriminative phrases such as "bald man blue shirt" over disconnected generic terms.
  Treat them as parallel retrieval variants, not one long sentence: preserve the requested scene,
  subjects, actions, and notable context with short caption-like phrases. You may include a small
  number of precise visual paraphrases that retain the same meaning, for example "damaged
  pathway", "broken concrete walkway", and "erosion" for a requested collapsed/eroded path.
  Every keyword must be directly entailed by the user's query or be such a meaning-preserving
  visual paraphrase. Never add a guessed setting, occupation, program, event, relationship, or
  other contextual synonym merely because it is common for the described objects. In particular,
  never invent the head noun of an ambiguous Vietnamese phrase: a damaged roadside area/path is
  not a "roadside stall", shop, tent, building, or vehicle unless the user explicitly says so.
  Do not convert clothing into an occupation: keep "person wearing blue medical scrubs" and never
  rewrite it as "medical worker" unless the user said so.
- object_queries: at most five requested objects that have distinguishing visual constraints.
  Each item must describe exactly ONE object in both English and natural Vietnamese. Keep every
  attribute, clothing detail, color, and action bound to that same object; never merge traits from
  multiple people. Include every action explicitly requested for that object. Do not create an
  object query for a group, count, relationship, relative position, scene, or vague background
  context: those belong only in semantic_keywords and visual_prompt because an object index entry
  represents one detected entity. Return [] if no ONE object has a useful distinguishing constraint.
- spatial_queries: at most five explicit subject-predicate-object relations. Use only
  left_of, right_of, above, below, or overlapping. Keep subject and object phrases separate;
  do not invent a relation that the user did not request. Return [] when no spatial relation
  is explicitly present.
- interaction_queries: at most five subject-action-object interactions. Use this when the
  query says that one entity performs an action involving another entity, such as riding a
  motorcycle, holding an umbrella, carrying a box, escorting a person, operating a screen,
  or feeding an animal. Keep attributes bound to the correct subject/object and keep the
  action separate. Do not emit an interaction when entities are merely nearby or both occur
  in the scene. For example, "a person riding a blue motorcycle" becomes one interaction;
  "a person standing beside a blue motorcycle" belongs in spatial_queries instead. The
  interaction is semantic and does not require inventing a spatial predicate.
- ocr_queries: exact text, names, numbers, slogans, or signs the user expects to be visibly
  readable. Preserve the original spelling; do not translate or invent text.
- program_queries: explicit program, series, broadcaster, channel, or broadcast-slot constraints.
  Do not infer them from a visual scene.

Do not add synonyms that change a required constraint. Return JSON only and follow the schema
exactly.

Example for a scene-level query:
Input: "một nhóm người đàn ông đứng cạnh lối đi ven đường bị sụp"
Good semantic_keywords: ["group of people", "several men", "damaged pathway",
"broken concrete walkway", "erosion"]
Good object_queries: []
Never produce "collapsed roadside stall" unless the input explicitly mentions a stall, kiosk, or
shop. For a requested group of men or women, include the gender-neutral retrieval variant
"group of people" as well as a separate gender-specific variant such as "several men"; captions
often describe the same visible group with the generic word "people". Do NOT use the weaker,
overlapping keyword "group of men" for this case. For a damaged or collapsed walkway/path, use
the full variants "damaged pathway", "broken concrete walkway", and "erosion" when they preserve
the request. Do NOT replace them with the vague alternatives "collapsed roadside", "broken
ground", or "damaged path", and do not replace them with a guessed roadside object. The people as
a group and the damaged path are scene evidence, not a single object."""

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
        "spatial_queries": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "subject_english_phrase": {"type": "STRING"},
                    "subject_vietnamese_phrase": {"type": "STRING"},
                    "predicate": {"type": "STRING"},
                    "object_english_phrase": {"type": "STRING"},
                    "object_vietnamese_phrase": {"type": "STRING"},
                },
                "required": [
                    "subject_english_phrase",
                    "subject_vietnamese_phrase",
                    "predicate",
                    "object_english_phrase",
                    "object_vietnamese_phrase",
                ],
            },
        },
        "interaction_queries": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "subject_english_phrase": {"type": "STRING"},
                    "subject_vietnamese_phrase": {"type": "STRING"},
                    "action_english_phrase": {"type": "STRING"},
                    "action_vietnamese_phrase": {"type": "STRING"},
                    "object_english_phrase": {"type": "STRING"},
                    "object_vietnamese_phrase": {"type": "STRING"},
                },
                "required": [
                    "subject_english_phrase",
                    "subject_vietnamese_phrase",
                    "action_english_phrase",
                    "action_vietnamese_phrase",
                    "object_english_phrase",
                    "object_vietnamese_phrase",
                ],
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
        "spatial_queries",
        "interaction_queries",
        "ocr_queries",
        "program_queries",
    ],
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


def _fold_vietnamese(value: str) -> str:
    """Lowercase Vietnamese text without accents for narrow query-plan guards."""

    decomposed = unicodedata.normalize("NFD", value.casefold())
    without_accents = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"\s+", " ", without_accents.replace("đ", "d")).strip()


def normalise_scene_keywords(query: str, parsed: ParsedQuery) -> ParsedQuery:
    """Add high-recall variants where Gemini commonly over-specialises a scene.

    Captions describe a visible group as "people" much more consistently than
    "group of men". Likewise, reports of a collapsed road are captioned as a
    damaged/eroded pathway. These are scene concepts, not a property of one
    detection; keeping the guard here prevents a weak parser response from
    eliminating a relevant frame before it reaches later ranking stages.
    """

    folded = _fold_vietnamese(query)
    is_group = "nhom" in folded and "nguoi" in folded
    is_men = "dan ong" in folded
    is_women = "phu nu" in folded
    is_path = any(phrase in folded for phrase in ("duong", "loi di"))
    is_damage = any(phrase in folded for phrase in ("sup", "sat lo", "hu hong", "vo"))

    keywords = list(parsed.semantic_keywords)
    if is_group:
        # A group is scene-level evidence. "group of men" is overly specific
        # for our captions and gives unrelated male crowds too much BM25 score.
        keywords = [
            item
            for item in keywords
            if item.casefold() not in {"group of men", "group of women"}
        ]
        keywords.append("group of people")
        if is_men:
            keywords.append("several men")
        elif is_women:
            keywords.append("several women")

    if is_path and is_damage:
        # Do not retain generic alternatives that drown out the discriminative
        # caption terms below.
        keywords = [
            item
            for item in keywords
            if item.casefold()
            not in {
                "collapsed roadside",
                "broken ground",
                "broken path",
                "damaged path",
                "damaged embankment",
            }
        ]
        keywords.extend(
            ["damaged pathway", "eroded pathway", "broken walkway", "erosion"]
        )

    unique_keywords: list[str] = []
    seen_keywords: set[str] = set()
    for keyword in keywords:
        normalized = " ".join(keyword.split()).strip(" ,;")
        key = normalized.casefold()
        if normalized and key not in seen_keywords:
            seen_keywords.add(key)
            unique_keywords.append(normalized)

    object_queries = parsed.object_queries
    if is_group:
        object_queries = [
            item for item in object_queries if "group of" not in item.english_phrase.casefold()
        ]

    return parsed.model_copy(
        update={
            "semantic_keywords": unique_keywords[:12],
            "object_queries": object_queries,
        }
    )


_FALLBACK_ENTITY_TRANSLATIONS = {
    "xe may": "motorcycle",
    "mo to": "motorcycle",
    "o to": "car",
    "xe hoi": "car",
    "xe tai": "truck",
    "xe dap": "bicycle",
    "cai o": "umbrella",
    "o": "umbrella",
    "cai thung": "box",
    "thung": "box",
    "con cho": "dog",
    "con meo": "cat",
    "man hinh": "screen",
    "dien thoai": "phone",
}
_FALLBACK_COLOR_TRANSLATIONS = {
    "xanh": "blue",
    "do": "red",
    "vang": "yellow",
    "den": "black",
    "trang": "white",
    "xam": "gray",
}
_FALLBACK_ACTION_PATTERNS = (
    ("riding", re.compile(r"\b(?:chay|lai|di|cuoi)\s+(?P<object>.+)$")),
    ("holding", re.compile(r"\b(?:cam|nam|om)\s+(?P<object>.+)$")),
    ("carrying", re.compile(r"\b(?:mang|khieng|xach|be)\s+(?P<object>.+)$")),
    ("operating", re.compile(r"\b(?:dieu khien|van hanh)\s+(?P<object>.+)$")),
    ("escorting", re.compile(r"\b(?:ap giai|dan giai)\s+(?P<object>.+)$")),
    ("petting", re.compile(r"\b(?:vuot ve)\s+(?P<object>.+)$")),
    ("feeding", re.compile(r"\b(?:cho an)\s+(?P<object>.+)$")),
    ("towing", re.compile(r"\bkeo\s+(?P<object>.+)$")),
)


def _fallback_entity_phrase(phrase: str) -> tuple[str, list[str]]:
    """Map common Vietnamese entity words to generic English retrieval terms."""

    folded = _fold_vietnamese(phrase)
    entity = "object"
    for vietnamese, english in sorted(_FALLBACK_ENTITY_TRANSLATIONS.items(), key=lambda item: -len(item[0])):
        if vietnamese in folded:
            entity = english
            break
    attributes: list[str] = []
    for vietnamese, english in _FALLBACK_COLOR_TRANSLATIONS.items():
        if vietnamese in folded:
            attributes.append(english)
    if "ao" in folded:
        attributes.append("shirt")
    if "dan ong" in folded or "nam" in folded:
        entity = "person"
    elif "phu nu" in folded or "nu" in folded:
        entity = "person"
    elif "nguoi" in folded and entity == "object":
        entity = "person"
    return entity, list(dict.fromkeys(attributes))


def fallback_parse_query(query: str) -> ParsedQuery:
    """Keep retrieval useful when the remote LLM parser is unavailable."""

    folded = _fold_vietnamese(query)
    for action, pattern in _FALLBACK_ACTION_PATTERNS:
        match = pattern.search(folded)
        if not match:
            continue
        subject_vi = folded[: match.start()].strip(" ,;:.-")
        object_vi = match.group("object").strip(" ,;:.-")
        subject_type, subject_attributes = _fallback_entity_phrase(subject_vi)
        object_type, object_attributes = _fallback_entity_phrase(object_vi)
        subject_en = " ".join([*subject_attributes, subject_type]).strip()
        object_en = " ".join([*object_attributes, object_type]).strip()
        keywords = list(dict.fromkeys([subject_en, action, object_en, *subject_attributes, *object_attributes]))
        interaction = {
            "subject_english_phrase": subject_en,
            "subject_vietnamese_phrase": subject_vi,
            "action_english_phrase": action,
            "action_vietnamese_phrase": folded[match.start() : match.end()].strip(),
            "object_english_phrase": object_en,
            "object_vietnamese_phrase": object_vi,
        }
        return ParsedQuery(
            visual_prompt=f"{subject_en} {action} {object_en}",
            semantic_keywords=keywords,
            interaction_queries=[interaction],
        )
    return ParsedQuery(visual_prompt=query, semantic_keywords=[query])


def enrich_explicit_interactions(query: str, parsed: ParsedQuery) -> ParsedQuery:
    """Repair an LLM plan when Vietnamese explicitly states an interaction.

    The deterministic Vietnamese action/entity guard is deliberately applied
    even when the provider returned a non-empty interaction list: a provider
    can produce a syntactically valid but semantically wrong predicate (for
    example ``standing`` for ``chạy xe máy``).  For explicit local-language
    cues, the normalized fallback is the safer binding for lexical retrieval;
    the rest of the provider plan is retained.
    """

    fallback = fallback_parse_query(query)
    if not fallback.interaction_queries:
        return parsed
    keywords = list(dict.fromkeys([*parsed.semantic_keywords, *fallback.semantic_keywords]))
    return parsed.model_copy(
        update={
            "semantic_keywords": keywords[:20],
            "interaction_queries": fallback.interaction_queries,
        }
    )


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
            return enrich_explicit_interactions(
                query,
                normalise_scene_keywords(query, parse_llm_json(content)),
            )
        except Exception as exc:
            # Fallback on any error (network, parse, validation)
            print(f"[Warning] QueryParser fallback used due to: {exc}")
            return fallback_parse_query(query)


class GeminiQueryParser:
    def __init__(self, api_key: str | None, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key) if api_key and genai else None

    async def parse(self, query: str) -> ParsedQuery:
        if not self.client:
            print("[Warning] GeminiQueryParser: No API key or genai lib found. Using fallback.")
            return fallback_parse_query(query)

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
                        max_output_tokens=768,
                    ),
                )
                response_text = extract_model_text(response)
                if response_text:
                    return enrich_explicit_interactions(
                        query,
                        normalise_scene_keywords(query, parse_llm_json(response_text)),
                    )
                if attempt == 0:
                    # Empty text is often a transient candidate-generation
                    # issue. Retry once, but do not create an unbounded request
                    # loop that could amplify quota usage.
                    await asyncio.sleep(0.25)
                    prompt += "\nReturn a non-empty JSON object now; do not return an empty response."
            raise LLMParserError("Gemini returned no text for query parsing")
        except Exception as exc:
            print(f"[Warning] GeminiQueryParser fallback used due to: {exc}")
            return fallback_parse_query(query)

