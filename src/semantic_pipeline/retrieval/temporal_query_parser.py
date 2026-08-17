"""Parser for BTC temporal event questions (E1..En)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


EVENT_LINE_RE = re.compile(r"(?im)^\s*(E\d+)\s*[:：]\s*(.+?)\s*$")
FIRST_RE = re.compile(r"\b(dau tien|lan dau|first|earliest|begins?|start(?:s|ed)?|bat dau)\b")
START_RE = re.compile(r"\b(bat dau|begins?|start(?:s|ed)?)\b")
LAST_RE = re.compile(r"\b(cuoi cung|last|latest|end(?:s|ed)?|final|hoan tat|hoan toan)\b")
CONTACT_RE = re.compile(r"\b(tiep xuc|cham|contact|touch(?:es|ed|ing)?|first contact)\b")
COMPLETE_RE = re.compile(r"\b(hoan toan|completely|fully|complete|completed|roi khoi|rời khỏi)\b")
VISIBLE_RE = re.compile(r"\b(thay|thay thay|visible|shown|see|seen|appear(?:s|ed|ance)?|first visible)\b")
ACTION_COMPLETION_RE = re.compile(r"\b(hoan tat|het|complete|completed|roi khoi|poured out)\b")
FULL_STATE_RE = re.compile(r"\b(hoan toan|completely|fully)\b")
ACTION_HINT_RE = re.compile(
    r"\b(cut(?:s|ting)?|slic(?:e|es|ed|ing)|chop(?:s|ped|ping)?|cat|thai|"
    r"pour(?:s|ed|ing)?|stir(?:s|red|ring)?|khuay|fold(?:s|ed|ing)?|gap|"
    r"rotat(?:e|es|ed|ing)|quay)\b"
)
CONCEPT_STOPWORDS = frozenset(
    {
        "ban", "tin", "ve", "cua", "cho", "trong", "tai", "voi", "va",
        "nhung", "mot", "cac", "noi", "su", "kien", "doan", "video",
        "khoanh", "khac", "dau", "tien", "lan", "cuoi", "cung", "thay",
        "first", "last", "moment", "video", "news", "about", "the", "and",
    }
)


def fold_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").replace("đ", "d")


def query_tokens(value: str) -> set[str]:
    return {
        item
        for item in re.findall(r"[^\W_]+", fold_text(value))
        if len(item) > 1 and item not in {"e", "khoanh", "khac", "moment"}
    }


def fallback_concept_groups(value: str) -> tuple[tuple[str, ...], ...]:
    """Preserve each meaningful user term as an AND-style requirement.

    Gemini can later add aliases to these groups, but cannot silently remove
    an explicit constraint from the user's query. Surface accents are kept so
    Vietnamese concepts such as ``đèn``/``đen`` and ``lồng``/``lông`` do not
    collapse into the same retrieval constraint.
    """

    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    normalized = unicodedata.normalize("NFC", value.casefold())
    for token in re.findall(r"[^\W_]+", normalized):
        folded = fold_text(token)
        if len(token) <= 1 or folded in CONCEPT_STOPWORDS or folded in seen:
            continue
        seen.add(folded)
        groups.append((token,))
    return tuple(groups)


@dataclass(frozen=True)
class TemporalEventQuery:
    event_index: int
    source_label: str
    text: str
    temporal_operator: str
    required_anchor: str
    tokens: frozenset[str]
    required_concept_groups: tuple[tuple[str, ...], ...] = ()
    event_type: str = "action_event"
    selection_rule: str = "representative"
    retrieval_prompts: tuple[str, ...] = ()
    target_predicates: tuple[str, ...] = ()
    context_predicates: tuple[str, ...] = ()
    transition_required: bool = False


@dataclass(frozen=True)
class ParsedTemporalQuery:
    shared_context: str
    events: tuple[TemporalEventQuery, ...]
    context_concept_groups: tuple[tuple[str, ...], ...] = ()


def _semantics_for(text: str) -> tuple[str, str, str, str, bool]:
    folded = fold_text(text)
    if CONTACT_RE.search(folded):
        return "first_contact", "first_contact", "first_contact", "earliest_true", True
    if START_RE.search(folded):
        return "first", "action_start", "action_start", "earliest_true", True
    if ACTION_COMPLETION_RE.search(folded):
        return "last_complete", "last_complete", "action_completion", "latest_true", True
    if FULL_STATE_RE.search(folded):
        return "complete", "state_complete", "state_attainment", "earliest_true", True
    if VISIBLE_RE.search(folded) and not ACTION_HINT_RE.search(folded):
        return "first_visible", "first_visible", "first_appearance", "earliest_true", True
    if LAST_RE.search(folded):
        return "last", "last_complete", "last_occurrence", "latest_true", True
    if FIRST_RE.search(folded):
        return "first", "action_start", "action_start", "earliest_true", True
    return "representative", "representative", "action_event", "representative", False


def parse_temporal_query(query: str) -> ParsedTemporalQuery:
    normalized = " ".join(query.split()).strip()
    if not normalized:
        raise ValueError("temporal query must not be blank")
    matches = list(EVENT_LINE_RE.finditer(query))
    if not matches:
        operator, anchor, event_type, selection_rule, transition_required = _semantics_for(normalized)
        return ParsedTemporalQuery(
            # A summary-only request is still a video-level query.  Preserve
            # it as shared context so the retriever can select the video from
            # its summary before choosing an evidence frame.
            shared_context=normalized,
            events=(
                TemporalEventQuery(
                    1,
                    "E1",
                    normalized,
                    operator,
                    anchor,
                    frozenset(query_tokens(normalized)),
                    fallback_concept_groups(normalized),
                    event_type,
                    selection_rule,
                    (normalized,),
                    (normalized,),
                    (),
                    transition_required,
                ),
            ),
            context_concept_groups=fallback_concept_groups(normalized),
        )
    shared_context = " ".join(query[: matches[0].start()].split()).strip(" :;,-")
    events: list[TemporalEventQuery] = []
    for position, match in enumerate(matches, start=1):
        label, text = match.group(1).upper(), " ".join(match.group(2).split())
        operator, anchor, event_type, selection_rule, transition_required = _semantics_for(text)
        events.append(
            TemporalEventQuery(
                position,
                label,
                text,
                operator,
                anchor,
                frozenset(query_tokens(text)),
                fallback_concept_groups(text),
                event_type,
                selection_rule,
                (text,),
                (text,),
                (),
                transition_required,
            )
        )
    return ParsedTemporalQuery(
        shared_context,
        tuple(events),
        fallback_concept_groups(shared_context),
    )
