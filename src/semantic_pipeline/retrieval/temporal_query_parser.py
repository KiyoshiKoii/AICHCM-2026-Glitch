"""Parser for BTC temporal event questions (E1..En)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


EVENT_LINE_RE = re.compile(r"(?im)^\s*(E\d+)\s*[:：]\s*(.+?)\s*$")
FIRST_RE = re.compile(r"\b(dau tien|lan dau|first|earliest|begins?|start(?:s|ed)?|bat dau)\b")
LAST_RE = re.compile(r"\b(cuoi cung|last|latest|end(?:s|ed)?|final|hoan tat|hoan toan)\b")
CONTACT_RE = re.compile(r"\b(tiep xuc|cham|contact|touch|first contact)\b")
COMPLETE_RE = re.compile(r"\b(hoan toan|completely|fully|complete|completed|roi khoi|rời khỏi)\b")
VISIBLE_RE = re.compile(r"\b(thay|thay thay|visible|shown|see|seen|appear|appears|first visible)\b")


def fold_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").replace("đ", "d")


def query_tokens(value: str) -> set[str]:
    return {
        item
        for item in re.findall(r"[^\W_]+", fold_text(value))
        if len(item) > 1 and item not in {"e", "khoanh", "khac", "moment"}
    }


@dataclass(frozen=True)
class TemporalEventQuery:
    event_index: int
    source_label: str
    text: str
    temporal_operator: str
    required_anchor: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class ParsedTemporalQuery:
    shared_context: str
    events: tuple[TemporalEventQuery, ...]


def _anchor_for(text: str) -> tuple[str, str]:
    folded = fold_text(text)
    if LAST_RE.search(folded):
        return "last_complete", "last_complete"
    if CONTACT_RE.search(folded):
        return "first_contact", "first_contact"
    if VISIBLE_RE.search(folded):
        return "first_visible", "first_visible"
    if COMPLETE_RE.search(folded):
        return "complete", "state_complete"
    if FIRST_RE.search(folded):
        return "first", "action_start"
    return "representative", "representative"


def parse_temporal_query(query: str) -> ParsedTemporalQuery:
    normalized = " ".join(query.split()).strip()
    if not normalized:
        raise ValueError("temporal query must not be blank")
    matches = list(EVENT_LINE_RE.finditer(query))
    if not matches:
        operator, anchor = _anchor_for(normalized)
        return ParsedTemporalQuery(
            # A summary-only request is still a video-level query.  Preserve
            # it as shared context so the retriever can select the video from
            # its summary before choosing an evidence frame.
            shared_context=normalized,
            events=(TemporalEventQuery(1, "E1", normalized, operator, anchor, frozenset(query_tokens(normalized))),),
        )
    shared_context = " ".join(query[: matches[0].start()].split()).strip(" :;,-")
    events: list[TemporalEventQuery] = []
    for position, match in enumerate(matches, start=1):
        label, text = match.group(1).upper(), " ".join(match.group(2).split())
        operator, anchor = _anchor_for(text)
        events.append(
            TemporalEventQuery(position, label, text, operator, anchor, frozenset(query_tokens(text)))
        )
    return ParsedTemporalQuery(shared_context, tuple(events))
