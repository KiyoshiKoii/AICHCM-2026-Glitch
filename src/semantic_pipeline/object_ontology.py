"""Open Images hierarchy and bilingual query bridge for BTC objects."""

from __future__ import annotations

import csv
import json
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    from common import paths
except ImportError:  # pragma: no cover
    from src.common import paths


DEFAULT_LEXICON_PATH = Path(__file__).with_name("object_lexicon.json")


def normalize_term(value: str) -> str:
    value = " ".join(value.strip().casefold().split())
    decomposed = unicodedata.normalize("NFD", value.replace("đ", "d"))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


class ObjectOntology:
    def __init__(
        self,
        *,
        hierarchy_path: str | Path = paths.OPENIMAGES_HIERARCHY,
        descriptions_path: str | Path = paths.OPENIMAGES_CLASS_DESCRIPTIONS,
        lexicon_path: str | Path = DEFAULT_LEXICON_PATH,
        require_external: bool = False,
    ) -> None:
        self.hierarchy_path = Path(hierarchy_path)
        self.descriptions_path = Path(descriptions_path)
        self.lexicon_path = Path(lexicon_path)
        if require_external:
            missing = [
                str(path)
                for path in (self.hierarchy_path, self.descriptions_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "Missing Open Images ontology files: "
                    f"{missing}. Run scripts/download_openimages_metadata.py"
                )

        self.labels = self._load_descriptions(self.descriptions_path)
        self.parents = self._load_hierarchy(self.hierarchy_path)
        self.lexicon = self._load_lexicon(self.lexicon_path)
        for mid, entry in self.lexicon.items():
            self.labels.setdefault(mid, entry["label"])
        self._term_to_mids = self._build_reverse_lexicon()

    @staticmethod
    def _load_descriptions(path: Path) -> dict[str, str]:
        if not path.is_file():
            return {}
        labels: dict[str, str] = {}
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) >= 2 and row[0].startswith("/m/"):
                    labels[row[0]] = row[1].strip()
        return labels

    @staticmethod
    def _load_hierarchy(path: Path) -> dict[str, set[str]]:
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        parents: defaultdict[str, set[str]] = defaultdict(set)

        def visit(node: dict, parent_mid: str | None = None) -> None:
            mid = node.get("LabelName")
            if isinstance(mid, str) and parent_mid is not None:
                parents[mid].add(parent_mid)
            next_parent = mid if isinstance(mid, str) else parent_mid
            for child in node.get("Subcategory", []) or []:
                if isinstance(child, dict):
                    visit(child, next_parent)

        if isinstance(payload, dict):
            visit(payload)
        elif isinstance(payload, list):
            for root in payload:
                if isinstance(root, dict):
                    visit(root)
        else:
            raise ValueError(f"Invalid Open Images hierarchy JSON: {path}")
        return dict(parents)

    @staticmethod
    def _load_lexicon(path: Path) -> dict[str, dict]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing committed BTC object lexicon: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Object lexicon must contain a JSON object: {path}")
        cleaned: dict[str, dict] = {}
        for mid, entry in payload.items():
            if not mid.startswith("/m/") or not isinstance(entry, dict):
                raise ValueError(f"Invalid lexicon entry for {mid!r}")
            label = entry.get("label")
            english = entry.get("en", [])
            vietnamese = entry.get("vi", [])
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Lexicon entry {mid} is missing label")
            if not all(isinstance(values, list) for values in (english, vietnamese)):
                raise ValueError(f"Lexicon aliases must be lists for {mid}")
            aliases = [label, *english, *vietnamese]
            if not all(isinstance(value, str) and value.strip() for value in aliases):
                raise ValueError(f"Lexicon aliases must be non-empty strings for {mid}")
            cleaned[mid] = {
                "label": label.strip(),
                "en": list(dict.fromkeys(value.strip() for value in english)),
                "vi": list(dict.fromkeys(value.strip() for value in vietnamese)),
            }
        return cleaned

    def _build_reverse_lexicon(self) -> dict[str, set[str]]:
        reverse: defaultdict[str, set[str]] = defaultdict(set)
        for mid, label in self.labels.items():
            reverse[normalize_term(label)].add(mid)
        for mid, entry in self.lexicon.items():
            for term in (entry["label"], *entry["en"], *entry["vi"]):
                reverse[normalize_term(term)].add(mid)
        return dict(reverse)

    @lru_cache(maxsize=2048)
    def ancestors(self, mid: str) -> tuple[str, ...]:
        """Return all parent MIDs, nearest-first, with cycle protection."""
        result: list[str] = []
        seen = {mid}
        frontier = sorted(self.parents.get(mid, set()))
        while frontier:
            parent = frontier.pop(0)
            if parent in seen:
                continue
            seen.add(parent)
            result.append(parent)
            frontier.extend(
                candidate
                for candidate in sorted(self.parents.get(parent, set()))
                if candidate not in seen
            )
        return tuple(result)

    def label_for_mid(self, mid: str) -> str | None:
        return self.labels.get(mid)

    def aliases_for_mid(self, mid: str) -> set[str]:
        entry = self.lexicon.get(mid)
        if entry is None:
            label = self.labels.get(mid)
            return {label.casefold()} if label else set()
        return {
            value.casefold()
            for value in (entry["label"], *entry["en"], *entry["vi"])
        }

    def expand_for_index(self, mid: str) -> set[str]:
        """Expand a detected class upward only; never add descendants."""
        terms = self.aliases_for_mid(mid)
        for ancestor in self.ancestors(mid):
            terms.update(self.aliases_for_mid(ancestor))
        return terms

    def labels_for_index(self, mid: str) -> set[str]:
        """Compact canonical labels for persisted metadata/search projection."""
        return {
            label.casefold()
            for candidate in (mid, *self.ancestors(mid))
            if (label := self.label_for_mid(candidate)) is not None
        }

    def map_query_terms(self, terms: Iterable[str]) -> tuple[set[str], list[str]]:
        mids: set[str] = set()
        unmapped: list[str] = []
        for term in terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            if cleaned.startswith("/m/") and cleaned in self.labels:
                mids.add(cleaned)
                continue
            normalized = normalize_term(cleaned)
            candidates = [normalized]
            if normalized.endswith("ies") and len(normalized) > 3:
                candidates.append(normalized[:-3] + "y")
            if normalized.endswith("s") and len(normalized) > 1:
                candidates.append(normalized[:-1])
            if normalized.endswith("es") and len(normalized) > 2:
                candidates.append(normalized[:-2])
            matches: set[str] = set()
            for candidate in candidates:
                matches.update(self._term_to_mids.get(candidate, set()))
            if matches:
                mids.update(matches)
            else:
                unmapped.append(cleaned)
        return mids, unmapped


@lru_cache(maxsize=1)
def get_default_ontology() -> ObjectOntology:
    return ObjectOntology()


def ancestors(mid: str) -> list[str]:
    return list(get_default_ontology().ancestors(mid))


def expand_for_index(mid: str) -> set[str]:
    return get_default_ontology().expand_for_index(mid)


def map_query_terms(terms: Iterable[str]) -> tuple[set[str], list[str]]:
    return get_default_ontology().map_query_terms(terms)
