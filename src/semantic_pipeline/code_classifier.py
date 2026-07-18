"""Deterministic SQL classification over caption and OCR output.

OCR answers "which characters are visible?"; this stage answers the separate
semantic question "does that text contain an SQL statement?". It is cheap to
rerun, testable offline, and independent of an Ollama entity prompt.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

try:
    from migrate_metadata import write_json_atomically
    from schemas import (
        DEFAULT_METADATA_PATH,
        CodeMetadata,
        FrameMetadata,
        load_metadata_file,
        validate_metadata_records,
    )
except ImportError:
    from .migrate_metadata import write_json_atomically
    from .schemas import (
        DEFAULT_METADATA_PATH,
        CodeMetadata,
        FrameMetadata,
        load_metadata_file,
        validate_metadata_records,
    )


CODE_CLASSIFIER_VERSION = "code-rules-v1"
DEFAULT_OUTPUT_PATH = DEFAULT_METADATA_PATH.with_name("metadata_code.json")

# Long phrases precede overlapping single-token patterns such as EXISTS.
SQL_PATTERNS: tuple[str, ...] = (
    "not exists",
    "not in",
    "group by",
    "order by",
    "insert into",
    "delete from",
    "create table",
    "alter table",
    "drop table",
    "select",
    "from",
    "where",
    "join",
    "having",
    "union",
    "exists",
    "distinct",
    "count",
    "null",
    "update",
    "set",
    "correlated",
)


def _normalise(text: str) -> str:
    normalised = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"\s+", " ", normalised).strip()


def _contains(text: str, phrase: str) -> bool:
    words = [re.escape(word) for word in phrase.split()]
    pattern = r"(?<![a-z0-9_])" + r"\s+".join(words) + r"(?![a-z0-9_])"
    return bool(re.search(pattern, text))


def _matched_patterns(*texts: str) -> list[str]:
    combined = " ".join(text for text in texts if text)
    return [pattern for pattern in SQL_PATTERNS if _contains(combined, pattern)]


def _is_select_template(text: str) -> bool:
    """Recognise pedagogical ``SELECT ... FROM`` templates, not a frame ID."""
    return bool(re.search(r"\bselect\s*(?:\.{3}|…|\.\s*\.\s*\.)\s*from\b", text))


def _statement_type(patterns: set[str]) -> str:
    if "select" in patterns and "from" in patterns:
        return "select"
    if "insert into" in patterns:
        return "insert"
    if "update" in patterns and "set" in patterns:
        return "update"
    if "delete from" in patterns:
        return "delete"
    if "create table" in patterns:
        return "create"
    if "alter table" in patterns:
        return "alter"
    if "drop table" in patterns:
        return "drop"
    return "unknown"


def classify_code(
    caption: str = "",
    ocr_text: str = "",
    ocr_text_raw: str = "",
) -> CodeMetadata:
    """Classify SQL from explicit syntax and retain the matched evidence."""
    sources = {
        "caption": _normalise(caption),
        "ocr_text": _normalise(ocr_text),
        "ocr_text_raw": _normalise(ocr_text_raw),
    }
    detected_patterns = _matched_patterns(*sources.values())
    matched = set(detected_patterns)
    # "set alias" is common explanatory prose in SELECT tutorials. Keep SET
    # as an indexed SQL pattern only when UPDATE is also present.
    patterns = [
        pattern
        for pattern in detected_patterns
        if pattern != "set" or "update" in matched
    ]

    # SELECT/FROM occurs in ordinary prose ("select an item from ..."). A third
    # SQL-specific signal avoids that false positive unless text says SQL.
    select_context = {
        "where",
        "join",
        "group by",
        "order by",
        "having",
        "union",
        "not exists",
        "not in",
        "exists",
        "distinct",
        "count",
    }
    explicit_sql = any(_contains(text, "sql") for text in sources.values())
    sql_statement = (
        ({"select", "from"} <= matched and (bool(matched & select_context) or explicit_sql))
        or "insert into" in matched
        or ({"update", "set"} <= matched)
        or "delete from" in matched
        or bool({"create table", "alter table", "drop table"} & matched)
    )
    if not sql_statement:
        return CodeMetadata(classifier_version=CODE_CLASSIFIER_VERSION)

    evidence: list[str] = []
    for source_name, source_text in sources.items():
        for pattern in patterns:
            if _contains(source_text, pattern):
                evidence.append(f"{source_name}:{pattern}")

    statement_type = _statement_type(matched)
    search_terms = ["sql", "sql code"]
    if statement_type != "unknown":
        search_terms.append(f"{statement_type} statement")
    if "null" in matched:
        search_terms.append("sql null")
    if "not in" in matched and "select" in matched:
        search_terms.append("not in subquery")
    if "correlated" in matched:
        search_terms.append("correlated subquery")
    if any(_is_select_template(text) for text in sources.values()):
        search_terms.extend(["sql query", "sql query template"])

    return CodeMetadata(
        language="sql",
        statement_type=statement_type,
        patterns=patterns,
        search_terms=search_terms,
        evidence=evidence,
        classifier_version=CODE_CLASSIFIER_VERSION,
    )


def classify_record(record: FrameMetadata) -> FrameMetadata:
    code = classify_code(record.caption, record.ocr_text, record.ocr_text_raw)
    return record.model_copy(update={"code": code})


def classify_records(records: list[FrameMetadata]) -> list[FrameMetadata]:
    return [classify_record(record) for record in records]


def classify_metadata_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    in_place: bool = False,
    force: bool = False,
) -> dict:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path and not in_place:
        raise ValueError("input and output are equal; pass --in-place explicitly")
    if in_place and input_path != output_path:
        raise ValueError("--in-place requires --output to equal --input")

    records = classify_records(validate_metadata_records(load_metadata_file(input_path)))
    write_json_atomically(
        output_path,
        [record.model_dump(mode="json") for record in records],
        overwrite=force or in_place,
    )
    sql_count = sum(record.code.language == "sql" for record in records)
    return {"total": len(records), "sql": sql_count, "unknown": len(records) - sql_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify SQL/code metadata from existing caption and OCR text."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Update --input atomically; --output is ignored.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output = args.input if args.in_place else args.output
    try:
        summary = classify_metadata_file(
            args.input,
            output,
            in_place=args.in_place,
            force=args.force,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.exit(status=1, message=f"Code classification failed: {exc}\n")
    print(
        "Code classification complete: "
        f"total={summary['total']}, sql={summary['sql']}, "
        f"unknown={summary['unknown']}, output={output.resolve()}"
    )


if __name__ == "__main__":
    main()
