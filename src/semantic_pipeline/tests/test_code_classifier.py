"""Offline tests for deterministic SQL/code classification."""

import json

import pytest

from code_classifier import (
    CODE_CLASSIFIER_VERSION,
    classify_code,
    classify_metadata_file,
)


def test_classifies_select_not_exists_and_correlated_raw_evidence():
    result = classify_code(
        caption="A screenshot of code",
        ocr_text="SELECT * FROM A WHERE NOT EXISTS (SELECT * FROM B)",
        ocr_text_raw="This is a correlated subquery",
    )

    assert result.language == "sql"
    assert result.statement_type == "select"
    assert "not exists" in result.patterns
    assert "correlated subquery" in result.search_terms
    assert "ocr_text_raw:correlated" in result.evidence
    assert result.classifier_version == CODE_CLASSIFIER_VERSION


@pytest.mark.parametrize(
    ("text", "statement_type"),
    [
        ("INSERT INTO users(id) VALUES (1)", "insert"),
        ("UPDATE users SET active = 1", "update"),
        ("DELETE FROM users", "delete"),
        ("CREATE TABLE users (id INT)", "create"),
    ],
)
def test_classifies_other_sql_statements(text, statement_type):
    result = classify_code(ocr_text=text)
    assert result.language == "sql"
    assert result.statement_type == statement_type


def test_does_not_mistake_ordinary_select_from_prose_for_sql():
    result = classify_code(ocr_text="Select a dessert from the menu")
    assert result.language == "unknown"
    assert result.patterns == []
    assert result.classifier_version == CODE_CLASSIFIER_VERSION


def test_recognises_generic_select_template_without_frame_specific_rule():
    result = classify_code(ocr_text="SELECT ... FROM A WHERE condition")
    assert result.language == "sql"
    assert "sql query template" in result.search_terms


def test_derives_not_in_null_search_evidence_from_raw_ocr():
    result = classify_code(
        ocr_text="SELECT id FROM songs WHERE category NOT IN (SELECT id FROM genres)",
        ocr_text_raw="The subquery may contain NULL",
    )
    assert "sql null" in result.search_terms
    assert "not in subquery" in result.search_terms


def test_file_classification_requires_explicit_in_place(tmp_path):
    source = tmp_path / "metadata.json"
    source.write_text(
        json.dumps(
            [
                {
                    "frame_id": "vid03_f0004",
                    "video_name": "vid03.mp4",
                    "frame_index": 4,
                    "caption": "A code screenshot",
                    "ocr_text": "select * from A where not exists (select * from B)",
                    "ocr_text_raw": "correlated",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pass --in-place"):
        classify_metadata_file(source, source)

    summary = classify_metadata_file(source, source, in_place=True)
    saved = json.loads(source.read_text(encoding="utf-8"))[0]
    assert summary == {"total": 1, "sql": 1, "unknown": 0}
    assert saved["code"]["language"] == "sql"
    assert saved["code"]["classifier_version"] == CODE_CLASSIFIER_VERSION
