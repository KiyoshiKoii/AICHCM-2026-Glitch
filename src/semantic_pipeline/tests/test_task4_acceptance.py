"""Unit tests for Task 4 acceptance checks and regression comparisons."""

from code_classifier import classify_record
from schemas import FrameMetadata
from task4_acceptance import (
    audit_btc_object_fusion,
    audit_code_classification,
    audit_entities,
    audit_filtered_queries,
    audit_required_rank1_query,
    audit_spatial,
    compare_text_quality,
)


def test_btc_object_fusion_gate_uses_detector_entities_and_native_identity():
    record = FrameMetadata.model_validate(
        {
            "schema_version": "1.1",
            "frame_id": "L21_V001_f0261",
            "video_name": "L21_V001",
            "frame_index": 261,
            "keyframe_n": 3,
            "object_counts": {"car": 1},
            "entities": {"objects": ["car"]},
            "detections": [
                {
                    "object_id": "car_0",
                    "label": "car",
                    "mid": "/m/0k4j",
                    "label_source": "btc_detector",
                    "bbox": [0.1, 0.2, 0.5, 0.8],
                    "confidence": 0.9,
                }
            ],
        }
    )
    audit = audit_btc_object_fusion([record])
    assert audit["passed"] is True
    assert audit["entity_objects"]["f1"] == 1.0


def complete_record():
    return FrameMetadata.model_validate(
        {
            "frame_id": "vid01_f0001",
            "video_name": "vid01.mp4",
            "frame_index": 1,
            "caption": "A person is left of a car outdoors.",
            "entities": {
                "setting": "outdoor",
                "objects": ["person", "car"],
            },
            "detections": [
                {
                    "object_id": "person_0",
                    "label": "person",
                    "bbox": [0.1, 0.2, 0.3, 0.8],
                    "confidence": 0.5,
                },
                {
                    "object_id": "car_0",
                    "label": "car",
                    "bbox": [0.6, 0.2, 0.9, 0.8],
                    "confidence": 0.5,
                },
            ],
            "spatial_relations": [
                {
                    "subject_id": "person_0",
                    "subject_label": "person",
                    "predicate": "left_of",
                    "object_id": "car_0",
                    "object_label": "car",
                    "confidence": 0.4,
                },
                {
                    "subject_id": "car_0",
                    "subject_label": "car",
                    "predicate": "right_of",
                    "object_id": "person_0",
                    "object_label": "person",
                    "confidence": 0.4,
                },
            ],
            "processing": {
                "entity_model": "llama3.2:3b",
                "prompt_version": "entities-v1.3",
                "detection_model": "florence",
                "spatial_rule_version": "spatial-v1",
            },
        }
    )


def retrieval_report(score=1.0, filters=None, filters_applied=False):
    return {
        "latency_ms": {"p95": 5.0},
        "queries": [
            {
                "query_id": "q1",
                "first_relevant_rank": 1,
                "filters": filters,
                "filters_applied": filters_applied,
                "metrics": {
                    "5": {
                        "recall": score,
                        "reciprocal_rank": score,
                        "ndcg": score,
                    }
                },
            }
        ],
    }


def test_entity_and_spatial_artifact_audits_pass_complete_record():
    record = complete_record()
    assert audit_entities([record])["passed"] is True
    spatial = audit_spatial([record])
    assert spatial["passed"] is True
    assert spatial["detections"] == 2
    assert spatial["relations"] == 2
    assert spatial["indexed_query_relations"] == 2


def test_code_audit_requires_current_version_and_sql_target_evidence():
    record = FrameMetadata.model_validate(
        {
            "frame_id": "vid03_f0004",
            "video_name": "vid03.mp4",
            "frame_index": 4,
            "caption": "A code template",
            "ocr_text": "select ... from A where not exists (select * from B)",
            "ocr_text_raw": "correlated subquery",
        }
    )
    record = classify_record(record)
    audit = audit_code_classification([record])
    assert audit["passed"] is True
    assert audit["target_checks"]["sql_query_search_term"] is True

    stale = record.model_copy(
        update={
            "code": record.code.model_copy(update={"classifier_version": "old-rules"})
        }
    )
    assert audit_code_classification([stale])["passed"] is False

    tampered = record.model_copy(
        update={
            "code": record.code.model_copy(update={"search_terms": ["sql"]})
        }
    )
    assert audit_code_classification([tampered])["metadata_matches_classifier"] is False


def test_spatial_audit_detects_missing_inverse():
    record = complete_record()
    record = record.model_copy(update={"spatial_relations": record.spatial_relations[:1]})
    audit = audit_spatial([record])
    assert audit["passed"] is False
    assert audit["missing_reciprocal_relations"]


def test_shared_text_quality_allows_at_most_one_percent_regression():
    assert compare_text_quality(retrieval_report(), retrieval_report(0.99))["passed"]
    assert not compare_text_quality(retrieval_report(), retrieval_report(0.98))[
        "passed"
    ]


def test_filtered_query_audit_requires_two_applied_rank_one_cases():
    report = retrieval_report(
        filters={"setting": "outdoor"}, filters_applied=True
    )
    assert audit_filtered_queries(report)["passed"] is False
    report["queries"].append(
        {
            **report["queries"][0],
            "query_id": "q2",
            "filters": {"spatial_relations": []},
        }
    )
    assert audit_filtered_queries(report)["passed"] is True


def test_required_sql_query_must_be_unfiltered_and_rank_first():
    report = retrieval_report()
    row = report["queries"][0]
    row.update(
        {
            "query_id": "sql_query_template",
            "retrieved_frame_ids": ["vid03_f0004", "vid03_f0003"],
        }
    )
    assert audit_required_rank1_query(report)["passed"] is True

    row["filters"] = {"code_language": "sql"}
    assert audit_required_rank1_query(report)["passed"] is False
