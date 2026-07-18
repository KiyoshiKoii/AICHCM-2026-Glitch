"""Generate a reproducible acceptance report for all three Task 4 tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from elasticsearch_backend import (
        DEFAULT_ALIAS_NAME,
        DEFAULT_ELASTICSEARCH_URL,
        collapse_spatial_relations_for_index,
        create_client,
    )
    from migrate_metadata import write_json_atomically
    from schemas import FrameMetadata, validate_metadata_file
except ImportError:
    from .elasticsearch_backend import (
        DEFAULT_ALIAS_NAME,
        DEFAULT_ELASTICSEARCH_URL,
        collapse_spatial_relations_for_index,
        create_client,
    )
    from .migrate_metadata import write_json_atomically
    from .schemas import FrameMetadata, validate_metadata_file

SEMANTIC_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = SEMANTIC_DIR.parent.parent
DEFAULT_ENTITY_METADATA = SEMANTIC_DIR / "sample_frames" / "metadata_entities.json"
DEFAULT_SPATIAL_METADATA = SEMANTIC_DIR / "sample_frames" / "metadata_spatial.json"
DEFAULT_BM25_REPORT = REPOSITORY_DIR / "baseline_report_bm25_v0.json"
DEFAULT_ELASTICSEARCH_REPORT = (
    SEMANTIC_DIR / "baseline_report_elasticsearch_v4_filters.json"
)
DEFAULT_SPATIAL_REPORT = SEMANTIC_DIR / "spatial_benchmark_report_v1.json"
DEFAULT_ENTITY_BENCHMARK_REPORT = SEMANTIC_DIR / "entity_benchmark_report_v1.json"
DEFAULT_SCALE_REPORT = SEMANTIC_DIR / "scale_benchmark_report_10k_v1.json"
DEFAULT_ACCEPTANCE_REPORT = SEMANTIC_DIR / "task4_acceptance_report.json"
PLACEHOLDER_VALUES = {
    "none",
    "none mentioned",
    "not mentioned",
    "not specified",
    "n/a",
    "unknown",
}
INVERSE_PREDICATES = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "overlapping": "overlapping",
}


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required Task 4 artifact does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def audit_entities(records: list[FrameMetadata]) -> dict:
    processed = [record for record in records if record.processing.entity_model]
    entity_values = [
        value
        for record in records
        for values in (
            record.entities.locations,
            record.entities.objects,
            record.entities.actions,
            record.entities.colors,
        )
        for value in values
    ]
    placeholders = sorted(
        {value for value in entity_values if value.casefold() in PLACEHOLDER_VALUES}
    )
    nonempty = sum(
        bool(
            record.entities.locations
            or record.entities.objects
            or record.entities.actions
            or record.entities.colors
            or record.entities.time_of_day.value != "unknown"
            or record.entities.setting.value != "unknown"
        )
        for record in records
    )
    models = Counter(record.processing.entity_model for record in processed)
    prompts = Counter(record.processing.prompt_version for record in processed)
    return {
        "records": len(records),
        "processed_records": len(processed),
        "coverage": len(processed) / len(records),
        "nonempty_records": nonempty,
        "placeholder_values": placeholders,
        "models": dict(models),
        "prompt_versions": dict(prompts),
        "passed": len(processed) == len(records) and not placeholders,
    }


def audit_spatial(records: list[FrameMetadata]) -> dict:
    processed = [record for record in records if record.processing.detection_model]
    relation_counts = {
        record.frame_id: len(record.spatial_relations) for record in records
    }
    indexed_relation_counts = {
        record.frame_id: len(
            collapse_spatial_relations_for_index(record.spatial_relations)
        )
        for record in records
    }
    all_relations = [
        relation for record in records for relation in record.spatial_relations
    ]
    missing_inverse: list[dict] = []
    for record in records:
        triples = {
            (
                relation.subject_id,
                relation.predicate.value,
                relation.object_id,
            )
            for relation in record.spatial_relations
        }
        for subject_id, predicate, object_id in triples:
            inverse = (object_id, INVERSE_PREDICATES[predicate], subject_id)
            if inverse not in triples:
                missing_inverse.append(
                    {
                        "frame_id": record.frame_id,
                        "relation": [subject_id, predicate, object_id],
                    }
                )
    largest_frame = max(relation_counts, key=relation_counts.get)
    largest_count = relation_counts[largest_frame]
    total_relations = len(all_relations)
    indexed_relations = sum(indexed_relation_counts.values())
    largest_indexed_frame = max(indexed_relation_counts, key=indexed_relation_counts.get)
    largest_indexed_count = indexed_relation_counts[largest_indexed_frame]
    return {
        "records": len(records),
        "processed_records": len(processed),
        "coverage": len(processed) / len(records),
        "detections": sum(len(record.detections) for record in records),
        "relations": total_relations,
        "raw_instance_relations": total_relations,
        "indexed_query_relations": indexed_relations,
        "frames_with_relations": sum(count > 0 for count in relation_counts.values()),
        "largest_relation_frame": largest_frame,
        "largest_relation_count": largest_count,
        "largest_relation_share": (
            largest_count / total_relations if total_relations else 0.0
        ),
        "largest_indexed_relation_frame": largest_indexed_frame,
        "largest_indexed_relation_count": largest_indexed_count,
        "largest_indexed_relation_share": (
            largest_indexed_count / indexed_relations if indexed_relations else 0.0
        ),
        "missing_reciprocal_relations": missing_inverse,
        "models": dict(
            Counter(record.processing.detection_model for record in processed)
        ),
        "rule_versions": dict(
            Counter(record.processing.spatial_rule_version for record in processed)
        ),
        "passed": len(processed) == len(records) and not missing_inverse,
    }


def compare_text_quality(bm25_report: dict, elasticsearch_report: dict) -> dict:
    bm25_rows = {row["query_id"]: row for row in bm25_report["queries"]}
    elasticsearch_rows = {
        row["query_id"]: row for row in elasticsearch_report["queries"]
    }
    shared_ids = sorted(bm25_rows.keys() & elasticsearch_rows.keys())
    if not shared_ids:
        raise ValueError("BM25 and Elasticsearch reports have no shared query IDs")
    metric_names = ("recall", "reciprocal_rank", "ndcg")
    regressions = {}
    for metric in metric_names:
        baseline_mean = sum(
            bm25_rows[query_id]["metrics"]["5"][metric] for query_id in shared_ids
        ) / len(shared_ids)
        candidate_mean = sum(
            elasticsearch_rows[query_id]["metrics"]["5"][metric]
            for query_id in shared_ids
        ) / len(shared_ids)
        regressions[metric] = {
            "bm25": baseline_mean,
            "elasticsearch": candidate_mean,
            "delta": candidate_mean - baseline_mean,
        }
    maximum_allowed_regression = 0.01
    return {
        "shared_queries": len(shared_ids),
        "at_k": 5,
        "metrics": regressions,
        "maximum_allowed_regression": maximum_allowed_regression,
        "passed": all(
            item["delta"] >= -maximum_allowed_regression - 1e-12
            for item in regressions.values()
        ),
    }


def audit_filtered_queries(elasticsearch_report: dict) -> dict:
    filtered = [
        row for row in elasticsearch_report["queries"] if row.get("filters") is not None
    ]
    failures = [
        row["query_id"]
        for row in filtered
        if not row.get("filters_applied") or row.get("first_relevant_rank") != 1
    ]
    return {
        "cases": len(filtered),
        "applied": sum(bool(row.get("filters_applied")) for row in filtered),
        "rank1": sum(row.get("first_relevant_rank") == 1 for row in filtered),
        "query_ids": [row["query_id"] for row in filtered],
        "failures": failures,
        "passed": len(filtered) >= 2 and not failures,
    }


def audit_live_elasticsearch(
    url: str,
    alias: str,
    expected_index: str,
    expected_frame_ids: set[str],
) -> dict:
    client = create_client(url)
    if not client.ping():
        raise RuntimeError(f"Elasticsearch is not reachable at {url}")
    aliases = client.indices.get_alias(name=alias)
    targets = sorted(aliases.keys())
    response = client.search(
        index=alias,
        size=max(len(expected_frame_ids), 1),
        query={"match_all": {}},
        source=False,
    )
    actual_ids = {hit["_id"] for hit in response["hits"]["hits"]}
    analyzed = client.indices.analyze(
        index=alias,
        analyzer="kis_english",
        text="The runners are running",
    )
    tokens = [token["token"] for token in analyzed["tokens"]]
    alias_ok = targets == [expected_index]
    ids_ok = actual_ids == expected_frame_ids
    analyzer_ok = "the" not in tokens and "are" not in tokens and bool(tokens)
    return {
        "url": url,
        "version": client.info()["version"]["number"],
        "alias": alias,
        "alias_targets": targets,
        "expected_index": expected_index,
        "document_count": len(actual_ids),
        "missing_ids": sorted(expected_frame_ids - actual_ids),
        "unexpected_ids": sorted(actual_ids - expected_frame_ids),
        "analyzer_probe_tokens": tokens,
        "checks": {
            "alias_target": alias_ok,
            "exact_document_ids": ids_ok,
            "english_analyzer": analyzer_ok,
        },
        "passed": alias_ok and ids_ok and analyzer_ok,
    }


def build_report(
    entity_path: Path,
    spatial_path: Path,
    bm25_report_path: Path,
    elasticsearch_report_path: Path,
    spatial_report_path: Path,
    entity_benchmark_report_path: Path,
    scale_report_path: Path,
    live_elasticsearch: dict | None = None,
) -> dict:
    entity_records = validate_metadata_file(entity_path)
    spatial_records = validate_metadata_file(spatial_path)
    if {record.frame_id for record in entity_records} != {
        record.frame_id for record in spatial_records
    }:
        raise ValueError("Entity and spatial metadata frame IDs differ")
    bm25_report = _load_json(bm25_report_path)
    elasticsearch_report = _load_json(elasticsearch_report_path)
    spatial_benchmark = _load_json(spatial_report_path)
    entity_benchmark = _load_json(entity_benchmark_report_path)
    scale_benchmark = _load_json(scale_report_path)

    entity_audit = audit_entities(entity_records)
    spatial_audit = audit_spatial(spatial_records)
    text_quality = compare_text_quality(bm25_report, elasticsearch_report)
    filtered_queries = audit_filtered_queries(elasticsearch_report)
    latency = elasticsearch_report["latency_ms"]
    latency_check = {
        "p95_ms": latency["p95"],
        "maximum_ms": 50.0,
        "passed": latency["p95"] <= 50.0,
    }
    core_checks = {
        "entity_pipeline": entity_audit["passed"],
        "entity_visual_benchmark": entity_benchmark["acceptance"]["passed"],
        "spatial_schema_and_reciprocals": spatial_audit["passed"],
        "spatial_manual_benchmark": spatial_benchmark["acceptance"]["passed"],
        "text_quality_no_regression": text_quality["passed"],
        "filtered_queries_end_to_end": filtered_queries["passed"],
        "elasticsearch_latency": latency_check["passed"],
        "elasticsearch_scale_pilot": scale_benchmark["pilot_gate"]["passed"],
    }
    if live_elasticsearch is not None:
        core_checks["live_elasticsearch"] = live_elasticsearch["passed"]

    label_accuracy = spatial_benchmark["metrics"]["labels"][
        "accuracy_on_localized_objects"
    ]
    queryable_recall = spatial_benchmark["metrics"]["queryable_relations"][
        "recall"
    ]
    annotated_frames = spatial_benchmark["annotated_frames"]
    relation_evaluated_frames = spatial_benchmark["annotation_coverage"][
        "relation_evaluated_frames"
    ]
    entity_searchable_f1 = entity_benchmark["metrics"]["overall"][
        "searchable_entities_micro"
    ]["f1"]
    production_gates = {
        "spatial_label_accuracy_at_least_0_8": label_accuracy >= 0.8,
        "queryable_relation_recall_at_least_0_8": queryable_recall >= 0.8,
        "at_least_10_manually_annotated_spatial_frames": annotated_frames >= 10,
        "at_least_5_relation_evaluated_spatial_frames": (
            relation_evaluated_frames >= 5
        ),
        "entity_searchable_visual_f1_at_least_0_7": entity_searchable_f1 >= 0.7,
        "elasticsearch_10k_scale_pilot": scale_benchmark["pilot_gate"]["passed"],
        "target_video_distribution_evaluated": False,
    }
    research_complete = all(core_checks.values())
    production_ready = research_complete and all(production_gates.values())
    limitations = [
        *spatial_benchmark["acceptance"].get("warnings", []),
        *entity_benchmark["acceptance"].get("warnings", []),
    ]
    if annotated_frames < 10:
        limitations.append(
            f"Spatial gold set has {annotated_frames} frames; production gate requires "
            "at least 10 spatial-rich frames."
        )
    if relation_evaluated_frames < 5:
        limitations.append(
            f"Only {relation_evaluated_frames} spatial gold frames have exhaustive "
            "relation annotations; production gate requires at least 5."
        )
    if entity_searchable_f1 < 0.7:
        limitations.append(
            f"Visual-truth searchable-entity F1 is {entity_searchable_f1:.4f}; "
            "production gate requires at least 0.70."
        )
    limitations.extend(
        [
            (
                "Spatial label grounding rules were designed from the current sample; "
                "a held-out target-video evaluation is still required."
            ),
            (
                "The 10k Elasticsearch pilot uses deterministic synthetic replication; "
                "it validates infrastructure latency/throughput, not real-corpus relevance."
            ),
            "Target-video distribution and production concurrency are not evaluated.",
        ]
    )
    return {
        "task": "Task 4 - Advanced Technology R&D",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "production_ready"
            if production_ready
            else "research_complete_with_known_limitations"
            if research_complete
            else "acceptance_failed"
        ),
        "research_complete": research_complete,
        "production_ready": production_ready,
        "core_checks": core_checks,
        "production_gates": production_gates,
        "elasticsearch": {
            "text_quality": text_quality,
            "filtered_queries": filtered_queries,
            "latency": latency_check,
            "live": live_elasticsearch,
        },
        "entity_extraction": {
            "artifact": entity_audit,
            "visual_benchmark": entity_benchmark,
        },
        "spatial_reasoning": {
            "artifact": spatial_audit,
            "benchmark": spatial_benchmark,
        },
        "scale_benchmark": scale_benchmark,
        "limitations": limitations,
        "provenance": {
            "git": _git_state(),
            "artifacts": {
                str(path.resolve()): {"sha256": _sha256(path)}
                for path in (
                    entity_path,
                    spatial_path,
                    bm25_report_path,
                    elasticsearch_report_path,
                    spatial_report_path,
                    entity_benchmark_report_path,
                    scale_report_path,
                )
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate every Task 4 R&D deliverable and write one report."
    )
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITY_METADATA)
    parser.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL_METADATA)
    parser.add_argument("--bm25-report", type=Path, default=DEFAULT_BM25_REPORT)
    parser.add_argument(
        "--elasticsearch-report", type=Path, default=DEFAULT_ELASTICSEARCH_REPORT
    )
    parser.add_argument("--spatial-report", type=Path, default=DEFAULT_SPATIAL_REPORT)
    parser.add_argument(
        "--entity-benchmark-report",
        type=Path,
        default=DEFAULT_ENTITY_BENCHMARK_REPORT,
    )
    parser.add_argument("--scale-report", type=Path, default=DEFAULT_SCALE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACCEPTANCE_REPORT)
    parser.add_argument("--check-live-elasticsearch", action="store_true")
    parser.add_argument("--url", default=DEFAULT_ELASTICSEARCH_URL)
    parser.add_argument("--alias", default=DEFAULT_ALIAS_NAME)
    parser.add_argument("--expected-index", default="semantic_frames_v4")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        spatial_records = validate_metadata_file(args.spatial)
        live = (
            audit_live_elasticsearch(
                args.url,
                args.alias,
                args.expected_index,
                {record.frame_id for record in spatial_records},
            )
            if args.check_live_elasticsearch
            else None
        )
        report = build_report(
            args.entities,
            args.spatial,
            args.bm25_report,
            args.elasticsearch_report,
            args.spatial_report,
            args.entity_benchmark_report,
            args.scale_report,
            live_elasticsearch=live,
        )
        write_json_atomically(args.output, report, overwrite=True)
    except Exception as exc:
        parser.exit(status=1, message=f"Task 4 acceptance failed: {exc}\n")

    print(
        f"Task 4 acceptance: status={report['status']}, "
        f"research_complete={report['research_complete']}, "
        f"production_ready={report['production_ready']}"
    )
    print(f"Saved report to {args.output.resolve()}")
    if not report["research_complete"]:
        parser.exit(status=2, message="One or more Task 4 core checks failed.\n")


if __name__ == "__main__":
    main()
