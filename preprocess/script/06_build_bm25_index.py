from __future__ import annotations

import argparse
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from common import DEFAULT_OUTPUT_DIR, ensure_dir, read_jsonl, tokenize, write_json

INDEX_VERSION = 3
TOKENIZER_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dependency-free BM25 index from stage-5 metadata documents. "
            "Very-high-document-frequency terms are pruned because they cannot "
            "discriminate videos and only distort document-length normalization."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "05_metadata_search_documents.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "06_metadata_bm25.pkl",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "06_bm25_summary.json",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--max-df-ratio",
        type=float,
        default=0.98,
        help="Prune terms occurring in more than this fraction of documents.",
    )
    return parser.parse_args()


def clean_video_id(value: Any) -> str:
    return str(value or "").strip()


def validate_parameters(k1: float, b: float, max_df_ratio: float) -> None:
    if k1 <= 0:
        raise ValueError("--k1 must be greater than 0")
    if not 0 <= b <= 1:
        raise ValueError("--b must be between 0 and 1")
    if not 0 < max_df_ratio <= 1:
        raise ValueError("--max-df-ratio must be in (0, 1]")


def main() -> None:
    args = parse_args()
    validate_parameters(args.k1, args.b, args.max_df_ratio)

    documents = read_jsonl(args.input_file)
    if not documents:
        raise RuntimeError(f"No search documents found: {args.input_file.resolve()}")

    issue_counts: Counter[str] = Counter()
    seen_video_ids: set[str] = set()
    duplicate_video_ids: list[str] = []
    empty_text_video_ids: list[str] = []
    empty_token_video_ids: list[str] = []
    search_text_version_counts: Counter[str] = Counter()

    raw_tokenized_docs: list[list[str]] = []
    valid_video_ids: list[str] = []

    for document in documents:
        video_id = clean_video_id(document.get("video_id"))
        text = str(document.get("text") or "").strip()

        if not video_id:
            issue_counts["missing_video_id"] += 1
        elif video_id in seen_video_ids:
            issue_counts["duplicate_video_id"] += 1
            duplicate_video_ids.append(video_id)
        else:
            seen_video_ids.add(video_id)

        if not text:
            issue_counts["empty_search_text"] += 1
            empty_text_video_ids.append(video_id or "<missing_video_id>")

        tokens = tokenize(text)
        if not tokens:
            issue_counts["empty_tokenized_document"] += 1
            empty_token_video_ids.append(video_id or "<missing_video_id>")

        search_text_version_counts[str(document.get("search_text_version"))] += 1
        raw_tokenized_docs.append(tokens)
        valid_video_ids.append(video_id)

    fatal_issue_keys = {
        "missing_video_id",
        "duplicate_video_id",
        "empty_search_text",
        "empty_tokenized_document",
    }
    fatal_issues = {
        key: issue_counts[key]
        for key in sorted(fatal_issue_keys)
        if issue_counts.get(key, 0) > 0
    }
    if fatal_issues:
        raise RuntimeError(
            "Cannot build BM25 index because stage-5 documents are invalid: "
            + ", ".join(f"{key}={count}" for key, count in fatal_issues.items())
        )

    total_docs = len(documents)
    raw_document_frequency: Counter[str] = Counter()
    raw_collection_frequency: Counter[str] = Counter()
    for tokens in raw_tokenized_docs:
        tf = Counter(tokens)
        raw_document_frequency.update(tf.keys())
        raw_collection_frequency.update(tf)

    pruned_terms = {
        term
        for term, df in raw_document_frequency.items()
        if df / total_docs > args.max_df_ratio
    }

    tokenized_docs = [
        [token for token in tokens if token not in pruned_terms]
        for tokens in raw_tokenized_docs
    ]

    empty_after_pruning = [
        valid_video_ids[index]
        for index, tokens in enumerate(tokenized_docs)
        if not tokens
    ]
    if empty_after_pruning:
        raise RuntimeError(
            "High-DF pruning emptied documents: " + ", ".join(empty_after_pruning[:20])
        )

    term_frequencies = [Counter(tokens) for tokens in tokenized_docs]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    raw_doc_lengths = [len(tokens) for tokens in raw_tokenized_docs]
    total_tokens = sum(doc_lengths)
    raw_total_tokens = sum(raw_doc_lengths)
    avg_doc_length = total_tokens / total_docs

    document_frequency: Counter[str] = Counter()
    collection_frequency: Counter[str] = Counter()
    for tf in term_frequencies:
        document_frequency.update(tf.keys())
        collection_frequency.update(tf)

    idf = {
        term: math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
        for term, df in document_frequency.items()
    }

    video_id_to_doc_index = {
        video_id: index for index, video_id in enumerate(valid_video_ids)
    }

    index: dict[str, Any] = {
        "version": INDEX_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "k1": args.k1,
        "b": args.b,
        "max_df_ratio": args.max_df_ratio,
        "pruned_terms": sorted(pruned_terms),
        "documents": documents,
        "term_frequencies": term_frequencies,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "idf": idf,
        "document_frequency": document_frequency,
        "video_id_to_doc_index": video_id_to_doc_index,
        "source": {
            "input_file": str(args.input_file),
            "document_count": total_docs,
            "search_text_version_counts": dict(sorted(search_text_version_counts.items())),
        },
    }

    ensure_dir(args.output_file.parent)
    ensure_dir(args.summary_file.parent)

    with args.output_file.open("wb") as file:
        pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)

    most_common_document_terms = [
        {
            "term": term,
            "document_frequency": frequency,
            "document_rate": round(frequency / total_docs, 6),
        }
        for term, frequency in document_frequency.most_common(30)
    ]
    most_common_collection_terms = [
        {"term": term, "collection_frequency": frequency}
        for term, frequency in collection_frequency.most_common(30)
    ]
    pruned_term_details = [
        {
            "term": term,
            "document_frequency": raw_document_frequency[term],
            "document_rate": round(raw_document_frequency[term] / total_docs, 6),
            "collection_frequency": raw_collection_frequency[term],
        }
        for term in sorted(
            pruned_terms,
            key=lambda item: (-raw_document_frequency[item], item),
        )
    ]

    summary = {
        "index_version": INDEX_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "input_document_count": total_docs,
        "indexed_document_count": total_docs,
        "bm25_parameters": {
            "k1": args.k1,
            "b": args.b,
            "max_df_ratio": args.max_df_ratio,
        },
        "raw_vocabulary_size": len(raw_document_frequency),
        "indexed_vocabulary_size": len(idf),
        "raw_total_token_count": raw_total_tokens,
        "indexed_total_token_count": total_tokens,
        "raw_average_document_length": round(raw_total_tokens / total_docs, 4),
        "indexed_average_document_length": round(avg_doc_length, 4),
        "minimum_document_length": min(doc_lengths),
        "maximum_document_length": max(doc_lengths),
        "search_text_version_counts": dict(sorted(search_text_version_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "duplicate_video_ids": duplicate_video_ids,
        "empty_search_text_video_ids": empty_text_video_ids,
        "empty_tokenized_document_video_ids": empty_token_video_ids,
        "pruned_high_df_terms": pruned_term_details,
        "most_common_terms_by_document_frequency": most_common_document_terms,
        "most_common_terms_by_collection_frequency": most_common_collection_terms,
        "input_jsonl": str(args.input_file),
        "output_index": str(args.output_file),
    }
    write_json(args.summary_file, summary)

    print(f"Indexed {total_docs} documents")
    print(f"Raw vocabulary size: {len(raw_document_frequency)}")
    print(f"Indexed vocabulary size: {len(idf)}")
    print(f"Pruned high-DF terms: {len(pruned_terms)}")
    print(f"Average indexed document length: {avg_doc_length:.2f} tokens")
    print(f"Index: {args.output_file.resolve()}")
    print(f"Summary: {args.summary_file.resolve()}")


if __name__ == "__main__":
    main()
