"""Versioned Elasticsearch mapping for Gemini visual metadata.

The index is deliberately lexical-first.  Captions are predominantly English,
while OCR, ticker text, and program metadata contain Vietnamese and exact
strings.  They therefore use separate analyzers and query clauses rather than
one opaque catch-all field.
"""

from __future__ import annotations

from copy import deepcopy


DEFAULT_INDEX_NAME = "semantic_frames_v1"
DEFAULT_ALIAS_NAME = "semantic_frames"


_DEFINITION = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "normalizer": {
                "lowercase_folded": {
                    "type": "custom",
                    "filter": ["lowercase", "asciifolding"],
                }
            },
            "filter": {
                "english_possessive": {"type": "stemmer", "language": "possessive_english"},
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"},
            },
            "analyzer": {
                "english_visual": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "english_possessive",
                        "lowercase",
                        "asciifolding",
                        "english_stop",
                        "english_stemmer",
                    ],
                },
                "folded_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                },
            },
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "frame_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "program_code": {"type": "keyword", "normalizer": "lowercase_folded"},
            "frame_number": {"type": "integer"},
            "caption": {"type": "text", "analyzer": "english_visual"},
            "detailed_caption": {"type": "text", "analyzer": "english_visual"},
            "ocr_text": {"type": "text", "analyzer": "folded_text"},
            "news_ticker_text": {"type": "text", "analyzer": "folded_text"},
            "video_title": {"type": "text", "analyzer": "folded_text"},
            "video_description": {"type": "text", "analyzer": "folded_text"},
            "video_keywords": {"type": "text", "analyzer": "folded_text"},
            "series_name": {"type": "keyword", "normalizer": "lowercase_folded"},
            "broadcast_slot": {"type": "keyword", "normalizer": "lowercase_folded"},
            "channel_name": {"type": "keyword", "normalizer": "lowercase_folded"},
            "source_network": {"type": "keyword", "normalizer": "lowercase_folded"},
            "episode_date": {"type": "date", "format": "yyyy-MM-dd"},
        },
    },
}


def index_definition() -> dict:
    """Return a mutable copy safe to pass into ``indices.create``."""

    return deepcopy(_DEFINITION)
