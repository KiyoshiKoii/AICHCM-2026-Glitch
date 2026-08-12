from semantic_pipeline.retrieval.temporal_query_expander import _apply_concepts
from semantic_pipeline.retrieval.temporal_query_parser import fallback_concept_groups


def test_alias_expansion_cannot_drop_an_explicit_user_constraint() -> None:
    text = "linh vật Olympic Paris"
    groups = _apply_concepts(
        text,
        fallback_concept_groups(text),
        [
            {
                "source_terms": ["linh", "vật"],
                "aliases": ["linh vật", "mascot", "Phryge"],
            },
            {
                "source_terms": ["Olympic", "Paris"],
                "aliases": ["Olympic Paris", "Paris 2024"],
            },
        ],
    )

    assert ("linh vật", "mascot", "Phryge") in groups
    assert ("Olympic Paris", "Paris 2024") in groups


def test_invalid_expansion_keeps_each_source_term_as_a_requirement() -> None:
    text = "linh vật Olympic Paris"
    groups = _apply_concepts(
        text,
        fallback_concept_groups(text),
        [{"source_terms": ["Tokyo"], "aliases": ["unrelated"]}],
    )

    assert groups == fallback_concept_groups(text)
