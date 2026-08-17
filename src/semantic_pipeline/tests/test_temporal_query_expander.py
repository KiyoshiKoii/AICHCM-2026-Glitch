from semantic_pipeline.retrieval.temporal_query_expander import (
    _apply_concepts,
    _apply_event_plan,
    temporal_query_to_retrieval_spec,
)
from semantic_pipeline.retrieval.temporal_query_parser import (
    fallback_concept_groups,
    parse_temporal_query,
)


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


def test_event_plan_uses_target_predicate_without_inventing_before_prompt() -> None:
    parsed = parse_temporal_query("E1: first moment a spoon touches the sauce")
    event = parsed.events[0]

    planned = _apply_event_plan(
        event,
        {
            "event_type": "first_contact",
            "selection_rule": "earliest_true",
            "retrieval_prompts": ["a chef stirring sauce with a spoon"],
            "target_predicates": ["the spoon touches the sauce"],
            "context_predicates": ["a chef prepares sauce"],
            "transition_required": True,
        },
    )
    spec = temporal_query_to_retrieval_spec(
        parsed.__class__(shared_context="", events=(planned,))
    )

    assert planned.required_anchor == "first_contact"
    assert planned.target_predicates == ("the spoon touches the sauce",)
    assert "before_prompts" not in spec["events"][0]


def test_gemini_cannot_override_an_explicit_earliest_rule() -> None:
    event = parse_temporal_query("E1: first moment a handle rotates").events[0]

    planned = _apply_event_plan(
        event,
        {
            "event_type": "action_start",
            "selection_rule": "latest_true",
            "retrieval_prompts": ["a worker rotates a handle"],
            "target_predicates": ["the handle is rotating"],
            "context_predicates": [],
            "transition_required": True,
        },
    )

    assert planned.selection_rule == "earliest_true"


def test_partial_alias_cannot_replace_a_complete_compound_concept() -> None:
    text = "phố lồng đèn"
    groups = _apply_concepts(
        text,
        fallback_concept_groups(text),
        [
            {
                "source_terms": ["phố", "lồng", "đèn"],
                "aliases": ["con phố", "đèn lồng", "lantern street"],
            }
        ],
    )

    assert groups == (("phố lồng đèn", "lantern street"),)
