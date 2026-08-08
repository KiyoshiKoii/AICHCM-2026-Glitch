import json

from object_ontology import ObjectOntology


def ontology(tmp_path):
    hierarchy = tmp_path / "hierarchy.json"
    hierarchy.write_text(
        json.dumps(
            {
                "LabelName": "/m/entity",
                "Subcategory": [
                    {
                        "LabelName": "/m/vehicle",
                        "Subcategory": [{"LabelName": "/m/car"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptions = tmp_path / "labels.csv"
    descriptions.write_text(
        "/m/entity,Entity\n/m/vehicle,Vehicle\n/m/car,Car\n",
        encoding="utf-8",
    )
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps(
            {
                "/m/vehicle": {
                    "label": "Vehicle",
                    "en": ["vehicle"],
                    "vi": ["xe"],
                },
                "/m/car": {
                    "label": "Car",
                    "en": ["car", "automobile"],
                    "vi": ["ô tô"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ObjectOntology(
        hierarchy_path=hierarchy,
        descriptions_path=descriptions,
        lexicon_path=lexicon,
        require_external=True,
    )


def test_index_expands_only_upward(tmp_path):
    value = ontology(tmp_path)
    assert value.ancestors("/m/car") == ("/m/vehicle", "/m/entity")
    assert value.labels_for_index("/m/car") == {"car", "vehicle", "entity"}
    assert "car" not in value.expand_for_index("/m/vehicle")


def test_query_mapping_handles_vietnamese_diacritics_and_reports_unknown(tmp_path):
    value = ontology(tmp_path)
    mids, unmapped = value.map_query_terms(["o to", "unknown thing"])
    assert mids == {"/m/car"}
    assert unmapped == ["unknown thing"]


def test_query_mapping_handles_simple_english_plurals(tmp_path):
    value = ontology(tmp_path)
    mids, unmapped = value.map_query_terms(["cars"])
    assert mids == {"/m/car"}
    assert unmapped == []
