import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from test_alignment_matching import align, document

from alignment.version_alignment.config import Config
from alignment.version_alignment.models import Alignment, Document, Evidence


def test_threshold_order():
    with pytest.raises(ValueError):
        Config(candidate_floor=0.9, match_floor=0.5)
    with pytest.raises(ValueError):
        Config(uncertain_margin=0.9, high_margin=0.1)
    with pytest.raises(ValueError):
        Config(semantic_weight=float("nan"))


def test_schema_matches_contract():
    schema = (
        Path(__file__).resolve().parents[1] / "alignment/version_alignment/schema.json"
    )
    assert (
        json.loads(schema.read_text())
        == TypeAdapter(Alignment | Document | Evidence).json_schema()
    )


def test_no_automatic_ambiguity_labels_or_invalid_cardinality():
    a, b = document(0, ["client MUST send"]), document(1, ["client MUST send"])
    row = align(a, b)[0][0].model_dump()
    row["ambiguity_label"] = "ambiguous"
    with pytest.raises(ValueError):
        Alignment.model_validate(row)
    row.pop("ambiguity_label")
    row["relation"] = "SPLIT"
    with pytest.raises(ValueError):
        Alignment.model_validate(row)
