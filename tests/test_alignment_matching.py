import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from alignment.version_alignment.candidates import Candidate, candidate_graph
from alignment.version_alignment.config import Config
from alignment.version_alignment.matching import select
from alignment.version_alignment.models import Clause, Document
from alignment.version_alignment.review import records
from alignment.version_alignment.scoring import change_summary, score


def document(rev, texts, headings=None):
    identity = f"draft-ietf-tls-tls13-{rev:02}"
    clauses = []
    for i, text in enumerate(texts):
        clauses.append(
            Clause(
                clause_id=f"{identity}:c{i}",
                document_id=identity,
                revision_id=f"{rev:02}",
                index=i + 1,
                block_id=f"b{i}",
                block_type="paragraph",
                section_id=str(i if headings else 1),
                section_heading=headings[i] if headings else "Protocol",
                section_path=["1"],
                raw_text=text,
                normalized_text=text,
                classification="NORMATIVE",
                source_location={"line_start": i + 1},
                text_start=0,
                text_end=len(text),
            )
        )
    return Document(
        family_id="draft-ietf-tls-tls13",
        document_id=identity,
        revision=f"{rev:02}",
        publication_date=date(2020, 1, rev + 1),
        available_at=date(2020, 1, rev + 1),
        source_path="synthetic",
        source_sha256="a" * 64,
        input_sha256="b" * 64,
        input_path="synthetic",
        clauses=clauses,
    )


def align(a, b, config=None):
    config = config or Config(backend="lexical")
    candidates = candidate_graph(a, b, config)
    chosen, _ = select(candidates, config)
    return records(
        a, b, candidates, chosen, config, "2026-01-01T00:00:00Z", "synthetic"
    ), candidates


def test_assignment_beats_greedy():
    edges = [
        Candidate((i,), (j,), {"combined": s})
        for i, j, s in [(0, 0, 0.90), (0, 1, 0.89), (1, 0, 0.88)]
    ]
    chosen, _ = select(edges, Config())
    assert {(e.source, e.target) for e in chosen} == {((0,), (1,)), ((1,), (0,))}


def test_insert_delete_empty_and_exact():
    text = "The client MUST send the complete handshake before application traffic."
    a = document(0, [text, "A violet banana grows in the orchard."])
    b = document(
        1, [text, "Cryptographic signatures authenticate the certificate chain."]
    )
    rows, _ = align(a, b)
    assert {r.relation for r in rows} == {"UNCHANGED", "INSERTED", "DELETED"}
    empty, _ = align(document(0, []), document(1, []))
    assert empty == []


def test_split_and_merge():
    one = "The client MUST verify the server certificate. The endpoint MUST reject an invalid signature."
    parts = [
        "The client MUST verify the server certificate.",
        "The endpoint MUST reject an invalid signature.",
    ]
    rows, _ = align(
        document(0, [one]),
        document(1, parts),
        Config(backend="lexical", uncertain_margin=0),
    )
    assert any(r.relation == "SPLIT" for r in rows)
    rows, _ = align(
        document(0, parts),
        document(1, [one]),
        Config(backend="lexical", uncertain_margin=0),
    )
    assert any(r.relation == "MERGED" for r in rows)


def test_protected_changes_not_erased():
    a = document(0, ["The client MUST send 10 messages."]).clauses[0]
    b = document(1, ["The server MUST NOT send 20 messages."]).clauses[0]
    components = score(a, b, 1, 0.98, Config())
    assert components["number_consistency"] == 0
    assert components["actor_consistency"] == 0
    assert set(change_summary(a.raw_text, b.raw_text)) >= {
        "number",
        "actor",
        "modal",
        "negation",
    }


def test_move_and_renumbering():
    text = "The client MUST verify all handshake signatures before sending application data."
    a = document(0, [text, "Other information."], ["Handshake", "Other"])
    b = document(1, ["Other information.", text], ["Handshake", "Security"])
    rows, _ = align(a, b, Config(backend="lexical", uncertain_margin=0))
    assert any(r.relation == "MOVED" for r in rows)
    a = document(0, [text])
    b = document(1, [text])
    b.clauses[0].section_id = "9"
    assert align(a, b)[0][0].relation == "UNCHANGED"


def test_repeated_text_uncertain_and_semantic_components():
    a = document(0, ["Send the message.", "Send the message."])
    b = document(1, ["Send the message.", "Send the message."])
    rows, edges = align(a, b)
    assert not any(e.locked for e in edges)
    assert any(r.relation == "UNCERTAIN" for r in rows)
    semantic = candidate_graph(
        a, b, Config(), np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)
    )
    assert semantic[0].scores["semantic"] is not None


def test_generic_group_is_not_split():
    a = document(0, ["The client MUST send the message."])
    b = document(
        1, ["The client MUST send the message.", "The client MUST send the message."]
    )
    assert all(
        len(e.target) == 1 for e in candidate_graph(a, b, Config(backend="lexical"))
    )


def test_contract_rejects_duplicates_and_bad_spans():
    d = document(0, ["A clause."])
    payload = d.model_dump()
    payload["clauses"] *= 2
    with pytest.raises(ValueError):
        Document.model_validate(payload)
    payload = d.clauses[0].model_dump()
    payload["text_end"] = 0
    with pytest.raises(ValueError):
        Clause.model_validate(payload)


def test_candidate_ties_are_stable_across_python_hash_seeds():
    code = """
import sys
sys.path.insert(0, 'tests')
from test_alignment_matching import document
from alignment.version_alignment.config import Config
from alignment.version_alignment.candidates import candidate_graph
from alignment.version_alignment.io import digest
a = document(0, ['alpha beta gamma delta epsilon', 'alpha beta gamma delta', 'alpha beta gamma'])
b = document(1, [f'{t} unique' for t in ['alpha', 'beta', 'gamma', 'delta', 'epsilon']])
edges = candidate_graph(a, b, Config(backend='lexical', local_top_k=1, global_top_k=1, section_window=0, candidate_floor=0))
print(digest([(e.source, e.target, e.scores) for e in edges]))
"""
    hashes = []
    for seed in ("1", "2", "3"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        hashes.append(result.stdout.strip())
    assert len(set(hashes)) == 1
