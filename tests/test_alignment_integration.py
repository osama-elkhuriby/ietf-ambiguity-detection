"""Synthetic full-timeline regression; no external models or services."""

import json

import pytest
from test_alignment_matching import document

from alignment.version_alignment.cli import run
from alignment.version_alignment.config import Config
from alignment.version_alignment.models import Alignment


def synthetic_root(tmp_path):
    root = tmp_path / "repo"
    metadata = root / "corpus/metadata"
    metadata.mkdir(parents=True)
    clauses = root / "corpus/processed/clauses_llm"
    clauses.mkdir(parents=True)
    versions = []
    for rev in range(3):
        doc = document(
            rev,
            [
                f"The client MUST send {rev + 1} messages before completing the handshake.",
                "The endpoint MUST verify the entire certificate chain before transmission.",
            ],
        )
        raw = root / f"source{rev}.txt"
        raw.write_text("synthetic raw source", encoding="utf-8")
        import hashlib

        sha = hashlib.sha256(raw.read_bytes()).hexdigest()
        payload = {
            "document_id": doc.document_id,
            "revision_id": doc.revision,
            "source_path": str(raw),
            "source_sha256": sha,
            "clauses": [c.model_dump() for c in doc.clauses],
        }
        (clauses / f"{doc.document_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        versions.append(
            {
                "rev": doc.revision,
                "submission_date": doc.publication_date.isoformat(),
                "local_txt_path": raw.name,
            }
        )
    (metadata / "drafts.json").write_text(
        json.dumps({"draft-ietf-tls-tls13": {"versions": versions}}), encoding="utf-8"
    )
    (metadata / "rfcs.json").write_text("{}", encoding="utf-8")
    return root


def test_full_synthetic_run_is_deterministic_and_exclusive(tmp_path):
    root = synthetic_root(tmp_path)
    outputs = [tmp_path / "run1", tmp_path / "run2"]
    reports = [run(root, o, Config(backend="lexical")) for o in outputs]
    assert all(r["pairs"] == 2 for r in reports)

    def rows(output):
        records = [
            Alignment.model_validate_json(x).model_dump(mode="json")
            for x in (output / "alignments.jsonl").read_text().splitlines()
        ]
        for r in records:
            r.pop("created_at")
        return records

    assert rows(outputs[0]) == rows(outputs[1])
    assert (
        json.loads((outputs[0] / "run_manifest.json").read_text())["status"]
        == "COMPLETE"
    )
    assert len(list((outputs[0] / "features").glob("*.jsonl"))) == 3
    assert (outputs[0] / "candidates.jsonl").is_file()
    with pytest.raises(FileExistsError):
        run(root, outputs[0], Config(backend="lexical"))


def test_failure_records_pair_stage_and_traceback(tmp_path, monkeypatch):
    root = synthetic_root(tmp_path)

    def fail(*args):
        raise RuntimeError("deliberate solver failure")

    monkeypatch.setattr("alignment.version_alignment.cli.select", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="deliberate solver"):
        run(root, output, Config(backend="lexical"))
    failure = json.loads((output / "failure.json").read_text())
    assert failure["stage"] == "global_selection"
    assert failure["pair"] == ["draft-ietf-tls-tls13-00", "draft-ietf-tls-tls13-01"]
    assert "deliberate solver failure" in failure["traceback"]
    assert failure["completed_pairs"] == 0
    assert not (output / "run_manifest.json").exists()


def test_rejects_unknown_or_nonconsecutive_pair(tmp_path):
    root = synthetic_root(tmp_path)
    with pytest.raises(ValueError, match="Unknown draft-ietf-tls-tls13"):
        run(
            root,
            tmp_path / "bad-family",
            Config(backend="lexical"),
            pair=("draft-dns-00", "draft-dns-01"),
        )
    with pytest.raises(ValueError, match="consecutive"):
        run(
            root,
            tmp_path / "bad-pair",
            Config(backend="lexical"),
            pair=("draft-ietf-tls-tls13-00", "draft-ietf-tls-tls13-02"),
        )
