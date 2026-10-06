"""Read-only post-run audit of alignment completeness and provenance."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

from .inputs import timeline
from .io import digest, file_hash, read_jsonl
from .models import Alignment, Document
from .temporal import prediction_view


def content_tokenizer(path: Path):
    """Recover source IDs without serialized batch padding or truncation settings."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def validate_run(run: Path, root: Path, protected: Path | None = None) -> dict:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETE" or (run / "failure.json").exists():
        raise ValueError("Run is not successfully complete")
    for relative, expected in manifest["artifacts"].items():
        if file_hash(run / relative) != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    metadata = json.loads((root / "corpus/metadata/drafts.json").read_text())
    rfcs = json.loads((root / "corpus/metadata/rfcs.json").read_text())
    family_id = manifest["family_id"]
    manifest_ids = [entry["document_id"] for entry in manifest["documents"]]
    rfc_id = manifest_ids[-1] if manifest_ids[-1].upper().startswith("RFC") else None
    expected = timeline(metadata, rfcs, family_id, rfc_id)
    expected_ids = [entry["document_id"] for entry in expected]
    if expected_ids != manifest_ids:
        raise ValueError("Run manifest does not match the current family timeline")
    docs = [
        Document.model_validate_json(
            (run / "inputs" / f"{e['document_id']}.json").read_text(encoding="utf-8")
        )
        for e in expected
    ]
    if len(list((run / "inputs").glob("*.json"))) != len(docs):
        raise ValueError("Input snapshot count does not match the family timeline")
    prepared_expected = set(manifest["input_info"]["prepared_documents"])
    if {d.document_id for d in docs if d.prepared} != prepared_expected:
        raise ValueError("Unexpected generated input snapshots")
    rows = [Alignment.model_validate(r) for r in read_jsonl(run / "alignments.jsonl")]
    row_hashes = {r.record_id: digest(r) for r in rows}
    if len(row_hashes) != len(rows):
        raise ValueError("Duplicate alignment record IDs")
    candidate_count = 0
    pair_candidate_count = 0
    relation_counts = Counter(r.relation for r in rows)
    ids = {d.document_id for d in docs}
    pair_ids = {(a.document_id, b.document_id) for a, b in pairwise(docs)}
    if {(r.from_document_id, r.to_document_id) for r in rows} != pair_ids:
        raise ValueError("Incomplete or unexpected pair timeline")
    for a, b in pairwise(docs):
        folder = run / "pairs" / f"{a.document_id}__{b.document_id}"
        pair_rows = [
            Alignment.model_validate(r) for r in read_jsonl(folder / "alignments.jsonl")
        ]
        source = Counter(s for r in pair_rows for s in r.source_clause_ids)
        target = Counter(t for r in pair_rows for t in r.target_clause_ids)
        if source != Counter(c.clause_id for c in a.clauses) or target != Counter(
            c.clause_id for c in b.clauses
        ):
            raise ValueError(
                f"Missing or multiply assigned occurrence in {folder.name}"
            )
        for row in pair_rows:
            if digest(row) != row_hashes.get(row.record_id) or not row.lineage_ids:
                raise ValueError(f"Pair/consolidated lineage mismatch: {row.record_id}")
        count = sum(1 for _ in read_jsonl(folder / "candidates.jsonl"))
        if count != json.loads((folder / "report.json").read_text())["candidate_count"]:
            raise ValueError(f"Candidate count mismatch: {folder.name}")
        pair_candidate_count += count
    for row in read_jsonl(run / "candidates.jsonl"):
        if (row["from_document_id"], row["to_document_id"]) not in pair_ids:
            raise ValueError("Candidate outside family timeline")
        candidate_count += 1
    if candidate_count != pair_candidate_count:
        raise ValueError("Consolidated candidate count mismatch")
    review = list(read_jsonl(run / "review_queue.jsonl"))
    if {r["alignment"]["record_id"] for r in review} != {
        r.record_id for r in rows if r.review_required
    }:
        raise ValueError("Review queue does not match required relationships")
    sample = list(read_jsonl(run / "review_sample.jsonl"))
    expected_sample_size = min(manifest["config"]["review_sample_size"], len(rows))
    if len(sample) != expected_sample_size or len(
        {r["alignment"]["record_id"] for r in sample}
    ) != len(sample):
        raise ValueError("Review sample size or uniqueness mismatch")
    if any(r["reviewer"] is not None for r in sample):
        raise ValueError("Automatic sample must not claim human review")
    lineage = list(read_jsonl(run / "lineage.jsonl"))
    if Counter(n["clause_id"] for n in lineage) != Counter(
        c.clause_id for d in docs for c in d.clauses
    ):
        raise ValueError("Lineage does not account for every clause occurrence")
    if list(read_jsonl(run / "diagnostics.jsonl")):
        raise ValueError("Unresolved feature diagnostics")
    feature_count = 0
    for d in docs:
        if d.revision is None:
            continue
        prefix, _ = prediction_view(docs, rows, d.document_id)
        batch = list(read_jsonl(run / "features" / f"{d.document_id}.jsonl"))
        if Counter(r["clause_id"] for r in batch) != Counter(
            c.clause_id for c in d.clauses
        ):
            raise ValueError(f"Incomplete features: {d.document_id}")
        for row in batch:
            if (
                row["dependency_document_ids"] != [p.document_id for p in prefix]
                or row["available_at"] > d.publication_date.isoformat()
            ):
                raise ValueError(f"Future feature dependency: {d.document_id}")
        feature_count += len(batch)
    draft_count = sum(document.revision is not None for document in docs)
    if len(list((run / "features").glob("*.jsonl"))) != draft_count:
        raise ValueError("Expected draft feature files only")
    model = json.loads((run / "model.json").read_text())
    window_count = 0
    if model["backend"] == "semantic":
        snapshot = (
            Path.home()
            / ".cache/huggingface/hub/models--sentence-transformers--"
            f"all-MiniLM-L6-v2/snapshots/{model['model_revision']}"
        )
        for relative, sha in model["artifacts"].items():
            if file_hash(snapshot / relative) != sha:
                raise ValueError(f"Model checksum mismatch: {relative}")
        tokenizer = content_tokenizer(snapshot / "tokenizer.json")
        for d in docs:
            by_clause = {}
            for w in read_jsonl(
                run / "embedding_windows" / f"{d.document_id}.jsonl"
            ):
                by_clause.setdefault(w["clause_id"], []).append(w)
            if set(by_clause) != {c.clause_id for c in d.clauses}:
                raise ValueError("Missing clause windows")
            for clause in d.clauses:
                tokens = tokenizer.encode(
                    clause.normalized_text, add_special_tokens=False
                ).ids
                covered_end = 0
                previous_start = -1
                for w in by_clause[clause.clause_id]:
                    a, b = w["token_start"], w["token_end"]
                    if (
                        not previous_start < a <= covered_end
                        or not a < b <= len(tokens)
                        or w["encoded_tokens"] > model["effective_token_limit"]
                    ):
                        raise ValueError(
                            f"Invalid/gapped window {clause.clause_id}: {a}:{b}"
                        )
                    if (
                        digest(tokens[a:b]) != w["content_token_sha256"]
                        or w["encoded_tokens"]
                        != b - a + model["special_tokens"]
                    ):
                        raise ValueError(f"Window token mismatch: {clause.clause_id}")
                    previous_start = a
                    covered_end = b
                    window_count += 1
                if covered_end != len(tokens):
                    raise ValueError(f"Truncated clause: {clause.clause_id}")
    protected_count = 0
    if protected:
        baseline = json.loads(protected.read_text())
        for path, sha in baseline.items():
            if file_hash(root / path) != sha:
                raise ValueError(f"Protected input changed: {path}")
        protected_count = len(baseline)
    report = json.loads((run / "report.json").read_text())
    if (
        report["relationship_counts"] != dict(relation_counts)
        or report["pairs"] != len(pair_ids)
        or report["documents"] != len(docs)
    ):
        raise ValueError("Report does not agree with serialized relationships")
    return {
        "status": "PASS",
        "documents": len(ids),
        "pairs": len(pair_ids),
        "relationships": len(rows),
        "candidates": candidate_count,
        "lineage_nodes": len(lineage),
        "feature_rows": feature_count,
        "windows_checked": window_count,
        "review_sample": len(sample),
        "protected_files_unchanged": protected_count,
        "temporal_guards": "PASS",
        "artifact_and_model_hashes": "PASS",
        "complete_token_coverage": "PASS",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--protected", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_run(args.run, args.root, args.protected), indent=2))
