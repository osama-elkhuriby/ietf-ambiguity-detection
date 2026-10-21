"""Prepare the frozen development cohort for matched non-transformer baselines.

This utility only validates and reshapes existing development data. It never fits
or loads a model. The output retains the transformer's five frozen outer folds so
the heuristic and classical runners can be invoked with ``--fold-column``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "experiments/transformer_final/final_annotation_human_reviewed_100_v1"
    / "manifests/dataset_v1/dataset.csv"
)
EXPECTED_SHA256 = "4e20a2571bfe71f38d04e5677e067ce4d2eda2008f00c297349b929bb662b4af"
EXPECTED_FAMILIES = {
    "draft-ietf-6man-rfc2460bis",
    "draft-ietf-httpbis-cache-header",
    "draft-ietf-httpbis-http2bis",
    "draft-ietf-httpbis-semantics",
    "draft-ietf-quic-http",
    "draft-ietf-quic-recovery",
    "draft-ietf-quic-tls",
    "draft-ietf-quic-transport",
    "draft-ietf-tls-tls13",
}
RESERVED_FAMILIES = {
    "draft-ietf-tcpm-rfc793bis",
    "draft-ietf-dnsop-svcb-https",
}
DEFAULT_OUTPUT = (
    ROOT / "results/planned/primary_development_baselines/primary_development_input.csv"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, output: Path) -> dict[str, object]:
    if output.exists() or output.with_name("manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite planned baseline artifacts in {output.parent}"
        )
    actual_hash = sha256_file(source)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Frozen development dataset hash mismatch: {actual_hash}")

    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {
        "item_id",
        "record_id",
        "family_id",
        "document_id",
        "section_heading",
        "previous_clause",
        "target_clause",
        "next_clause",
        "label",
        "final_source",
        "final_status",
        "fold",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Frozen dataset schema is missing: {sorted(missing)}")
    families = set(frame["family_id"])
    if families != EXPECTED_FAMILIES:
        raise RuntimeError(f"Unexpected development families: {sorted(families)}")
    if families & RESERVED_FAMILIES:
        raise RuntimeError("Reserved TCP/DNS family entered baseline preparation")
    labels = frame["label"].astype(int)
    folds = frame["fold"].astype(int)
    if len(frame) != 2143 or int(labels.sum()) != 138:
        raise RuntimeError("Expected the 2,143-row, 138-positive development cohort")
    if sorted(folds.unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Expected frozen outer folds 1 through 5")
    if (frame.assign(_fold=folds).groupby("family_id")["_fold"].nunique() != 1).any():
        raise RuntimeError("A draft family is split across frozen outer folds")

    prepared = pd.DataFrame(
        {
            "item_id": frame["item_id"],
            "record_id": frame["record_id"],
            "family_id": frame["family_id"],
            "document": frame["document_id"],
            "section_heading": frame["section_heading"],
            "previous_clause": frame["previous_clause"],
            "target_clause": frame["target_clause"],
            "next_clause": frame["next_clause"],
            "final_label": labels.map({0: "No", 1: "Yes"}),
            "validation_fold": folds,
            "final_source": frame["final_source"],
            "final_status": frame["final_status"],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False, lineterminator="\n")
    manifest: dict[str, object] = {
        "operation": "schema-only conversion for matched baseline evaluation; no model run",
        "source_path": str(source.resolve()),
        "source_sha256": actual_hash,
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "rows": len(prepared),
        "positives": int(labels.sum()),
        "families": sorted(families),
        "folds": sorted(folds.unique().tolist()),
        "reserved_family_guard": sorted(RESERVED_FAMILIES),
    }
    output.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
