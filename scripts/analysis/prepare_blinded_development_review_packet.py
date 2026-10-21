"""Create an independently ordered, label-blinded development review packet.

The packet is for annotation-definition/reliability study only. Its balanced,
stratified sampling does not estimate corpus prevalence or detector performance.
The script pins the permitted nine-family development cohort and refuses any TCP
or DNS family. It performs no model fitting or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "experiments/transformer_final/final_annotation_human_reviewed_100_v1"
    / "manifests/dataset_v1/dataset.csv"
)
EXPECTED_SHA256 = "4e20a2571bfe71f38d04e5677e067ce4d2eda2008f00c297349b929bb662b4af"
RESERVED_FAMILIES = {
    "draft-ietf-tcpm-rfc793bis",
    "draft-ietf-dnsop-svcb-https",
}
DEFAULT_OUTPUT = ROOT / "results/planned/blinded_development_review_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, output_dir: Path, *, per_stratum: int, seed: int) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite packet directory: {output_dir}")
    actual_hash = sha256_file(source)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Frozen development dataset hash mismatch: {actual_hash}")
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {
        "item_id", "record_id", "family_id", "section_heading", "previous_clause",
        "target_clause", "next_clause", "label", "fold",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Frozen dataset schema is missing: {sorted(missing)}")
    families = set(frame["family_id"])
    if families & RESERVED_FAMILIES:
        raise RuntimeError("Reserved TCP/DNS family entered review-packet preparation")
    if len(families) != 9 or len(frame) != 2143 or frame["label"].astype(int).sum() != 138:
        raise RuntimeError("Source is not the frozen 2,143-row development cohort")
    if per_stratum < 1:
        raise ValueError("per-stratum must be positive")

    sampled = (
        frame.groupby(["family_id", "label"], group_keys=False, sort=True)
        .apply(lambda group: group.sample(min(len(group), per_stratum), random_state=seed))
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    stable_order = rng.permutation(len(sampled))
    sampled = sampled.iloc[stable_order].reset_index(drop=True)
    sampled["case_id"] = [f"DEV-{index:04d}" for index in range(1, len(sampled) + 1)]

    visible = [
        "case_id", "section_heading", "previous_clause", "target_clause", "next_clause"
    ]
    response_columns = [
        "judgment_yes_no_uncertain",
        "materially_distinct_behavior_a",
        "materially_distinct_behavior_b",
        "context_resolves_wording_yes_no",
        "difficulty_only_yes_no",
        "confidence_1_to_5",
        "rationale",
    ]
    output_dir.mkdir(parents=True)
    for reviewer, reviewer_seed in (("reviewer_a", seed + 1), ("reviewer_b", seed + 2)):
        packet = sampled[visible].sample(frac=1, random_state=reviewer_seed).copy()
        for column in response_columns:
            packet[column] = ""
        packet.to_csv(output_dir / f"{reviewer}.csv", index=False, lineterminator="\n")

    key = sampled[
        ["case_id", "item_id", "record_id", "family_id", "fold", "label"]
    ].rename(columns={"label": "reference_label_not_for_reviewers"})
    key.to_csv(output_dir / "key_PRIVATE.csv", index=False, lineterminator="\n")
    instructions = """# Blinded development review instructions

Reviewers must work independently and must not see `key_PRIVATE.csv`, model scores,
historical decisions, candidate source, or one another's judgments. Mark **Yes**
only when the wording admits at least two reasonable, materially different
implementation behaviours, or when a concrete missing constraint prevents a
unique operational interpretation. Difficult prose, subjective terminology,
normative discretion, and wording resolved by the supplied context are not by
themselves ambiguity. Use **Uncertain** when evidence outside the supplied context
is required. After both files are locked, an adjudicator records disagreements in
a separate table without rewriting the original reviewer files.

This deliberately balanced development sample is suitable for agreement and
definition diagnostics only. It is not a prevalence or generalisation estimate.
"""
    (output_dir / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    manifest: dict[str, object] = {
        "purpose": "blinded annotation agreement/definition diagnostic on development data",
        "source_path": str(source.resolve()),
        "source_sha256": actual_hash,
        "selection": "family-by-reference-label stratified random sample",
        "per_stratum_cap": per_stratum,
        "seed": seed,
        "rows": len(sampled),
        "families": sorted(families),
        "reserved_family_guard": sorted(RESERVED_FAMILIES),
        "performance_evaluation_permitted": False,
        "reviewer_independence_required": True,
        "private_key_sha256": sha256_file(output_dir / "key_PRIVATE.csv"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    result = prepare(args.source, args.output_dir, per_stratum=args.per_stratum, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
