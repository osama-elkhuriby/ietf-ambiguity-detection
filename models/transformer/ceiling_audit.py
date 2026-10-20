"""Audit the silver-label ceiling and export OOF errors without test access."""

from __future__ import annotations

import argparse
import json
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT
from models.transformer.experiment import load_development_split, load_frozen_folds
from models.transformer.feasibility_audit import _routing_models
from models.transformer.lr_screen_analysis import DEVELOPMENT_DIR, FOLD_ASSIGNMENT

CONSENSUS_PATH = ROOT / "corpus/processed/llm_annotations/v1/consensus.csv"
MODEL_NAMES = ("gemma3:27b", "gpt-oss:20b", "qwen3:32b")


def _safe_name(model: str) -> str:
    return model.replace(":", "_").replace("-", "_")


def parse_vote_payload(payload: str) -> dict[str, float]:
    """Return rationales-free binary votes, with invalid responses represented as NaN."""

    parsed = json.loads(payload)
    votes: dict[str, float] = {}
    for model in MODEL_NAMES:
        item = parsed.get(model, {})
        vote = item.get("vote") if item.get("status") == "valid" else None
        votes[_safe_name(model)] = (
            1.0 if vote == "yes" else 0.0 if vote == "no" else np.nan
        )
    return votes


def _development_with_votes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = load_development_split(DEVELOPMENT_DIR, "train")
    tcp = load_development_split(DEVELOPMENT_DIR, "validation")
    allowlist = set(train["record_id"]) | set(tcp["record_id"])
    source = pd.read_csv(CONSENSUS_PATH, usecols=["record_id", "votes_json"])
    source = source[source["record_id"].isin(allowlist)].copy()
    if set(source["record_id"]) != allowlist:
        missing = sorted(allowlist - set(source["record_id"]))
        raise RuntimeError(f"Consensus votes missing for development rows: {missing[:3]}")
    vote_columns = source["votes_json"].map(parse_vote_payload).apply(pd.Series)
    rationales_free = pd.concat(
        [source[["record_id"]].reset_index(drop=True), vote_columns.reset_index(drop=True)],
        axis=1,
    )
    folds = load_frozen_folds(
        train,
        FOLD_ASSIGNMENT,
        expected_sha256=sha256_file(FOLD_ASSIGNMENT),
        expected_train_sha256=sha256_file(DEVELOPMENT_DIR / "train.csv"),
    )
    fold_by_record: dict[str, int] = {}
    scored_ids: set[str] = set()
    for fold in folds:
        fold_by_record.update(
            {train.iloc[index]["record_id"]: fold.number for index in fold.validation_indices}
        )
        scored_ids.update(train.iloc[fold.scoring_indices]["record_id"])
    train = train.merge(rationales_free, on="record_id", validate="one_to_one")
    train["fold"] = train["record_id"].map(fold_by_record)
    train["included_in_oof_scoring"] = train["record_id"].isin(scored_ids)
    tcp = tcp.merge(rationales_free, on="record_id", validate="one_to_one")
    tcp["fold"] = "tcp"
    tcp["included_in_oof_scoring"] = False
    return train, tcp, rationales_free


def _scope_frames(train: pd.DataFrame, tcp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    development = pd.concat([train, tcp], ignore_index=True)
    scopes = {
        "development_all": development,
        "train_all": train,
        "oof_scored_pooled": train[train["included_in_oof_scoring"]],
        "tcp_validation": tcp,
    }
    for fold in range(1, 6):
        scopes[f"fold_{fold}"] = train[
            (train["fold"] == fold) & train["included_in_oof_scoring"]
        ]
    return scopes


def leave_one_llm_out_metrics(scopes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Score each model vote against consensus of the other two valid agreeing votes."""

    columns = tuple(_safe_name(model) for model in MODEL_NAMES)
    records: list[dict[str, Any]] = []
    for scope_name, scope in scopes.items():
        for tier in ("ALL", "HIGH", "MEDIUM"):
            tier_frame = scope if tier == "ALL" else scope[scope["label_tier"] == tier]
            for held_out in columns:
                others = [column for column in columns if column != held_out]
                eligible = tier_frame[
                    tier_frame[held_out].notna()
                    & tier_frame[others[0]].notna()
                    & tier_frame[others[1]].notna()
                    & (tier_frame[others[0]] == tier_frame[others[1]])
                ]
                target = eligible[others[0]].to_numpy(dtype=np.int64)
                prediction = eligible[held_out].to_numpy(dtype=np.float64)
                pr_auc = (
                    float(average_precision_score(target, prediction))
                    if len(np.unique(target)) == 2
                    else None
                )
                records.append(
                    {
                        "scope": scope_name,
                        "label_tier": tier,
                        "held_out_llm": held_out,
                        "eligible_rows": len(eligible),
                        "excluded_without_two_model_consensus": len(tier_frame)
                        - len(eligible),
                        "other_two_consensus_positives": int(target.sum()),
                        "pr_auc": pr_auc,
                    }
                )
    return pd.DataFrame(records)


def pairwise_agreement_metrics(scopes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute pairwise agreement and Cohen's kappa on mutually valid votes."""

    columns = tuple(_safe_name(model) for model in MODEL_NAMES)
    records: list[dict[str, Any]] = []
    for scope_name, scope in scopes.items():
        for first, second in combinations(columns, 2):
            eligible = scope[scope[first].notna() & scope[second].notna()]
            first_votes = eligible[first].to_numpy(dtype=np.int64)
            second_votes = eligible[second].to_numpy(dtype=np.int64)
            records.append(
                {
                    "scope": scope_name,
                    "llm_a": first,
                    "llm_b": second,
                    "eligible_rows": len(eligible),
                    "raw_agreement": float(np.mean(first_votes == second_votes)),
                    "cohen_kappa": float(cohen_kappa_score(first_votes, second_votes)),
                    "llm_a_positive_rate": float(first_votes.mean()),
                    "llm_b_positive_rate": float(second_votes.mean()),
                }
            )
    return pd.DataFrame(records)


def add_vote_fraction(frame: pd.DataFrame) -> pd.DataFrame:
    """Add yes-vote fraction on the fixed three-model denominator."""

    output = frame.copy()
    columns = [_safe_name(model) for model in MODEL_NAMES]
    output["valid_vote_count"] = output[columns].notna().sum(axis=1)
    output["yes_vote_fraction"] = output[columns].fillna(0.0).sum(axis=1) / 3.0
    return output


def rank_correlations(
    train: pd.DataFrame, models: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Measure whether OOF model scores track the three-model yes-vote fraction."""

    votes = add_vote_fraction(train[train["included_in_oof_scoring"]])[
        ["record_id", "fold", "label_tier", "valid_vote_count", "yes_vote_fraction"]
    ]
    selected = {
        "deberta_logit_average": models["deberta_1e-5_logit_average"],
        "tfidf_unweighted": models["tfidf_unweighted"],
    }
    records: list[dict[str, Any]] = []
    for name, predictions in selected.items():
        merged = votes.merge(
            predictions[["record_id", "ambiguity_probability"]],
            on="record_id",
            validate="one_to_one",
        )
        for scope_name, scope in [
            ("oof_scored_pooled", merged),
            *[(f"fold_{fold}", merged[merged["fold"] == fold]) for fold in range(1, 6)],
        ]:
            for validity, subset in (
                ("all_consensus_rows", scope),
                ("three_valid_votes", scope[scope["valid_vote_count"] == 3]),
            ):
                result = spearmanr(
                    subset["ambiguity_probability"], subset["yes_vote_fraction"]
                )
                records.append(
                    {
                        "model": name,
                        "scope": scope_name,
                        "vote_validity": validity,
                        "rows": len(subset),
                        "spearman_rho": float(result.statistic),
                        "p_value": float(result.pvalue),
                    }
                )
    return pd.DataFrame(records)


def error_analysis_export(
    train: pd.DataFrame, models: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Export extreme OOF errors without votes or rationales."""

    metadata = train[
        ["record_id", "family_id", "label_tier", "target_clause"]
    ].copy()
    deberta = models["deberta_1e-5_logit_average"][
        ["record_id", "label", "ambiguity_probability"]
    ].rename(columns={"ambiguity_probability": "deberta_logit_average_score"})
    tfidf = models["tfidf_unweighted"][["record_id", "ambiguity_probability"]].rename(
        columns={"ambiguity_probability": "tfidf_unweighted_score"}
    )
    merged = deberta.merge(tfidf, on="record_id", validate="one_to_one").merge(
        metadata, on="record_id", validate="one_to_one"
    )
    records: list[pd.DataFrame] = []
    for model_name, score_column in (
        ("deberta_logit_average", "deberta_logit_average_score"),
        ("tfidf_unweighted", "tfidf_unweighted_score"),
    ):
        high_negatives = (
            merged[merged["label"] == 0]
            .sort_values([score_column, "record_id"], ascending=[False, True])
            .head(40)
            .copy()
        )
        high_negatives["error_slice"] = "highest_scoring_oof_negative"
        low_positives = (
            merged[merged["label"] == 1]
            .sort_values([score_column, "record_id"], ascending=[True, True])
            .head(40)
            .copy()
        )
        low_positives["error_slice"] = "lowest_scoring_oof_positive"
        selected = pd.concat([high_negatives, low_positives], ignore_index=True)
        selected["model"] = model_name
        selected["model_score"] = selected[score_column]
        selected["rank_within_slice"] = selected.groupby("error_slice").cumcount() + 1
        records.append(selected)
    output = pd.concat(records, ignore_index=True)
    return output[
        [
            "model",
            "error_slice",
            "rank_within_slice",
            "record_id",
            "family_id",
            "label_tier",
            "label",
            "model_score",
            "deberta_logit_average_score",
            "tfidf_unweighted_score",
            "target_clause",
        ]
    ]


def run(output_dir: Path) -> Path:
    """Write immutable ceiling and error-analysis artifacts."""

    if output_dir.exists():
        raise FileExistsError(f"Ceiling audit output is immutable: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}-{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        train, tcp, _ = _development_with_votes()
        scopes = _scope_frames(train, tcp)
        models, _ = _routing_models()
        loo = leave_one_llm_out_metrics(scopes)
        agreement = pairwise_agreement_metrics(scopes)
        correlations = rank_correlations(train, models)
        errors = error_analysis_export(train, models)
        artifacts = {
            "leave_one_llm_out.csv": loo,
            "pairwise_agreement.csv": agreement,
            "score_vote_fraction_spearman.csv": correlations,
            "oof_extreme_errors.csv": errors,
        }
        for name, frame in artifacts.items():
            frame.to_csv(temporary / name, index=False, lineterminator="\n")
        summary = {
            "schema_version": 1,
            "vote_source": str(CONSENSUS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "vote_source_sha256": sha256_file(CONSENSUS_PATH),
            "rationales_read_or_exported": False,
            "development_rows": len(train) + len(tcp),
            "training_rows": len(train),
            "tcp_rows": len(tcp),
            "oof_scored_rows": int(train["included_in_oof_scoring"].sum()),
            "model_names": list(MODEL_NAMES),
            "vote_fraction_definition": "yes votes divided by three fixed models; invalid votes add zero yes votes",
            "error_export_rows": len(errors),
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hashes = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(
                {"schema_version": 1, "artifacts": hashes},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/transformer/governance/step2d_ceiling_v1",
    )
    args = parser.parse_args()
    print(run(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
