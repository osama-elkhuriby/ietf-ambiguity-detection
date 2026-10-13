from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "corpus/processed/features/v3"
OUTPUT_ROOT = ROOT / "results/vote_weights"


def load_rows() -> pd.DataFrame:
    parts = []
    for split in ("train", "validation", "test"):
        rows = pd.read_csv(
            FEATURE_ROOT / "rows" / f"{split}_rows.csv",
            dtype={"record_id": str, "family_id": str},
        )
        labels = pd.read_csv(
            FEATURE_ROOT / "targets" / f"{split}_labels.csv",
            dtype={"record_id": str},
        )
        weights = pd.read_csv(
            FEATURE_ROOT / "weights" / f"{split}_sample_weights.csv",
            dtype={"record_id": str},
        )
        part = (
            rows[["record_id", "family_id"]]
            .merge(labels, on="record_id", validate="one_to_one")
            .merge(weights, on="record_id", validate="one_to_one")
        )
        part["split"] = split
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != 2546 or result["record_id"].duplicated().any():
        raise ValueError("Unexpected v3 row count or duplicate record IDs")
    if set(result["training_weight"].unique()) != {0.5, 1.0}:
        raise ValueError("Unexpected sample-weight values")
    return result


def write_report(frame: pd.DataFrame) -> None:
    distribution = (
        frame.groupby(["family_id", "label", "training_weight"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0.5: "n_weight_0_5", 1.0: "n_weight_1_0"})
        .reset_index()
    )
    for column in ("n_weight_0_5", "n_weight_1_0"):
        if column not in distribution:
            distribution[column] = 0
    distribution["n_rows"] = (
        distribution["n_weight_0_5"] + distribution["n_weight_1_0"]
    )
    distribution["share_weight_0_5"] = (
        distribution["n_weight_0_5"] / distribution["n_rows"]
    )
    distribution["weight_sum"] = (
        distribution["n_weight_0_5"] * 0.5
        + distribution["n_weight_1_0"]
    )
    distribution = distribution[
        [
            "family_id",
            "label",
            "n_rows",
            "n_weight_0_5",
            "n_weight_1_0",
            "share_weight_0_5",
            "weight_sum",
        ]
    ]
    distribution.to_csv(OUTPUT_ROOT / "weight_distribution.csv", index=False)

    overall_rows = []
    for label, group in frame.groupby("label", sort=True):
        half_count = int(group["training_weight"].eq(0.5).sum())
        overall_rows.append(
            {
                "label": int(label),
                "n_rows": len(group),
                "n_weight_0_5": half_count,
                "n_weight_1_0": int(group["training_weight"].eq(1.0).sum()),
                "share_weight_0_5": half_count / len(group),
                "mean_weight": float(group["training_weight"].mean()),
            }
        )
    overall = pd.DataFrame(overall_rows)
    weighted_positive_fraction = float(
        frame.loc[frame["label"].eq(1), "training_weight"].sum()
        / frame["training_weight"].sum()
    )
    small_families = {
        "draft-ietf-6man-rfc2460bis",
        "draft-ietf-httpbis-cache-header",
    }
    evaluated = frame.loc[~frame["family_id"].isin(small_families)]
    eval_weighted_positive_fraction = float(
        evaluated.loc[evaluated["label"].eq(1), "training_weight"].sum()
        / evaluated["training_weight"].sum()
    )
    majority_half = overall.loc[overall["share_weight_0_5"].gt(0.5), "label"].tolist()
    positive_weight_row = overall.loc[overall["label"].eq(1)].iloc[0]
    negative_weight_row = overall.loc[overall["label"].eq(0)].iloc[0]

    lines = [
        "# Vote-weight distribution audit",
        "",
        "## Summary",
        "",
        f"- Full dataset: {len(frame)} rows, {int(frame['label'].sum())} positives.",
        f"- Weighted positive fraction, computed as positive-row weight sum divided by total weight sum: {weighted_positive_fraction:.4%}.",
        f"- On the 9-family evaluated cohort (2,540 rows, 453 positives; the two <30-row families are excluded only from evaluation), weighted positive fraction: {eval_weighted_positive_fraction:.4%}.",
        f"- Class(es) for which a majority of rows have weight 0.5: {', '.join(str(value) for value in majority_half)}.",
        "",
        "## Distribution by label",
        "",
        "| Label | Rows | Weight 0.5 | Weight 1.0 | Share at 0.5 | Mean weight |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.n_rows} | {row.n_weight_0_5} | "
            f"{row.n_weight_1_0} | {row.share_weight_0_5:.1%} | {row.mean_weight:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Distribution by family and label",
            "",
            "The complete count table is in `weight_distribution.csv`.",
            "",
            "| Family | Label | Rows | Weight 0.5 | Weight 1.0 | Share at 0.5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in distribution.itertuples(index=False):
        lines.append(
            f"| {row.family_id} | {row.label} | {row.n_rows} | "
            f"{row.n_weight_0_5} | {row.n_weight_1_0} | {row.share_weight_0_5:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Decision gate",
            "",
        ]
    )
    if majority_half:
        lines.extend(
            [
                "STOP: at least one class is mostly weight 0.5. No model was run, as required.",
                    f"The label-1 class is {positive_weight_row.share_weight_0_5:.1%} "
                    f"weight 0.5 ({int(positive_weight_row.n_weight_0_5)}/"
                    f"{int(positive_weight_row.n_rows)} rows); the label-0 class is "
                    f"{negative_weight_row.share_weight_0_5:.1%} weight 0.5 "
                    f"({int(negative_weight_row.n_weight_0_5)}/"
                    f"{int(negative_weight_row.n_rows)} rows).",
                ]
            )
    else:
        lines.append("No class is majority weight 0.5; the gate did not trigger.")
    lines.extend(
        [
            "",
            "Weights derive from the same LLM votes as the labels; gains reflect fit to consensus labels, not verified ambiguity detection. Labels come from a candidate cohort selected using future information.",
            "",
            "No stored weight was used as a feature, matrix column, preprocessing input, metric weight, or selection criterion. No repository artifacts outside `results/vote_weights/` were changed by this audit.",
            "",
        ]
    )
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = load_rows()
    write_report(frame)
    print((OUTPUT_ROOT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
