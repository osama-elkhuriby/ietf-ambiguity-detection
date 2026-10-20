"""Train logistic regression with engineered, TF-IDF, and revision features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.classical.annotation_logreg_experiment2 import run_cli

FEATURE_MODE = "engineered_tfidf_revision"


def main() -> None:
    run_cli(FEATURE_MODE)


if __name__ == "__main__":
    main()
