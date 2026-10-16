"""Train and validate logistic regression with TF-IDF and clause embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.classical.annotation_logreg_experiment import run_cli

FEATURE_MODE = "tfidf_embeddings"


def main() -> None:
    run_cli(FEATURE_MODE)


if __name__ == "__main__":
    main()
