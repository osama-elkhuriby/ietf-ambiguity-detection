"""Make corpus/ scripts importable as top-level modules in tests."""
from __future__ import annotations

import sys
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
if str(CORPUS_DIR) not in sys.path:
    sys.path.insert(0, str(CORPUS_DIR))
