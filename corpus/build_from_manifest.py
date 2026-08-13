"""
Batch corpus builder — run the Stage 1 pipeline for every draft family listed in
corpus/metadata/final_rfc_manifest.csv.

Usage:
    python corpus/build_from_manifest.py
    python corpus/build_from_manifest.py --dry-run
    python corpus/build_from_manifest.py --area dns transport
    python corpus/build_from_manifest.py --skip-errata
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from paths import REPO_ROOT
from storage import FINAL_RFC_MANIFEST_PATH, PROTOCOL_AREA


def draft_names_from_manifest(
    manifest_path: Path = FINAL_RFC_MANIFEST_PATH,
    areas: set[str] | None = None,
) -> list[str]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names: list[str] = []
        for row in reader:
            draft = row["draft_family"]
            if areas and PROTOCOL_AREA.get(draft) not in areas:
                continue
            names.append(draft)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build corpus data for every draft family in final_rfc_manifest.csv."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FINAL_RFC_MANIFEST_PATH,
        help="CSV manifest listing draft families to fetch.",
    )
    parser.add_argument(
        "--area",
        nargs="*",
        choices=sorted(set(PROTOCOL_AREA.values())),
        help="Limit catch-up to selected protocol areas.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the drafts that would be fetched without calling build_corpus.py.",
    )
    parser.add_argument(
        "--skip-errata",
        action="store_true",
        help="Fetch drafts and RFCs only; skip the bulk errata download.",
    )
    args = parser.parse_args()

    areas = set(args.area) if args.area else None
    drafts = draft_names_from_manifest(args.manifest, areas=areas)
    if not drafts:
        print("No draft families matched the requested filters.")
        return

    print(f"Manifest targets: {len(drafts)} draft families")
    for name in drafts:
        print(f"  {name} ({PROTOCOL_AREA.get(name, 'unknown')})")

    if args.dry_run:
        return

    build_script = REPO_ROOT / "corpus" / "build_corpus.py"
    command = [sys.executable, str(build_script), *drafts]
    if args.skip_errata:
        print("\nNote: --skip-errata is not implemented in build_corpus.py yet; running full pipeline.")

    print("\nRunning build_corpus.py ...")
    subprocess.run(command, check=True, cwd=str(REPO_ROOT / "corpus"))

    print("\n=== Post-build metadata sync ===")
    import sync_metadata

    sync_metadata.sync_all(refresh_manifest=True, write_audit=True)


if __name__ == "__main__":
    main()
