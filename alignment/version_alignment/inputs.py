"""TLS-only discovery, chronology validation and non-destructive preparation."""

from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path
from typing import Any

from alignment.clause_segmentation import segment_document
from alignment.segmentation import parse_document

from . import TLS_FAMILY
from .io import file_hash, write_json
from .models import Clause, Document
from .normalize import normalize


def timeline(metadata: dict[str, Any], rfcs: dict[str, Any]) -> list[dict[str, Any]]:
    record = metadata[TLS_FAMILY]
    versions = sorted(record["versions"], key=lambda v: int(v["rev"]))
    numbers = [int(v["rev"]) for v in versions]
    if not numbers or numbers != list(range(max(numbers) + 1)):
        raise ValueError(f"TLS revisions missing/duplicated: {numbers}")
    result: list[dict[str, Any]] = []
    for v in versions:
        day = date.fromisoformat(v["submission_date"][:10])
        if result and day < result[-1]["available_at"]:
            raise ValueError(f"Chronology conflicts with revision {v['rev']}")
        result.append(
            {
                "document_id": f"{TLS_FAMILY}-{v['rev']}",
                "revision": v["rev"],
                "publication_date": day,
                "available_at": day,
                "date_precision": "day",
                "source_path": v["local_txt_path"],
            }
        )
    rfc = rfcs.get("RFC8446")
    if rfc:
        if rfc.get("draft") != result[-1]["document_id"]:
            raise ValueError(
                "RFC8446 final-draft association conflicts with TLS timeline"
            )
        day = date.fromisoformat(rfc["date"][:10])
        # RFC index exposes month precision; first-of-month is not exact availability.
        upper = day.replace(day=calendar.monthrange(day.year, day.month)[1])
        if day < result[-1]["available_at"]:
            raise ValueError(
                "RFC month overlaps/precedes final draft: exact publication evidence required"
            )
        result.append(
            {
                "document_id": "RFC8446",
                "revision": None,
                "publication_date": day,
                "available_at": upper,
                "date_precision": "month",
                "source_path": rfc["local_txt_path"],
            }
        )
    return result


def discover(root: Path) -> dict[str, Path]:
    """Use only updated TLS outputs; never silently substitute older classifications."""
    found: dict[str, Path] = {}
    folder = root / "corpus/processed/clauses_llm"
    for path in sorted(folder.rglob("*.json")):
        if not (
            path.stem.startswith(TLS_FAMILY + "-") or path.stem.upper() == "RFC8446"
        ):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload.get("document_id")
        if identity != path.stem:
            raise ValueError(f"Filename/document mismatch: {path}")
        if identity in found:
            raise ValueError(
                f"Duplicate preferred document {identity}: {found[identity]}, {path}"
            )
        found[identity] = path
    return found


def prepare_inputs(
    root: Path,
    run: Path,
    *,
    prepare_missing: bool = False,
    selected: set[str] | None = None,
) -> tuple[list[Document], dict[str, Any]]:
    metadata_paths = [
        root / "corpus/metadata" / name for name in ("drafts.json", "rfcs.json")
    ]
    metadata, rfcs = [json.loads(p.read_text(encoding="utf-8")) for p in metadata_paths]
    entries = timeline(metadata, rfcs)
    known = {e["document_id"] for e in entries}
    if selected and not selected <= known:
        raise ValueError(f"Unknown TLS selections: {sorted(selected - known)}")
    entries = [e for e in entries if selected is None or e["document_id"] in selected]
    found = discover(root)
    missing = [e["document_id"] for e in entries if e["document_id"] not in found]
    if missing and not prepare_missing:
        raise ValueError(
            f"Missing preferred clause snapshots: {missing}; use --prepare-missing"
        )
    documents = []
    for entry in entries:
        identity = entry["document_id"]
        raw = root / entry["source_path"]
        if not raw.is_file():
            raise ValueError(f"Missing raw source: {raw}")
        prepared = identity not in found
        if prepared:
            payload = segment_document(parse_document(raw)).to_dict()
            input_hash = None
            origin = str(raw)
        else:
            origin = str(found[identity])
            input_hash = file_hash(found[identity])
            payload = json.loads(found[identity].read_text(encoding="utf-8"))
        if payload["document_id"] != identity or payload["source_sha256"] != file_hash(
            raw
        ):
            raise ValueError(f"Source hash/document mismatch: {identity}")
        snapshot = run / "inputs" / f"{identity}.json"
        clauses = []
        for item in payload["clauses"]:
            clause = Clause.model_validate(item)
            clause.normalized_text, clause.normalization_steps = normalize(
                clause.raw_text, clause.block_type
            )
            clauses.append(clause)
        document = Document(
            family_id=TLS_FAMILY,
            **entry,
            source_sha256=file_hash(raw),
            input_sha256=input_hash or file_hash(raw),
            input_path=origin,
            prepared=prepared,
            clauses=clauses,
            warnings=payload.get("warnings", []),
        )
        write_json(snapshot, document)
        documents.append(document)
    return documents, {
        "metadata_hashes": {p.name: file_hash(p) for p in metadata_paths},
        "prepared_documents": missing,
        "rfc_link_source": "RFC index draft field; historical cross-check independence unknown",
        "input_policy": "clauses_llm preferred; missing inputs deterministically prepared only on request",
    }
