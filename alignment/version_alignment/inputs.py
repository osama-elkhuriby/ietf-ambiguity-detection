"""Document-family discovery, chronology validation and input preparation."""

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


def timeline(
    metadata: dict[str, Any],
    rfcs: dict[str, Any],
    family_id: str = TLS_FAMILY,
    rfc_id: str | None = None,
) -> list[dict[str, Any]]:
    record = metadata[family_id]
    versions = sorted(record["versions"], key=lambda v: int(v["rev"]))
    numbers = [int(v["rev"]) for v in versions]
    if not numbers or numbers != list(range(max(numbers) + 1)):
        raise ValueError(f"{family_id} revisions missing/duplicated: {numbers}")
    result: list[dict[str, Any]] = []
    for v in versions:
        day = date.fromisoformat(v["submission_date"][:10])
        if result and day < result[-1]["available_at"]:
            raise ValueError(f"Chronology conflicts with revision {v['rev']}")
        result.append(
            {
                "document_id": f"{family_id}-{v['rev']}",
                "revision": v["rev"],
                "publication_date": day,
                "available_at": day,
                "date_precision": "day",
                "source_path": v["local_txt_path"],
            }
        )
    if rfc_id is None:
        matches = [
            key
            for key, value in rfcs.items()
            if value.get("draft") == result[-1]["document_id"]
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple RFC endpoints associated with {family_id}: {matches}"
            )
        rfc_id = matches[0] if matches else None
    rfc = rfcs.get(rfc_id) if rfc_id else None
    if rfc:
        if rfc.get("draft") != result[-1]["document_id"]:
            raise ValueError(
                f"{rfc_id} final-draft association conflicts with {family_id} timeline"
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
                "document_id": rfc_id,
                "revision": None,
                "publication_date": day,
                "available_at": upper,
                "date_precision": "month",
                "source_path": rfc["local_txt_path"],
            }
        )
    return result


def discover(
    root: Path,
    family_id: str = TLS_FAMILY,
    rfc_id: str | None = "RFC8446",
    clause_source: str = "clauses_llm",
) -> dict[str, Path]:
    """Discover one explicit classifier-output source without silent fallback."""
    if clause_source not in {"clauses", "clauses_llm"}:
        raise ValueError(f"Unknown clause source: {clause_source}")
    found: dict[str, Path] = {}
    folder = root / "corpus/processed" / clause_source
    for path in sorted(folder.rglob("*.json")):
        if not (
            path.stem.startswith(family_id + "-")
            or (rfc_id is not None and path.stem.upper() == rfc_id.upper())
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
    family_id: str = TLS_FAMILY,
    rfc_id: str | None = None,
    clause_source: str = "clauses_llm",
    prepare_all: bool = False,
) -> tuple[list[Document], dict[str, Any]]:
    metadata_paths = [
        root / "corpus/metadata" / name for name in ("drafts.json", "rfcs.json")
    ]
    metadata, rfcs = [json.loads(p.read_text(encoding="utf-8")) for p in metadata_paths]
    entries = timeline(metadata, rfcs, family_id, rfc_id)
    known = {e["document_id"] for e in entries}
    if selected and not selected <= known:
        raise ValueError(
            f"Unknown {family_id} selections: {sorted(selected - known)}"
        )
    entries = [e for e in entries if selected is None or e["document_id"] in selected]
    endpoint = next(
        (entry["document_id"] for entry in entries if entry["revision"] is None),
        rfc_id,
    )
    found = (
        {}
        if prepare_all
        else discover(root, family_id, endpoint, clause_source)
    )
    missing = [e["document_id"] for e in entries if e["document_id"] not in found]
    if missing and not (prepare_missing or prepare_all):
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
            family_id=family_id,
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
        "clause_source": clause_source,
        "prepared_all_from_raw": prepare_all,
        "input_policy": (
            "all inputs deterministically prepared from current raw sources"
            if prepare_all
            else f"{clause_source} required; missing inputs deterministically "
            "prepared only on request"
        ),
    }
