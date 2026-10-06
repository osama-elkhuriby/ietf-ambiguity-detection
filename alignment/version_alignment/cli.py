"""Immutable document-family alignment runner; never invokes collection or LLM refinement."""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import logging
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from eval.alignment_metrics import evaluate

from . import ALGORITHM_VERSION, TLS_FAMILY
from .candidates import candidate_graph
from .config import load_config
from .embeddings import Embedder
from .features import features
from .inputs import prepare_inputs
from .io import digest, file_hash, read_jsonl, write_json, write_jsonl
from .lineage import build_lineage
from .matching import select
from .review import records, stratified_sample
from .temporal import prediction_view

LOG = logging.getLogger(__name__)


def run(
    root: Path,
    output: Path,
    config,
    *,
    prepare_missing=False,
    allow_download=False,
    pair: tuple[str, str] | None = None,
    prepare_only=False,
    gold: Path | None = None,
    family_id: str = TLS_FAMILY,
    rfc_id: str | None = None,
    clause_source: str = "clauses_llm",
    prepare_all: bool = False,
):
    start = time.perf_counter()
    code_paths = sorted((root / "alignment/version_alignment").glob("*.py")) + [
        root / "eval/alignment_metrics.py"
    ]
    initial_code_hashes = {
        p.relative_to(root).as_posix(): file_hash(p) for p in code_paths if p.is_file()
    }
    created = datetime.now(UTC).isoformat()
    current_pair = None
    current_document = None
    stage = "input_preparation"
    pair_reports = []
    # Reject an existing output directory, including partially completed prior runs.
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "request.json",
        {
            "family_id": family_id,
            "rfc_id": rfc_id,
            "clause_source": clause_source,
            "created_at": created,
            "config": config.model_dump(),
            "prepare_missing": prepare_missing,
            "prepare_all": prepare_all,
            "pair": pair,
            "algorithm_version": ALGORITHM_VERSION,
        },
    )
    try:
        docs, input_info = prepare_inputs(
            root,
            output,
            prepare_missing=prepare_missing,
            selected=set(pair) if pair else None,
            family_id=family_id,
            rfc_id=rfc_id,
            clause_source=clause_source,
            prepare_all=prepare_all,
        )
        if pair and (
            len(docs) != 2
            or docs[0].document_id != pair[0]
            or docs[1].document_id != pair[1]
            or (
                docs[1].revision is not None
                and int(docs[1].revision) != int(docs[0].revision) + 1
            )
        ):
            raise ValueError(
                "Smoke pair must be consecutive revisions in source-to-target order"
            )
        LOG.info(
            "Validated %s documents; prepared %s",
            len(docs),
            input_info["prepared_documents"],
        )
        stage = "temporal_preflight"
        for document in docs:
            if document.revision is not None:
                prediction_view(docs, [], document.document_id)
        all_rows = []
        embedder = None
        if not prepare_only:
            stage = "model_initialization"
            embedder = Embedder(
                config,
                root / "corpus/processed/alignments/_embedding_cache",
                allow_download=allow_download,
            )
            write_json(output / "model.json", embedder.metadata)
            model_id = digest(embedder.metadata)

            def encode_document(document):
                vectors = embedder.encode([c.normalized_text for c in document.clauses])
                if vectors is not None:
                    write_jsonl(
                        output / "embedding_windows" / f"{document.document_id}.jsonl",
                        (
                            {
                                **w,
                                "clause_id": document.clauses[
                                    w["text_index"]
                                ].clause_id,
                                "document_id": document.document_id,
                            }
                            for w in embedder.last_windows
                        ),
                    )
                return vectors

            stage = "embedding"
            current_document = docs[0].document_id
            previous_vectors = encode_document(docs[0])
            for a, b in itertools.pairwise(docs):
                current_pair = [a.document_id, b.document_id]
                current_document = b.document_id
                pair_start = time.perf_counter()
                LOG.info(
                    "Aligning %s -> %s (%s -> %s clauses)",
                    a.document_id,
                    b.document_id,
                    len(a.clauses),
                    len(b.clauses),
                )
                stage = "embedding"
                vectors = encode_document(b)
                stage = "candidate_generation"
                candidates = candidate_graph(a, b, config, previous_vectors, vectors)
                stage = "global_selection"
                chosen, diagnostics = select(candidates, config)
                stage = "relationship_output"
                rows = records(a, b, candidates, chosen, config, created, model_id)
                candidate_rows = [
                    {
                        "schema_version": "1.0",
                        "from_document_id": a.document_id,
                        "to_document_id": b.document_id,
                        "source_clause_ids": [a.clauses[i].clause_id for i in e.source],
                        "target_clause_ids": [b.clauses[i].clause_id for i in e.target],
                        "scores": e.scores,
                        "exact": e.exact,
                        "locked": e.locked,
                    }
                    for e in candidates
                ]
                report = evaluate([r.model_dump(mode="json") for r in rows])
                report.update(
                    from_document_id=a.document_id,
                    to_document_id=b.document_id,
                    candidate_count=len(candidates),
                    runtime_seconds=time.perf_counter() - pair_start,
                    diagnostics=diagnostics,
                )
                pair_reports.append(report)
                all_rows.extend(rows)
                pair_folder = output / "pairs" / f"{a.document_id}__{b.document_id}"
                write_jsonl(pair_folder / "candidates.jsonl", candidate_rows)
                write_json(pair_folder / "report.json", report)
                LOG.info(
                    "Pair finished in %.1fs: %s",
                    report["runtime_seconds"],
                    report["relationship_counts"],
                )
                previous_vectors = vectors
            stage = "lineage_and_review_outputs"
            write_jsonl(
                output / "candidates.jsonl",
                (
                    row
                    for path in sorted((output / "pairs").glob("*/candidates.jsonl"))
                    for row in read_jsonl(path)
                ),
            )
            nodes = build_lineage(docs, all_rows)
            for a, b in itertools.pairwise(docs):
                pair_rows = [
                    r
                    for r in all_rows
                    if r.from_document_id == a.document_id
                    and r.to_document_id == b.document_id
                ]
                write_jsonl(
                    output
                    / "pairs"
                    / f"{a.document_id}__{b.document_id}"
                    / "alignments.jsonl",
                    pair_rows,
                )
            write_jsonl(output / "alignments.jsonl", all_rows)
            write_jsonl(output / "lineage.jsonl", nodes)
            clause_map = {c.clause_id: c for d in docs for c in d.clauses}

            def bundle(row):
                return {
                    "alignment": row.model_dump(mode="json"),
                    "source_clauses": [
                        clause_map[c].model_dump(mode="json")
                        for c in row.source_clause_ids
                    ],
                    "target_clauses": [
                        clause_map[c].model_dump(mode="json")
                        for c in row.target_clause_ids
                    ],
                    "reviewer": None,
                    "reviewed_relation": None,
                    "review_notes": None,
                    "evidence_role": "ANNOTATION_METADATA",
                    "available_at": None,
                }

            write_jsonl(
                output / "review_queue.jsonl",
                (bundle(r) for r in all_rows if r.review_required),
            )
            write_jsonl(
                output / "review_sample.jsonl",
                (
                    bundle(r)
                    for r in stratified_sample(
                        all_rows, config.review_sample_size, config.seed
                    )
                ),
            )
            feature_diagnostics = []
            stage = "guarded_features"
            for document in docs:
                if document.revision is None:
                    continue
                try:
                    batch = features(docs, all_rows, document.document_id)
                    write_jsonl(
                        output / "features" / f"{document.document_id}.jsonl", batch
                    )
                except ValueError as exc:
                    # Incomplete date evidence blocks features, not offline label-support alignment.
                    feature_diagnostics.append(
                        {
                            "document_id": document.document_id,
                            "error": str(exc),
                            "severity": "FEATURES_BLOCKED",
                        }
                    )
            write_jsonl(output / "diagnostics.jsonl", feature_diagnostics)
            if feature_diagnostics:
                raise ValueError(
                    f"Feature guards blocked {len(feature_diagnostics)} documents; see diagnostics.jsonl"
                )
        reviewed = (
            [
                json.loads(line)
                for line in gold.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if gold
            else None
        )
        report = evaluate(
            [r.model_dump(mode="json") for r in all_rows],
            read_jsonl(output / "candidates.jsonl") if not prepare_only else None,
            reviewed,
        )
        report.update(
            documents=len(docs),
            pairs=len(pair_reports),
            prepared_documents=input_info["prepared_documents"],
            runtime_seconds=time.perf_counter() - start,
            pair_reports=pair_reports,
            human_review_sample_size=min(config.review_sample_size, len(all_rows)),
            features_blocked=feature_diagnostics if not prepare_only else [],
            long_clauses_encoded=embedder.long_count if embedder else 0,
            windowing_statistics=embedder.stats if embedder else [],
            token_window_policy="original token-ID slices; no decode/retokenize; limit includes special tokens",
        )
        stage = "manifest_and_integrity"
        for document in docs:
            if file_hash(root / document.source_path) != document.source_sha256:
                raise ValueError(
                    f"Original source changed during run: {document.document_id}"
                )
            if (
                not document.prepared
                and file_hash(Path(document.input_path)) != document.input_sha256
            ):
                raise ValueError(
                    f"Original clauses changed during run: {document.document_id}"
                )
        report["original_selected_inputs_unchanged"] = True
        write_json(output / "report.json", report)
        manifest = {
            "schema_version": "1.0",
            "status": "PREPARED" if prepare_only else "COMPLETE",
            "family_id": family_id,
            "created_at": created,
            "algorithm_version": ALGORITHM_VERSION,
            "config": config.model_dump(),
            "input_info": input_info,
            "documents": [
                {
                    "document_id": d.document_id,
                    "source_sha256": d.source_sha256,
                    "input_sha256": d.input_sha256,
                    "snapshot_sha256": file_hash(
                        output / "inputs" / f"{d.document_id}.json"
                    ),
                }
                for d in docs
            ],
            "code_hashes_at_start": initial_code_hashes,
            "package_versions": {
                n: importlib.metadata.version(n)
                for n in ("numpy", "scipy", "pydantic", "rapidfuzz", "networkx")
            },
            "artifacts": {
                p.relative_to(output).as_posix(): file_hash(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
            "gold_sha256": file_hash(gold) if gold else None,
        }
        write_json(output / "run_manifest.json", manifest)
        LOG.info("Run complete: %s", output)
        return report
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_at": created,
                "failed_at": datetime.now(UTC).isoformat(),
                "stage": stage,
                "document_id": current_document,
                "pair": current_pair,
                "completed_pairs": len(pair_reports),
                "completed_pair_ids": [
                    [p["from_document_id"], p["to_document_id"]] for p in pair_reports
                ],
                "traceback": traceback.format_exc(),
                "config": config.model_dump(),
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Versioned clause-alignment runner")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing paths are rejected",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=["semantic", "lexical"])
    parser.add_argument("--model-revision")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--prepare-missing", action="store_true")
    parser.add_argument(
        "--prepare-all",
        action="store_true",
        help=(
            "Deterministically parse and classify every input from current raw "
            "sources inside the new run; do not overwrite corpus clause artifacts."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pair", nargs=2, metavar=("FROM_DOCUMENT", "TO_DOCUMENT"))
    parser.add_argument("--gold", type=Path)
    parser.add_argument(
        "--family",
        default=TLS_FAMILY,
        help="Draft family identifier from corpus/metadata/drafts.json.",
    )
    parser.add_argument(
        "--rfc",
        help="Optional final RFC document ID; inferred from metadata by default.",
    )
    parser.add_argument(
        "--clause-source",
        choices=["clauses", "clauses_llm"],
        default="clauses_llm",
        help="Explicit first-stage clause snapshot directory under corpus/processed.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config(
        args.config, backend=args.backend, model_revision=args.model_revision
    )
    try:
        run(
            args.root.resolve(),
            args.output.resolve(),
            config,
            prepare_missing=args.prepare_missing,
            allow_download=args.allow_download,
            pair=tuple(args.pair) if args.pair else None,
            prepare_only=args.prepare_only,
            gold=args.gold,
            family_id=args.family,
            rfc_id=args.rfc,
            clause_source=args.clause_source,
            prepare_all=args.prepare_all,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        LOG.error("%s", exc)
        return 2
    return 0
