"""Structured storage layout and path helpers for Stage 1 corpus data."""
from __future__ import annotations

from pathlib import Path

from paths import REPO_ROOT, repo_path

RAW_DRAFTS_DIR = repo_path("corpus", "raw", "drafts")
RAW_RFCS_DIR = repo_path("corpus", "raw", "rfcs")
RAW_ERRATA_DIR = repo_path("corpus", "raw", "errata")
RFC_INDEX_CACHE_DIR = repo_path("corpus", "raw", "rfcs", ".rfcindex-cache")

DRAFTS_METADATA_PATH = repo_path("corpus", "metadata", "drafts.json")
RFCS_METADATA_PATH = repo_path("corpus", "metadata", "rfcs.json")
LINKS_METADATA_PATH = repo_path("corpus", "metadata", "draft_rfc_links.json")
ERRATA_METADATA_PATH = repo_path("corpus", "metadata", "errata.json")
FINAL_RFC_MANIFEST_PATH = repo_path("corpus", "metadata", "final_rfc_manifest.csv")
VERIFIED_TECHNICAL_ERRATA_PATH = repo_path(
    "corpus", "metadata", "verified_technical_errata.json"
)

# Curated mapping from manifest draft families to evaluation protocol areas.
PROTOCOL_AREA: dict[str, str] = {
    "draft-ietf-ospf-vers2": "routing",
    "draft-ietf-idr-bgp4": "routing",
    "draft-ietf-6man-rfc2460bis": "routing",
    "draft-ietf-tls-tls13": "security",
    "draft-ietf-quic-tls": "security",
    "draft-ietf-quic-transport": "transport",
    "draft-ietf-quic-recovery": "transport",
    "draft-ietf-tcpm-rfc793bis": "transport",
    "draft-ietf-dnsop-svcb-https": "dns",
    "draft-ietf-httpbis-semantics": "application",
    "draft-ietf-httpbis-http2bis": "application",
    "draft-ietf-quic-http": "application",
}

# Pre-Datatracker-era families: RFC plaintext + errata only (no revision history).
RFC_ONLY_DRAFT_FAMILIES: frozenset[str] = frozenset(
    {
        "draft-ietf-ospf-vers2",
        "draft-ietf-idr-bgp4",
    }
)


def draft_collection_mode(draft_family: str, revision_count_expected: int | None = None) -> str:
    """Return ``full-history`` or ``rfc-only`` for manifest/audit reporting."""
    if draft_family in RFC_ONLY_DRAFT_FAMILIES:
        return "rfc-only"
    if revision_count_expected == 0:
        return "rfc-only"
    return "full-history"


def to_repo_relative(path: Path | str | None) -> str | None:
    """Return a repo-relative POSIX path when the file is under REPO_ROOT."""
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_local_path(stored: str | None) -> Path | None:
    """Resolve a stored path (repo-relative or absolute) on the local filesystem."""
    if not stored:
        return None
    candidate = Path(stored)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def expected_draft_dir(draft_name: str) -> Path:
    return RAW_DRAFTS_DIR / draft_name


def expected_rfc_path(rfc_id: str) -> Path:
    from rfc_ids import normalize_rfc_id

    return RAW_RFCS_DIR / f"{normalize_rfc_id(rfc_id)}.txt"


def relocate_misplaced_rfc_index_cache() -> bool:
    """Move legacy corpus/corpus/raw/rfcs/.rfcindex-cache into the canonical tree."""
    legacy_cache = repo_path("corpus", "corpus", "raw", "rfcs", ".rfcindex-cache")
    if not legacy_cache.exists():
        return False

    RFC_INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for item in legacy_cache.iterdir():
        target = RFC_INDEX_CACHE_DIR / item.name
        if target.exists():
            continue
        item.replace(target)

    # Remove empty legacy directories if possible.
    for parent in (
        legacy_cache,
        legacy_cache.parent,
        legacy_cache.parent.parent,
        legacy_cache.parent.parent.parent,
    ):
        try:
            parent.rmdir()
        except OSError:
            break
    return True
