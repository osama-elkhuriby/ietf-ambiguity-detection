from datetime import date
from pathlib import Path

import pytest

from alignment.version_alignment import TLS_FAMILY
from alignment.version_alignment.inputs import prepare_inputs, timeline
from alignment.version_alignment.normalize import normalize


def test_timeline_rejects_gap_duplicate_and_reverse_dates():
    def metadata(revs):
        return {
            TLS_FAMILY: {
                "versions": [
                    {"rev": r, "submission_date": d, "local_txt_path": "x"}
                    for r, d in revs
                ]
            }
        }

    for revs in [
        [("00", "2020-01-01"), ("02", "2020-02-01")],
        [("00", "2020-01-01"), ("00", "2020-01-01")],
        [("00", "2020-02-01"), ("01", "2020-01-01")],
    ]:
        with pytest.raises(ValueError):
            timeline(metadata(revs), {})
    result = timeline(metadata([("01", "2020-02-01"), ("00", "2020-01-01")]), {})
    assert [r["revision"] for r in result] == ["00", "01"]


def test_month_precision():
    metadata = {
        TLS_FAMILY: {
            "versions": [
                {"rev": "00", "submission_date": "2020-01-01", "local_txt_path": "x"}
            ]
        }
    }
    rfcs = {
        "RFC8446": {
            "draft": TLS_FAMILY + "-00",
            "date": "2020-02-01",
            "local_txt_path": "rfc",
        }
    }
    assert timeline(metadata, rfcs)[-1]["available_at"] == date(2020, 2, 29)


@pytest.mark.parametrize(
    "text",
    [
        "The client MUST NOT send 0x01 before 30 ms.",
        "The server SHOULD NOT change field_name; see [RFC8446], Section 4.2.",
    ],
)
def test_normalization_preserves_protected_tokens(text):
    assert normalize(text.replace(" ", " \n "), "paragraph")[0] == text


def test_normalization_formal_and_list():
    assert (
        normalize("  uint16 x;\r\n  opaque y;", "formal_syntax")[0]
        == "  uint16 x;\n  opaque y;"
    )
    assert normalize("- client MUST send", "list_item")[0] == "client MUST send"
    assert normalize("1. client MUST send", "list_item")[0].startswith("1.")
    assert normalize("cafe\u0301", "paragraph")[0] == "caf\u00e9"


def test_real_tls_input_snapshots_are_exclusive(tmp_path):
    root = Path(__file__).resolve().parents[1]
    selected = {TLS_FAMILY + "-22", "RFC8446"}
    documents, info = prepare_inputs(
        root, tmp_path, prepare_missing=True, selected=selected
    )
    assert len(documents) == 2 and all(d.prepared for d in documents)
    assert set(info["prepared_documents"]) == selected
    assert all(d.clauses for d in documents)
    with pytest.raises(FileExistsError):
        prepare_inputs(root, tmp_path, prepare_missing=True, selected=selected)
