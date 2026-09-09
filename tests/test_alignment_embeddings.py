"""Exercise batching, windows and cache validation without downloading a model."""

import json
from pathlib import Path

import numpy as np
import pytest

from alignment.version_alignment.config import Config
from alignment.version_alignment.embeddings import (
    Embedder,
    effective_token_limit,
    token_windows,
)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))

    def decode(self, ids, **kwargs):
        raise AssertionError("Encoding must never decode or retokenize windows")

    def prepare_for_model(self, ids, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [101, *ids, 102], "attention_mask": [1] * (len(ids) + 2)}

    def num_special_tokens_to_add(self, **kwargs):
        return 2


class Model:
    max_seq_length = 8
    tokenizer = Tokenizer()

    def get_embedding_dimension(self):
        return 2


class FakeEmbedder(Embedder):
    def _encode_prepared(self, prepared, effective_limit):
        assert max(len(x["input_ids"]) for x in prepared) <= effective_limit
        assert all(
            x["input_ids"][0] == 101 and x["input_ids"][-1] == 102 for x in prepared
        )
        return np.array([[1, 0] for _ in prepared], dtype=np.float32)


def test_long_windows_cache_and_corruption(tmp_path):
    encoder = FakeEmbedder(Config(backend="lexical", window_overlap=2), tmp_path)
    encoder.model = Model()
    vectors = encoder.encode(["word " * 15, "word"])
    assert encoder.long_count == 1
    np.testing.assert_array_equal(vectors, [[1, 0], [1, 0]])
    np.testing.assert_array_equal(encoder.encode(["word " * 15, "word"]), vectors)
    next(tmp_path.glob("*.npy")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        encoder.encode(["word " * 15, "word"])


def test_lexical_does_not_load_model(tmp_path):
    encoder = Embedder(Config(backend="lexical"), tmp_path)
    assert encoder.encode(["text"]) is None


@pytest.mark.parametrize("length", [0, 1, 6, 7, 23, 10000])
@pytest.mark.parametrize("overlap", [0, 2, 5])
def test_boundaries_termination_and_exact_coverage(length, overlap):
    ids = list(range(length))
    chunks = list(token_windows(ids, 6, overlap))
    assert len(chunks) <= length
    assert all(part == ids[a:b] and 0 < b - a <= 6 for part, a, b in chunks)
    assert set().union(*(set(range(a, b)) for _, a, b in chunks)) == set(range(length))
    assert all(chunks[i + 1][1] > chunks[i][1] for i in range(len(chunks) - 1))
    if chunks:
        assert chunks[-1][2] == length


@pytest.mark.parametrize("limit,overlap", [(0, 0), (6, 6), (6, -1)])
def test_invalid_window_settings_fail(limit, overlap):
    with pytest.raises(ValueError):
        list(token_windows([1, 2], limit, overlap))


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_blank_clause_is_error_not_silent_skip(tmp_path, text):
    encoder = FakeEmbedder(Config(backend="lexical", window_overlap=2), tmp_path)
    encoder.model = Model()
    with pytest.raises(ValueError, match="Empty/whitespace"):
        encoder.encode([text])
    assert encoder.encode([]).shape == (0, 2)


def test_special_tokens_count_toward_effective_limit(tmp_path):
    encoder = FakeEmbedder(Config(backend="lexical", window_overlap=2), tmp_path)
    encoder.model = Model()
    encoder.encode(["word " * 6, "word " * 7])
    assert [w["encoded_tokens"] for w in encoder.last_windows] == [8, 8, 5]
    assert encoder.stats[0]["long_clauses"] == 1
    encoder.model.tokenizer.model_max_length = 7
    assert effective_token_limit(encoder.model) == 7
    del encoder.model.tokenizer.model_max_length


def test_tls08_original_failing_clause_token_ids():
    path = Path(__file__).parent / "fixtures/alignment/tls08_overlength.json"
    fixture = json.loads(path.read_text())
    assert fixture["clause_id"] == "draft-ietf-tls-tls13-08:A.4:b0799:c01"
    assert fixture["old_decoded_token_count"] > 254
    ids = fixture["token_ids"]
    chunks = list(token_windows(ids, 254, 32))
    assert len(ids) == 935
    assert all(part == ids[a:b] and len(part) + 2 <= 256 for part, a, b in chunks)
    assert set().union(*(set(range(a, b)) for _, a, b in chunks)) == set(range(935))


def test_real_pinned_encoder_tls08_and_protocol_punctuation(tmp_path):
    """Optional local integration; never downloads, and exercises real forward batches."""
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    snapshot = (
        Path.home()
        / f".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/{revision}"
    )
    if not (snapshot / "model.safetensors").exists():
        pytest.skip("Pinned local MiniLM snapshot unavailable; no network allowed")
    encoder = Embedder(Config(model_revision=revision, batch_size=2), tmp_path)
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/alignment/tls08_overlength.json").read_text()
    )
    texts = [
        fixture["text"],
        "[CLS] uint16 length; opaque field<0..2^16-1>; MUST NOT send 0xFF! " * 80,
    ]
    actual_ids = encoder.model.tokenizer.encode(
        texts[0], add_special_tokens=False, verbose=False
    )
    assert actual_ids == fixture["token_ids"]
    # Explicitly reproduce the old decode expansion before testing direct-ID inference.
    old = encoder.model.tokenizer.decode(
        actual_ids[fixture["window_start"] : fixture["window_start"] + 254],
        clean_up_tokenization_spaces=False,
    )
    assert (
        len(
            encoder.model.tokenizer.encode(old, add_special_tokens=False, verbose=False)
        )
        > 254
    )
    result = encoder.encode(texts)
    assert result.shape == (2, 384) and np.isfinite(result).all()
    assert max(w["encoded_tokens"] for w in encoder.last_windows) <= 256
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)


def test_validation_ignores_serialized_batch_padding_and_truncation(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    from alignment.version_alignment.validate_run import content_tokenizer

    tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "client": 2}))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_padding(length=128, pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=8)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    # Reproduce the false truncation alarm: serialized padding adds 127 fake IDs.
    assert len(Tokenizer.from_file(str(path)).encode("client").ids) == 128
    restored = content_tokenizer(path)
    assert restored.encode("client", add_special_tokens=False).ids == [2]
    assert restored.encode("client " * 935, add_special_tokens=False).ids == [2] * 935
    assert restored.encode(" \t\n", add_special_tokens=False).ids == []
