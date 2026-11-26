"""Content-addressed blobs: dedup, integrity, sharding, deletion."""
from __future__ import annotations

import pytest

from loci_enterprise.storage import (Blob, LocalBlobStore, MemoryBlobStore,
                                     digest_of, store_file)


@pytest.fixture(params=["memory", "local"])
def blobs(request, tmp_path):
    return MemoryBlobStore() if request.param == "memory" else LocalBlobStore(tmp_path / "blobs")


def test_put_returns_content_hash_and_size(blobs):
    blob = blobs.put(b"hello")
    assert blob.digest == digest_of(b"hello") and blob.size == 5


def test_get_round_trip(blobs):
    blobs.put(b"policy text")
    assert blobs.get(digest_of(b"policy text")) == b"policy text"


def test_same_content_is_stored_once(blobs):
    a = blobs.put(b"same")
    b = blobs.put(b"same")
    assert a.digest == b.digest
    assert blobs.stats()["blobs"] == 1


def test_exists(blobs):
    blob = blobs.put(b"x")
    assert blobs.exists(blob.digest)
    assert not blobs.exists("0" * 64)


def test_delete(blobs):
    blob = blobs.put(b"x")
    assert blobs.delete(blob.digest)
    assert not blobs.exists(blob.digest)
    assert not blobs.delete(blob.digest)


def test_missing_blob_raises(blobs):
    with pytest.raises((KeyError, FileNotFoundError)):
        blobs.get("a" * 64)


def test_corrupt_blob_is_detected(tmp_path):
    store = LocalBlobStore(tmp_path / "b")
    blob = store.put(b"original")
    path = store._path(blob.digest)
    path.write_bytes(b"tampered")
    with pytest.raises(IOError):
        store.get(blob.digest)


def test_local_store_shards_by_hash_prefix(tmp_path):
    store = LocalBlobStore(tmp_path / "b")
    blob = store.put(b"shard me")
    path = store._path(blob.digest)
    assert path.parent.name == blob.digest[2:4]
    assert path.parent.parent.name == blob.digest[:2]


def test_local_store_rejects_path_traversal(tmp_path):
    store = LocalBlobStore(tmp_path / "b")
    with pytest.raises(ValueError):
        store._path("../../etc/passwd")


def test_store_file_guesses_media_type(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text("# 政策", encoding="utf-8")
    store = MemoryBlobStore()
    blob, data = store_file(store, p)
    assert blob.media_type == "text/markdown"
    assert data == p.read_bytes()


def test_stats_report_counts_and_bytes(blobs):
    blobs.put(b"aa")
    blobs.put(b"bbb")
    stats = blobs.stats()
    assert stats["blobs"] == 2 and stats["bytes"] == 5
