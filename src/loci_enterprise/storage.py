"""Blob storage for original files, addressed by content.

The document store keeps extracted text; the original file still has to live
somewhere — for re-parsing when a parser improves, for audit, and for the
"download the source" button every knowledge base eventually needs.

Content addressing (SHA-256 of the bytes) buys three properties cheaply:

* **deduplication**: the same PDF uploaded twice costs one copy;
* **integrity**: reading a blob verifies the hash, so silent corruption is a
  loud error instead of a wrong answer;
* **immutable identity**: a stored blob never changes, so a document version
  can always point back to exactly the bytes it was parsed from.

Two backends ship: a local filesystem (default; sharded by hash prefix so a
directory with a million files stays usable) and an in-memory one for tests.
An object-store backend (S3-compatible) implements the same three methods.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Blob:
    digest: str
    size: int
    media_type: str = ""

    @property
    def key(self) -> str:
        return self.digest


class BlobStore:
    name = "blob"

    def put(self, data: bytes, *, media_type: str = "") -> Blob:
        raise NotImplementedError

    def get(self, digest: str) -> bytes:
        raise NotImplementedError

    def exists(self, digest: str) -> bool:
        raise NotImplementedError

    def delete(self, digest: str) -> bool:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError


def digest_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalBlobStore(BlobStore):
    """Files under root/<aa>/<bb>/<digest> — two levels keep listings small."""

    name = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("digest must be 64 lowercase hex characters")
        return self.root / digest[:2] / digest[2:4] / digest

    def put(self, data: bytes, *, media_type: str = "") -> Blob:
        digest = digest_of(data)
        path = self._path(digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # write to a temp file in the same directory, then rename: a crash
            # mid-write must not leave a truncated blob that looks valid
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        return Blob(digest=digest, size=len(data), media_type=media_type)

    def get(self, digest: str) -> bytes:
        data = self._path(digest).read_bytes()
        if digest_of(data) != digest:
            raise IOError(f"blob {digest[:12]} is corrupt (hash mismatch)")
        return data

    def exists(self, digest: str) -> bool:
        try:
            return self._path(digest).is_file()
        except ValueError:
            return False

    def delete(self, digest: str) -> bool:
        path = self._path(digest)
        if path.is_file():
            path.unlink()
            return True
        return False

    def stats(self) -> dict:
        blobs = [p for p in self.root.rglob("*") if p.is_file()
                 and not p.name.startswith(".tmp-")]
        return {"backend": self.name, "blobs": len(blobs),
                "bytes": sum(p.stat().st_size for p in blobs)}


class MemoryBlobStore(BlobStore):
    name = "memory"

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self._types: dict[str, str] = {}

    def put(self, data: bytes, *, media_type: str = "") -> Blob:
        digest = digest_of(data)
        self._data.setdefault(digest, data)
        if media_type:
            self._types[digest] = media_type
        return Blob(digest=digest, size=len(data), media_type=media_type)

    def get(self, digest: str) -> bytes:
        if digest not in self._data:
            raise KeyError(digest)
        return self._data[digest]

    def exists(self, digest: str) -> bool:
        return digest in self._data

    def delete(self, digest: str) -> bool:
        return self._data.pop(digest, None) is not None

    def stats(self) -> dict:
        return {"backend": self.name, "blobs": len(self._data),
                "bytes": sum(len(v) for v in self._data.values())}


def store_file(blobs: BlobStore, path: str | Path) -> tuple[Blob, bytes]:
    """Read a file once, store it, and return both the blob and its bytes."""
    data = Path(path).read_bytes()
    return blobs.put(data, media_type=_guess_type(Path(path).suffix)), data


def _guess_type(suffix: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".json": "application/json",
        ".html": "text/html",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".epub": "application/epub+zip",
    }.get(suffix.lower(), "application/octet-stream")
