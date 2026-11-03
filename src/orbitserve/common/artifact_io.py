"""Atomic artifact writes with SHA-256 calculated over the bytes being written."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


class HashingWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data):
        count = self.stream.write(data)
        self.digest.update(memoryview(data)[:count])
        self.size += count
        return count

    def tell(self):
        return self.stream.tell()

    def flush(self):
        self.stream.flush()

    @property
    def sha256(self):
        return self.digest.hexdigest()


@contextmanager
def atomic_output(path):
    """Publish complete bytes on success; each writer owns its temporary file.

    Writers targeting the same destination may finish in either order. The last
    successful replace wins, without mixing bytes or deleting a peer's output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name[:80]}.",
                                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            output = HashingWriter(stream)
            yield output
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with atomic_output(path) as output:
        output.write(data)
    return output.sha256
