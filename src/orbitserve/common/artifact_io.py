"""Atomic artifact writes with SHA-256 calculated over the bytes being written."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            output = HashingWriter(stream)
            yield output
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
