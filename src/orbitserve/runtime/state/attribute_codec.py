"""Portable numeric-column encoding; legacy JSON blocks remain readable."""

import json
import struct
import sys
from array import array

from .attribute_cache import AttributeBlock, _Fields

MAGIC = b"OSAB2\x00"


def encode(block):
    arrays, indices, descriptors = [], {}, []

    def column(value):
        if id(value) not in indices:
            indices[id(value)] = len(arrays)
            data = array(value.typecode, value)
            if sys.byteorder != "little":
                data.byteswap()
            arrays.append(data.tobytes())
            descriptors.append([value.typecode, len(value)])
        return indices[id(value)]

    def fields(value):
        return {
            "constants": value.constants,
            "columns": {
                key: {"array": column(values)}
                if isinstance(values, array)
                else {"values": values}
                for key, values in value.columns.items()
            },
        }

    header = {
        "times": column(block.times),
        "group_ids": column(block.group_ids),
        "row_ids": column(block.row_ids),
        "groups": [
            [fields(g), [fields(d) for d in directions]]
            for g, directions in block.groups
        ],
        "arrays": descriptors,
    }
    metadata = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<I", len(metadata)) + metadata + b"".join(arrays)


def decode(data):
    if not data.startswith(MAGIC) or len(data) < len(MAGIC) + 4:
        raise ValueError("invalid physical column block")
    length = struct.unpack_from("<I", data, len(MAGIC))[0]
    offset = len(MAGIC) + 4
    header = json.loads(data[offset : offset + length])
    offset += length
    arrays = []
    for code, count in header["arrays"]:
        if (
            code not in {"d", "q", "H"}
            or type(count) is not int
            or not 0 < count <= 256
        ):
            raise ValueError("invalid physical column shape")
        values = array(code)
        end = offset + count * values.itemsize
        if end > len(data):
            raise ValueError("truncated physical column block")
        values.frombytes(data[offset:end])
        if sys.byteorder != "little":
            values.byteswap()
        arrays.append(values)
        offset = end
    if offset != len(data):
        raise ValueError("trailing physical column bytes")

    def fields(value):
        result = _Fields.__new__(_Fields)
        result.constants = value["constants"]
        result.columns = {
            key: arrays[column["array"]]
            if "array" in column
            else tuple(column["values"])
            for key, column in value["columns"].items()
        }
        return result

    block = AttributeBlock.__new__(AttributeBlock)
    block.times, block.group_ids, block.row_ids = (
        arrays[header[name]] for name in ("times", "group_ids", "row_ids")
    )
    block.groups = [(fields(g), [fields(d) for d in ds]) for g, ds in header["groups"]]
    if (
        block.times.typecode != "d"
        or block.group_ids.typecode != "H"
        or block.row_ids.typecode != "H"
        or len(block.times) != len(block.group_ids)
        or len(block.times) != len(block.row_ids)
    ):
        raise ValueError("invalid physical row index")
    max_rows = [-1] * len(block.groups)
    for i, group in enumerate(block.group_ids):
        if group >= len(block.groups):
            raise ValueError("invalid physical group index")
        max_rows[group] = max(max_rows[group], block.row_ids[i])
    for group, row in enumerate(max_rows):
        for f in (block.groups[group][0], *block.groups[group][1]):
            if any(row >= len(v) for v in f.columns.values()):
                raise ValueError("invalid physical column index")
    block._charge()
    return block
