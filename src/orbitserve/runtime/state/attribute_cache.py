"""Compact numeric columns for decoded, immutable physical attribute blocks."""

import sys
import math
from array import array


def _canonical(value):
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    return value


class _Fields:
    def __init__(self, rows, shared):
        self.constants = {}
        self.columns = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            first = values[0]
            if all(type(v) is type(first) and v == first for v in values):
                self.constants[key] = _canonical(first)
                continue
            if all(type(v) is float for v in values):
                column = array("d", values)
            elif all(type(v) is int and -(2**63) <= v < 2**63 for v in values):
                column = array("q", values)
            else:
                self.columns[key] = tuple(_canonical(v) for v in values)
                continue
            identity = column.typecode, column.tobytes()
            self.columns[key] = shared.setdefault(identity, column)

    def at(self, index):
        return {
            **_canonical(self.constants),
            **{key: _canonical(values[index]) for key, values in self.columns.items()},
        }

    def value(self, key, index):
        return (
            self.constants[key] if key in self.constants else self.columns[key][index]
        )


class AttributeBlock:
    """Retain one copy of constants and equal directional numeric columns."""

    def __init__(self, rows):
        self.times = array("d", (row[0] for row in rows))
        if (not 0 < len(self.times) <= 256 or any(not math.isfinite(t) or t < 0 for t in self.times)
                or any(b <= a for a, b in zip(self.times, self.times[1:]))):
            raise ValueError('physical block times must be finite, nonnegative and strictly increasing')
        self.group_ids = array("H", [0] * len(rows))
        self.row_ids = array("H", [0] * len(rows))
        groups = {}
        for index, (_, row) in enumerate(rows):
            shape = (tuple(row["geometry"]), *(tuple(d) for d in row["directions"]))
            groups.setdefault(shape, []).append((index, row))
        self.groups = []
        shared = {}
        for group_id, items in enumerate(groups.values()):
            for row_id, (index, _) in enumerate(items):
                self.group_ids[index], self.row_ids[index] = group_id, row_id
            geometry = _Fields([row["geometry"] for _, row in items], shared)
            directions = [
                _Fields([row["directions"][i] for _, row in items], shared)
                for i in range(len(items[0][1]["directions"]))
            ]
            self.groups.append((geometry, directions))
        self._charge()

    def _charge(self):
        # Conservative charge for numeric buffers, boxed fallback columns and
        # constant metadata. The cache has a fixed byte budget, not row count.
        fields = [f for g, ds in self.groups for f in (g, *ds)]
        def size(value, seen):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            result = sys.getsizeof(value)
            if isinstance(value, dict):
                result += sum(size(k, seen) + size(v, seen) for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                result += sum(size(v, seen) for v in value)
            return result

        self.nbytes = (
            len(self.times) * 12
            + sum(len(v) * v.itemsize for v in {
                id(v): v for f in fields for v in f.columns.values()
                if isinstance(v, array)
            }.values())
            + sum(
                size(f.constants, set())
                + 1024
                + sum(len(v) * 40 for v in f.columns.values() if isinstance(v, tuple))
                for f in fields
            )
        )

    def geometry(self, index):
        return self.groups[self.group_ids[index]][0].at(self.row_ids[index])

    def geometry_value(self, index, key):
        return self.groups[self.group_ids[index]][0].value(key, self.row_ids[index])

    def direction(self, index, source, target):
        row = self.row_ids[index]
        for fields in self.groups[self.group_ids[index]][1]:
            if (
                fields.value("source_node_id", row) == source
                and fields.value("target_node_id", row) == target
            ):
                return fields.at(row)
        raise ValueError("physical attribute direction missing")

    def at(self, index):
        geometry, directions = self.groups[self.group_ids[index]]
        row = self.row_ids[index]
        return {
            "geometry": geometry.at(row),
            "directions": [fields.at(row) for fields in directions],
        }
