"""Authoritative integer attosecond clock; float seconds are a display projection.

Model durations are rounded upward once at the clock boundary. Arithmetic on
Time retains integer ticks, including differences at large absolute times.
"""

from decimal import Decimal, ROUND_CEILING
import math

TICKS_PER_SECOND = 10**18
TICKS_PER_NS = 10**9
CLOCK = {
    "schema": "orbitserve.clock.v1",
    "unit": "attosecond",
    "ticks_per_second": str(TICKS_PER_SECOND),
    "rounding": "ceil_duration",
    "authority": "integer_ticks",
}


def ticks(value):
    if isinstance(value, Time):
        return value.ticks
    if isinstance(value, bool):
        raise ValueError("time must be finite")
    if isinstance(value, int):
        return value * TICKS_PER_SECOND
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError("time must be finite")
    numerator, denominator = decimal.as_integer_ratio()
    return -(-numerator * TICKS_PER_SECOND // denominator)


class Time(float):
    """Seconds compatible with physics APIs, with exact scheduler arithmetic."""

    def __new__(cls, value=0):
        return cls.from_ticks(ticks(value))

    @classmethod
    def from_ticks(cls, value):
        result = float.__new__(cls, value / TICKS_PER_SECOND)
        result.ticks = int(value)
        return result

    @classmethod
    def from_ns(cls, value):
        return cls.from_ticks(
            int(
                (Decimal(str(value)) * TICKS_PER_NS).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
        )

    def __add__(self, other):
        return Time.from_ticks(self.ticks + ticks(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Time.from_ticks(self.ticks - ticks(other))

    def __rsub__(self, other):
        return Time.from_ticks(ticks(other) - self.ticks)

    def __lt__(self, other):
        return (
            self.ticks < ticks(other) if math.isfinite(other) else float(self) < other
        )

    def __le__(self, other):
        return (
            self.ticks <= ticks(other) if math.isfinite(other) else float(self) <= other
        )

    def __gt__(self, other):
        return (
            self.ticks > ticks(other) if math.isfinite(other) else float(self) > other
        )

    def __ge__(self, other):
        return (
            self.ticks >= ticks(other) if math.isfinite(other) else float(self) >= other
        )

    def __eq__(self, other):
        if isinstance(other, Time):
            return self.ticks == other.ticks
        return float.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    __hash__ = float.__hash__

    def __reduce__(self):
        return Time.from_ticks, (self.ticks,)


class ClockLog(list):
    """Keep exact timestamps beside the JSON-compatible seconds projection."""

    def append(self, row):
        stamp(row)
        super().append(row)


def stamp(row):
    for key, value in tuple(row.items()):
        if isinstance(value, Time):
            row[key.removesuffix("_s") + "_ticks"] = str(value.ticks)
    return row
