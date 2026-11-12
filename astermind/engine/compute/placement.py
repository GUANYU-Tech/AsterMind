"""Map logical model ranks to physical GPU devices."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RankPlacement:
    cell_id: str
    pool_name: str
    rank: int
    satellite_id: str
    gpu_device_id: str

    def __post_init__(self):
        for name in ("cell_id", "pool_name", "satellite_id", "gpu_device_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("rank must be a nonnegative integer")
