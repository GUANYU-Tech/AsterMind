from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field


Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_.:/-]+$")]


Nanoseconds = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]


Bandwidth = Annotated[float, Field(ge=1e-6, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
