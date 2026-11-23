from __future__ import annotations

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_.:/-]+$")]


Nanoseconds = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]


Bandwidth = Annotated[float, Field(ge=1e-6, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComputeModeConfig(Contract):
    compute_mode: Literal["midlayer", "roofline", "detail_mode", "fast_mode"] = "midlayer"
    detail_prediction_method: Literal["midlayer", "roofline"] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_compute_mode(cls, value):
        if not isinstance(value, dict):
            return value
        legacy = value.get("compute_mode")
        if "compute_mode" not in value and value.get("detail_prediction_method") is not None:
            return {**value, "compute_mode": "detail_mode"}
        if legacy in ("midlayer", "roofline"):
            explicit = value.get("detail_prediction_method")
            if explicit is not None and explicit != legacy:
                raise ValueError("legacy compute_mode conflicts with detail_prediction_method")
        if legacy == "batch_roofline":
            return {**value, "compute_mode": "fast_mode"}
        return value

    @model_validator(mode="after")
    def check_detail_prediction_method(self):
        if self.compute_mode == "fast_mode" and self.detail_prediction_method is not None:
            raise ValueError("detail_prediction_method requires detail_mode")
        return self

    @property
    def executor_compute_mode(self):
        if self.compute_mode == "detail_mode":
            return self.detail_prediction_method or "midlayer"
        return self.compute_mode
