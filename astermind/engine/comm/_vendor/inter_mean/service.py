"""Compose analytical hop means into whole-message routed service stages.

Link fields are runtime ISLSim/topology outputs, not configurable model rates.
The C++ executor shares each stage's expected forward resource work.
This planner starts no flows and samples no error or retry outcome.
"""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import Field, StrictInt, model_validator

from .mean import MAX_U64, predict_mean_hop
from .link import InterLinkState
from .link import ReferenceSettings
from .settings import InterMeanSettings

MEAN_EVALUATION = "analytical_staged_isl_mean"
MEAN_PATH_POLICY = "whole_message_store_and_forward_mean"
Index = Annotated[StrictInt, Field(ge=0, le=(1 << 63) - 1)]


class MeanDirectedLink(ReferenceSettings):
    index: Index
    source: Annotated[str, Field(min_length=1, max_length=256)]
    target: Annotated[str, Field(min_length=1, max_length=256)]
    bandwidth_gbps: Annotated[float, Field(gt=0)]
    latency_ns: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def validate_endpoints(self):
        if self.source == self.target:
            raise ValueError("A resource link must have different endpoints")
        return self


class MeanPathLink(MeanDirectedLink):
    is_inter_satellite: bool
    reverse: MeanDirectedLink | None = None

    @model_validator(mode="after")
    def validate_reverse(self):
        if self.reverse is not None and (
            not self.is_inter_satellite
            or self.reverse.index == self.index
            or self.reverse.source != self.target
            or self.reverse.target != self.source
        ):
            raise ValueError("Reverse resource must be the exact opposite ISL")
        return self


class MeanPathInput(ReferenceSettings):
    data_bytes: Annotated[StrictInt, Field(ge=0, le=MAX_U64)]
    links: list[MeanPathLink] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_route(self):
        if len({link.index for link in self.links}) != len(self.links):
            raise ValueError("Route resource indices must be unique")
        if not any(link.is_inter_satellite for link in self.links):
            raise ValueError("Mean inter-satellite route requires at least one ISL")
        for first, second in zip(self.links, self.links[1:], strict=False):
            if first.target != second.source:
                raise ValueError("Route resources must form a connected directed path")
        return self


def _clock(value, name):
    if not math.isfinite(value) or not 0 <= value < MAX_U64:
        raise ValueError(f"{name} is outside the simulator clock range")
    return value


def _stage(links, size, work, processing, prediction=None):
    resources = (
        [
            {"index": link.index, "bytes": value}
            for link, value in zip(links, work, strict=True)
        ]
        if size
        else []
    )
    if any(
        not math.isfinite(value) or not 0 <= value < MAX_U64 or (size and value == 0)
        for value in work
    ):
        raise ValueError("Mean resource work is outside the supported range")
    propagation = math.fsum(link.latency_ns for link in links) if size else 0.0
    serial = max(
        (
            8 * value / link.bandwidth_gbps
            for link, value in zip(links, work, strict=True)
        ),
        default=0.0,
    )
    result = {
        "kind": "isl" if prediction is not None else "local",
        "source": links[0].source,
        "target": links[-1].target,
        "link_indices": [link.index for link in links],
        "payload_bytes": size,
        "service_bytes": size,
        "resources": resources,
        "startup_ns": 0.0,
        "processing_ns": processing,
        "propagation_ns": propagation,
        "isolated_serialization_ns": serial,
        "isolated_ns": _clock(serial + propagation + processing, "Stage mean"),
    }
    if prediction is not None:
        result["hop_prediction"] = prediction
    return result


def build_mean_path_plan(request: MeanPathInput, settings: InterMeanSettings):
    """Use expected work plus fixed tails, with no packet/retry execution.

    ISLs are whole-message stages in route order. Contiguous local access links
    are a pipelined fluid segment carrying application bytes. Recovery feedback
    time uses nominal reverse capacity; it adds no reverse resource demand.
    Forward expected retry work is subject to runtime allocation.
    This is an explicit approximation, not an exact queueing/ARQ expectation.
    """
    stages, local = [], []
    size = request.data_bytes
    hop_settings = settings.model_copy(update={"startup_us": 0.0})

    def flush_local():
        if local:
            stages.append(_stage(local, size, [float(size)] * len(local), 0.0))
            local.clear()

    for link in request.links:
        if not link.is_inter_satellite:
            local.append(link)
            continue
        flush_local()
        reverse = (
            InterLinkState(link.reverse.bandwidth_gbps, link.reverse.latency_ns)
            if link.reverse
            else InterLinkState(0.0, 0.0, False)
        )
        prediction = predict_mean_hop(
            size,
            InterLinkState(link.bandwidth_gbps, link.latency_ns),
            reverse,
            hop_settings,
        )
        timing = prediction["timing_ns"]
        # Evaluated at nominal mapped capacity; convert both first-pass and
        # expected retry serialization back into raw resource-time-equivalent B.
        work = (
            (
                prediction["equivalent_resource_bytes"]
                + timing["retry_serialization_ns"] * link.bandwidth_gbps / 8
            )
            if size
            else 0.0
        )
        fixed = math.fsum(
            timing[name]
            for name in (
                "processing_ns",
                "fec_pipeline_ns",
                "retry_feedback_ns",
                "source_reload_ns",
            )
        )
        stages.append(_stage([link], size, [work], fixed, prediction))
    flush_local()
    startup = _clock(settings.startup_us * 1000, "Startup") if size else 0.0
    stages[0]["startup_ns"] = startup
    stages[0]["isolated_ns"] = _clock(
        stages[0]["isolated_ns"] + startup, "First stage mean"
    )
    isolated = _clock(math.fsum(stage["isolated_ns"] for stage in stages), "Path mean")
    return {
        "schema_version": 1,
        "model": "inter_surrogate_mean",
        "evaluation": MEAN_EVALUATION,
        "path_policy": MEAN_PATH_POLICY,
        "progress_basis": "application_bytes",
        "bandwidth_source": "islsim",
        "bandwidth_basis": "net_after_existing_fec",
        "profile": settings.profile,
        "payload_bytes": size,
        "service_bytes": size,
        "time_unit": "ns",
        "link_indices": [link.index for link in request.links],
        "isl_count": sum(link.is_inter_satellite for link in request.links),
        "stages": stages,
        "isolated_ns": isolated,
        "completion_delay_ns": math.ceil(isolated),
        "sampled_errors": False,
        "scheduled_ack_or_retry_flows": False,
        "feedback_resource_policy": "nominal_reverse_time_only",
    }

