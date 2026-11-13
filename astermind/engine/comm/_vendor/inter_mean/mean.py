"""ISLSim-fed serialization and analytical selective-repeat recovery means.

No events, samples, ACK flows or retry state are created. This single-hop
kernel feeds the selectable AsterMind analytical inter-satellite backend.
Its receiver-delivery mean is exact under the independent batch-round policy,
up to an explicitly bounded summation error, not for arbitrary sliding windows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .link import (
    InterLinkState,
    InterLinkUnavailableError,
)
from .settings import InterMeanSettings

MAX_U64 = (1 << 64) - 1


class InterMeanIncompleteError(ValueError):
    """The selected reliable reference cannot yield a finite completion mean."""


def _uint(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= MAX_U64:
        raise ValueError(f"{name} is outside uint64 range")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} exceeds numeric range") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _any_error(probability, trials):
    if probability == 0 or trials == 0:
        return 0.0
    if probability == 1:
        return 1.0
    return -math.expm1(trials * math.log1p(-probability))


def rs_codeword_failure_probability(ber, n, correctable_symbols, symbol_bits):
    """Ideal bounded-distance RS under independent bit/symbol errors.

    A codeword is declared bad when more than t symbols are erroneous. This
    does not model burst correlation, miscorrection or a measured BER curve.
    """
    probability = _finite(ber, "BER")
    if probability > 1:
        raise ValueError("BER exceeds one")
    length = _uint(n, "RS n")
    limit = _uint(correctable_symbols, "RS t")
    bits = _uint(symbol_bits, "RS symbol bits")
    if not length or length > 65535 or not bits or limit >= length:
        raise ValueError("Unsupported RS probability dimensions")
    symbol_error = _any_error(probability, bits)
    if symbol_error in (0, 1):
        return symbol_error
    log_p, log_good = math.log(symbol_error), math.log1p(-symbol_error)
    log_n_factorial = math.lgamma(length + 1)
    terms = (
        math.exp(
            log_n_factorial
            - math.lgamma(i + 1)
            - math.lgamma(length - i + 1)
            + i * log_p
            + (length - i) * log_good
        )
        for i in range(limit + 1, length + 1)
    )
    return min(1.0, math.fsum(terms))


@dataclass(frozen=True)
class MeanFrameBudget:
    payload_bytes: int
    frame_count: int
    full_frame_count: int
    application_bytes_per_frame: int
    full_mac_bytes: int
    last_mac_bytes: int
    last_application_bytes: int
    mac_bytes: int
    ethernet_time_bytes: int


def mean_frame_budget(data_bytes, settings: InterMeanSettings):
    size = _uint(data_bytes, "data_bytes")
    framing = settings.framing
    capacity = framing.application_bytes_per_frame
    count = (size + capacity - 1) // capacity
    full_count = max(0, count - 1)
    last = size - full_count * capacity if count else 0
    header = framing.mac_header_bytes + 4 * framing.vlan_tags + framing.mac_fcs_bytes
    full_mac = max(framing.minimum_mac_bytes, header + framing.mtu_bytes)
    last_mac = (
        max(
            framing.minimum_mac_bytes,
            header + framing.upper_header_bytes + framing.business_header_bytes + last,
        )
        if count
        else 0
    )
    mac_bytes = full_count * full_mac + last_mac
    time_bytes = (
        mac_bytes
        + framing.preamble_sfd_bytes * count
        + framing.interframe_gap_bytes * max(0, count - 1)
    )
    _uint(time_bytes, "Ethernet equivalent time budget")
    return MeanFrameBudget(
        size,
        count,
        full_count,
        capacity,
        full_mac,
        last_mac,
        last,
        mac_bytes,
        time_bytes,
    )


def expected_extra_rounds(groups, tolerance=1e-12, max_terms=100000):
    """E[max(K_i)-1], with a union-bound remainder for grouped geometricals."""
    tolerance = _finite(tolerance, "summation tolerance")
    if (
        tolerance <= 0
        or isinstance(max_terms, bool)
        or not isinstance(max_terms, int)
        or max_terms < 1
    ):
        raise ValueError("Summation controls must be positive")
    active = []
    for count, probability in groups:
        count = _uint(count, "frame group count")
        p = _finite(probability, "frame failure probability")
        if p > 1:
            raise ValueError("Frame failure probability exceeds one")
        if count and p == 1:
            raise InterMeanIncompleteError("A frame cannot succeed under this channel")
        if count and p > 0:
            active.append((count, p))
    if not active:
        return 0.0, 0.0
    if sum(count for count, _ in active) == 1:
        p = active[0][1]
        return p / (1 - p), 0.0
    terms = []
    for r in range(1, max_terms + 1):
        log_all_finished = math.fsum(count * math.log1p(-(p**r)) for count, p in active)
        terms.append(-math.expm1(log_all_finished))
        remainder = math.fsum(count * p ** (r + 1) / (1 - p) for count, p in active)
        if remainder <= tolerance:
            return math.fsum(terms), remainder
    raise ArithmeticError("Recovery mean did not meet its declared error bound")


def _allocated(link, settings, value, direction):
    nominal = settings.mapping_efficiency * link.capacity_gbps
    service = (
        nominal if value is None else _finite(value, f"{direction} allocated rate")
    )
    if not math.isfinite(nominal) or nominal <= 0 or service <= 0 or service > nominal:
        raise ValueError(
            "Allocated service rate must be positive and <= mapped capacity"
        )
    return nominal, service


def predict_mean_hop(
    data_bytes: int,
    forward: InterLinkState,
    reverse: InterLinkState,
    settings: InterMeanSettings,
    *,
    allocated_forward_gbps=None,
    allocated_reverse_gbps=None,
):
    """Predict receiver delivery, using supplied ISLSim link states only.

    Optional allocated rates are runtime scheduler outputs AFTER mapping;
    they are not independent configurable link capacities. Scope eligibility
    (>10 Gbps) is checked on C_net, not on a competing flow's smaller allocation.
    """
    frame = mean_frame_budget(data_bytes, settings)
    fec = settings.fec
    result = {
        "model": "inter_surrogate_mean",
        "profile": settings.profile,
        "execution": "analytical_only",
        "completion_semantics": (
            "receiver_delivery"
            if settings.recovery_mode != "none"
            else "nominal_one_pass_without_reliability_guarantee"
        ),
        "bandwidth_source": "islsim",
        "frame_budget": asdict(frame),
        "c_net_gbps": forward.capacity_gbps,
        "mapping_efficiency": settings.mapping_efficiency,
        "fec_code_rate": fec.k / fec.n,
        "fec_raw_time_factor": fec.n / fec.k,
        "fec_applied_bandwidth_factor": 1.0,
        "phi_credit": 1.0,
        "recovery_policy": settings.recovery_mode,
        "sampled_errors": False,
        "scheduled_ack_or_retry_flows": False,
    }
    timing = dict.fromkeys(
        (
            "startup_ns",
            "serialization_ns",
            "propagation_ns",
            "processing_ns",
            "fec_pipeline_ns",
            "retry_serialization_ns",
            "retry_feedback_ns",
            "source_reload_ns",
            "retransmission_ns",
            "feedback_cycle_ns",
            "total_ns",
        ),
        0.0,
    )
    if not frame.frame_count:
        result.update(
            timing_ns=timing,
            expected_extra_rounds=0.0,
            expected_retransmitted_frames=0.0,
            numerical_error_bound_ns=0.0,
            completion_delay_ns=0,
            frame_error_groups=[],
            phi_share=1.0,
        )
        return result
    if not forward.feasible:
        raise InterLinkUnavailableError("The forward ISLSim link is unavailable")
    if forward.capacity_gbps <= 10:
        raise ValueError("This inter-satellite preset requires C_net >10 Gbps")
    nominal, rate = _allocated(forward, settings, allocated_forward_gbps, "forward")
    resource_bytes = frame.ethernet_time_bytes / settings.mapping_efficiency
    if not math.isfinite(resource_bytes) or resource_bytes >= MAX_U64:
        raise ValueError("Mapped equivalent resource work exceeds uint64 range")
    result.update(
        c0_gbps=nominal,
        allocated_service_gbps=rate,
        phi_share=nominal / rate,
        equivalent_resource_bytes=resource_bytes,
    )
    codeword_failure = None
    if fec.enabled and fec.ber_basis == "pre_fec_rs":
        codeword_failure = rs_codeword_failure_probability(
            fec.pre_fec_ber,
            fec.n,
            fec.correctable_symbols,
            fec.symbol_bits,
        )

    def error_probability(mac_bytes):
        if codeword_failure is not None:
            data_bits = fec.k * fec.symbol_bits
            words = (mac_bytes * 8 + data_bits - 1) // data_bits
            return _any_error(codeword_failure, words)
        ber = fec.post_fec_ber if fec.ber_basis == "post_fec" else fec.pre_fec_ber
        return _any_error(ber, mac_bytes * 8)

    groups = [
        {
            "count": count,
            "mac_bytes": size,
            "failure_probability": error_probability(size),
        }
        for count, size in (
            (frame.full_frame_count, frame.full_mac_bytes),
            (1, frame.last_mac_bytes),
        )
        if count
    ]
    timing.update(
        startup_ns=_finite(settings.startup_us * 1000, "startup"),
        serialization_ns=8 * frame.ethernet_time_bytes / rate,
        propagation_ns=forward.propagation_ns,
        processing_ns=_finite(
            (settings.hop_processing_us + settings.routing_processing_us) * 1000,
            "hop processing",
        ),
        fec_pipeline_ns=_finite(fec.encode_ns + fec.decode_ns, "FEC pipeline")
        if fec.enabled
        else 0.0,
    )
    extra_rounds = retries = error_bound = 0.0
    if settings.recovery_mode != "none" and any(
        g["failure_probability"] for g in groups
    ):
        if not reverse.feasible:
            raise InterLinkUnavailableError("Analytical recovery needs a reverse link")
        if reverse.capacity_gbps <= 10:
            raise ValueError("Analytical feedback requires reverse C_net >10 Gbps")
        _, reverse_rate = _allocated(
            reverse, settings, allocated_reverse_gbps, "reverse"
        )
        extra_rounds, bound = expected_extra_rounds(
            [(g["count"], g["failure_probability"]) for g in groups]
        )
        retries = math.fsum(
            g["count"] * g["failure_probability"] / (1 - g["failure_probability"])
            for g in groups
        )
        framing = settings.framing
        # A gap is paid only between frames of a nonempty retry batch.
        repeated_work = (
            math.fsum(
                g["count"]
                * (
                    g["mac_bytes"]
                    + framing.preamble_sfd_bytes
                    + framing.interframe_gap_bytes
                )
                * g["failure_probability"]
                / (1 - g["failure_probability"])
                for g in groups
            )
            - framing.interframe_gap_bytes * extra_rounds
        )
        if repeated_work < -framing.interframe_gap_bytes * bound - 1e-9:
            raise ArithmeticError("Negative analytical retransmission work")
        repeated_work = max(0.0, repeated_work)
        feedback = _finite(
            math.fsum(
                (
                    forward.propagation_ns,
                    reverse.propagation_ns,
                    2 * timing["processing_ns"],
                    2 * timing["fec_pipeline_ns"],
                    settings.feedback_extra_us * 1000,
                    8 * settings.feedback_time_equivalent_bytes / reverse_rate,
                )
            ),
            "feedback cycle",
        )
        reload = _finite(settings.source_reload_per_round_us * 1000, "reload per round")
        timing.update(
            retry_serialization_ns=8 * repeated_work / rate,
            retry_feedback_ns=extra_rounds * feedback,
            source_reload_ns=extra_rounds * reload,
            feedback_cycle_ns=feedback,
        )
        timing["retransmission_ns"] = math.fsum(
            (
                timing["retry_serialization_ns"],
                timing["retry_feedback_ns"],
            )
        )
        error_bound = bound * (
            feedback + reload + 8 * framing.interframe_gap_bytes / rate
        )
    timing["total_ns"] = math.fsum(
        timing[name]
        for name in (
            "startup_ns",
            "serialization_ns",
            "propagation_ns",
            "processing_ns",
            "fec_pipeline_ns",
            "retransmission_ns",
            "source_reload_ns",
        )
    )
    for name, value in timing.items():
        _finite(value, name)
    if timing["total_ns"] >= MAX_U64:
        raise ValueError("Mean latency exceeds the simulator clock range")
    result.update(
        timing_ns=timing,
        frame_error_groups=groups,
        fec_codeword_failure_probability=codeword_failure,
        expected_extra_rounds=extra_rounds,
        expected_retransmitted_frames=retries,
        numerical_error_bound_ns=error_bound,
        completion_delay_ns=math.ceil(timing["total_ns"]),
    )
    return result

