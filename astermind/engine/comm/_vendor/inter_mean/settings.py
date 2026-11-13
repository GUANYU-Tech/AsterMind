"""Defaults for the ISLSim-fed analytical inter-satellite reference model.

No capacity is configured here: ISLSim supplies C_net after its existing FEC
deduction and before the mapping deduction introduced by this model.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from .link import ReferenceSettings

NonnegativeTime = Annotated[float, Field(ge=0)]


class InterMeanFraming(ReferenceSettings):
    """Ethernet client-time accounting; not an additional GFP byte budget."""

    mtu_bytes: Annotated[StrictInt, Field(ge=68, le=9000)] = 1500
    upper_header_bytes: Annotated[StrictInt, Field(ge=0)] = 28
    business_header_bytes: Annotated[StrictInt, Field(ge=0)] = 0
    vlan_tags: Annotated[StrictInt, Field(ge=0, le=2)] = 0
    mac_header_bytes: Annotated[StrictInt, Field(ge=1)] = 14
    mac_fcs_bytes: Annotated[StrictInt, Field(ge=0)] = 4
    minimum_mac_bytes: Annotated[StrictInt, Field(ge=1)] = 64
    preamble_sfd_bytes: Annotated[StrictInt, Field(ge=0)] = 8
    interframe_gap_bytes: Annotated[StrictInt, Field(ge=0)] = 12

    @property
    def application_bytes_per_frame(self):
        return self.mtu_bytes - self.upper_header_bytes - self.business_header_bytes

    @model_validator(mode="after")
    def validate_layout(self):
        if self.application_bytes_per_frame <= 0:
            raise ValueError("MTU must exceed configured upper/business headers")
        if self.minimum_mac_bytes < self.mac_header_bytes + self.mac_fcs_bytes:
            raise ValueError("Minimum MAC length must contain its header and FCS")
        return self


class InterMeanFec(ReferenceSettings):
    enabled: bool = True
    ber_basis: Literal["pre_fec_rs", "post_fec"] = "pre_fec_rs"
    pre_fec_ber: Annotated[float, Field(ge=0, le=1)] = 1e-4
    post_fec_ber: Annotated[float, Field(ge=0, le=1)] = 1e-9
    n: Annotated[StrictInt, Field(ge=1, le=65535)] = 528
    k: Annotated[StrictInt, Field(ge=1, le=65535)] = 514
    correctable_symbols: Annotated[StrictInt, Field(ge=0)] = 7
    symbol_bits: Annotated[StrictInt, Field(ge=1, le=16)] = 10
    encode_ns: NonnegativeTime = 0.0
    decode_ns: NonnegativeTime = 0.0

    @model_validator(mode="after")
    def validate_code(self):
        if self.k > self.n or 2 * self.correctable_symbols > self.n - self.k:
            raise ValueError("Invalid RS dimensions/correction capability")
        if self.n > (1 << self.symbol_bits) - 1:
            raise ValueError("RS length exceeds the selected symbol field")
        return self


class InterMeanSettings(ReferenceSettings):
    """Parameters for the opt-in analytical inter-satellite backend."""

    profile: Literal["otn_ethernet_time_reference"] = "otn_ethernet_time_reference"
    bandwidth_source: Literal["islsim"] = "islsim"
    bandwidth_basis: Literal["net_after_existing_fec"] = "net_after_existing_fec"
    # Supplied reference scenario coefficient, not a universal OTN efficiency.
    mapping_efficiency: Annotated[float, Field(gt=0, le=1)] = 0.99381412
    startup_us: NonnegativeTime = 0.0
    hop_processing_us: NonnegativeTime = 2000.0
    routing_processing_us: NonnegativeTime = 0.0
    recovery_mode: Literal["selective_repeat_round_mean", "none"] = (
        "selective_repeat_round_mean"
    )
    feedback_extra_us: NonnegativeTime = 0.0
    feedback_time_equivalent_bytes: Annotated[StrictInt, Field(ge=0)] = 0
    source_reload_per_round_us: NonnegativeTime = 0.0
    framing: InterMeanFraming = Field(default_factory=InterMeanFraming)
    fec: InterMeanFec = Field(default_factory=InterMeanFec)


DEFAULT_PROVENANCE = {
    "bandwidth": {
        "source": "ISLSim total_bw_gbps, supplied at runtime",
        "boundary": "after existing FEC deduction; before new mapping cost",
        "default_capacity": None,
        "additional_capacity_cap": None,
    },
    "mapping_efficiency": {
        "source": "user-supplied Ethernet/FlexO reference expression",
        "status": "scenario coefficient; not asserted universal or device-calibrated",
    },
    "hop_processing_us": {
        "source": "supplemental laser terminal table: combined processing <=2 ms",
        "status": "per-hop scenario mapping of an upper-limit indicator",
    },
    "fec": {
        "source": "user-selected RS(528,514) parameter table",
        "symbol_bits_and_t": "IEEE Clause 91 reference: 10-bit symbols, t=7",
        "ber_values": "indicator upper limits, not measured mean BERs",
        "post_fec_ber_role": "reference unless ber_basis='post_fec'",
        "latency_zero": "independent FEC pipeline delay ignored by assumption",
        "upstream_efficiency_note": "legacy ISLSim 0.85 is not RS rate 514/528",
    },
    "framing": {
        "source": "supplied Ethernet-time expression and platform framing defaults",
        "upper_headers": "28B IPv4/UDP reference, configurable; not terminal data",
        "boundary": "MAC + preamble/SFD + interframe gaps, no extra GFP header",
    },
    "recovery": {
        "source": "project analytical selective-repeat batch-round model",
        "status": "explicit reference policy, not a calibrated laser ARQ protocol",
        "assumptions": [
            "independent frame outcomes across attempts",
            "error-free feedback once after each unsuccessful batch round",
            "constant allocated rates during the evaluated transfer",
            "sufficient retained-frame storage",
        ],
    },
    "zero_defaults": {
        "startup": "established link, no acquisition/first-connect time",
        "routing": "additional routing CPU delay not separately represented",
        "feedback": "ideal feedback bytes/extra processing; propagation retained",
        "source_reload": "no additional source-memory reload cost",
    },
}

