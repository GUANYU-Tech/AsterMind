"""Communication configuration; task resolution and execution stay in Engine."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from astermind.engine.comm._vendor.inter_mean.settings import InterMeanSettings
from .intra_topology import IntraHierarchy
from .intra_profile import IntraProfile

Scope = Literal['single_su', 'intra_cross_su', 'inter_satellite']
LAYERED_COLLECTIVES = frozenset(('all_reduce', 'all_gather', 'all_gatherv',
                                 'reduce_scatter', 'reduce_scatterv'))
LAYER_ALGORITHMS = frozenset(('ring', 'tree', 'fullmesh', 'fc2', 'mesh', 'direct', 'pairwise'))
ALGORITHMS = {
    'all_reduce': frozenset(('ring', 'tree', 'fullmesh', 'hierarchical')),
    'all_gather': frozenset(('ring', 'fullmesh', 'hierarchical')),
    'all_to_all': frozenset(('pairwise', 'hierarchical')),
    'all_gatherv': frozenset(('direct', 'pairwise')),
    'reduce_scatterv': frozenset(('direct', 'pairwise')),
    'broadcast': frozenset(('direct', 'pairwise')),
    'send_recv': frozenset(('direct', 'send_recv')),
}
DEFAULT_ALGORITHMS = {
    'all_reduce': 'ring', 'all_gather': 'ring', 'all_to_all': 'pairwise',
    'all_gatherv': 'pairwise', 'reduce_scatter': 'pairwise', 'reduce_scatterv': 'pairwise',
    'broadcast': 'direct', 'send_recv': 'send_recv',
}


class CellCommPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    algorithms: dict[Scope, dict[str, str]] = Field(default_factory=dict)
    intra_mode: Literal['surrogate', 'IntraSatelliteRoofline'] | None = None
    hardware_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode='after')
    def validate_algorithms(self):
        for scope, operations in self.algorithms.items():
            for operation, algorithm in operations.items():
                available = ALGORITHMS.get(operation, frozenset())
                if scope != 'inter_satellite' and operation in LAYERED_COLLECTIVES:
                    available |= LAYER_ALGORITHMS | {'hierarchical'}
                if algorithm not in available:
                    raise ValueError(f'unsupported Comm policy operation/algorithm: {operation}/{algorithm}')
                if algorithm == 'hierarchical' and scope != 'intra_cross_su':
                    raise ValueError('hierarchical is currently configurable only for intra_cross_su')
        return self


class CommPolicy(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)
    ep_strategy: Literal["allgather", "direct"]="allgather"
    mode: Literal["surrogate"]="surrogate"
    execution: Literal["runtime"]="runtime"
    routing_policy: Literal["min_hops", "min_propagation", "earliest_arrival"]="min_propagation"
    ground_processing_ns: float=Field(default=0,ge=0,allow_inf_nan=False)
    intra: Literal["protobridge_h200"]="protobridge_h200"
    intra_profile_file: str | None = None
    intra_profile: IntraProfile | None = None

    @property
    def resolved_intra_profile(self):
        return self.intra_profile or IntraProfile()

    @model_validator(mode='after')
    def load_intra_profile(self):
        if self.intra_profile_file:
            if self.intra_profile is not None:
                raise ValueError('supply intra_profile_file or intra_profile, not both')
            object.__setattr__(self, 'intra_profile', IntraProfile.load(self.intra_profile_file))
            # Serialize a self-contained configuration, not a path requiring a second read.
            object.__setattr__(self, 'intra_profile_file', None)
        return self
    _intra_hierarchy: IntraHierarchy | None = PrivateAttr(default=None)

    @property
    def intra_hierarchy(self):
        """Resolved physical metadata; never a user policy field."""
        return self._intra_hierarchy

    intra_mode: Literal["surrogate", "IntraSatelliteRoofline"]="surrogate"
    cell_policies: dict[str, CellCommPolicy]=Field(default_factory=dict)
    default_hardware_id: str = Field(default='nvidia_h200', min_length=1)
    gpu_hardware: dict[str, str] = Field(default_factory=dict)
    gpu_port_gbps: dict[str, Annotated[float, Field(gt=0, allow_inf_nan=False)]] = Field(default_factory=dict)
    inter: Literal["inter_surrogate"]="inter_surrogate"
    share_isl_within_round: bool=Field(default=False, strict=True)
    inter_parameters: InterMeanSettings=Field(default_factory=lambda:InterMeanSettings(recovery_mode='none'))
    runtime_capacity_basis: Literal["raw_line", "net_after_existing_fec", "application_effective"]="net_after_existing_fec"
    runtime_protocol: Literal["inter_reference", "payload_only"]="inter_reference"
    retry_limit: int=Field(default=8,ge=0)
    task_timeout_s: float | None=Field(default=None,gt=0,allow_inf_nan=False)
    trace_service: bool=True
    resource_profiles: dict[str, dict]=Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def set_fluid_defaults(cls, values):
        if isinstance(values,dict) and values.get('inter') in ('analytical_mean','frame_arq'):
            raise ValueError('retired inter backend; use inter_surrogate (packet backend is not integrated)')
        if isinstance(values,dict) and isinstance(values.get('inter_parameters'),dict):
            values={**values,'inter_parameters':{'recovery_mode':'none',**values['inter_parameters']}}
        return values

    @model_validator(mode='after')
    def validate_frame_backend(self):
        if self.share_isl_within_round:
            raise ValueError('share_isl_within_round is retired; use runtime for shared resource competition')
        if any(not name.strip() for name in self.cell_policies):
            raise ValueError('cell_policies keys must be nonempty Cell IDs')
        ids = [self.default_hardware_id, *self.gpu_hardware.keys(), *self.gpu_hardware.values(), *self.gpu_port_gbps]
        ids.extend(cell.hardware_id for cell in self.cell_policies.values() if cell.hardware_id is not None)
        if any(not identity.strip() or identity.strip() != identity for identity in ids):
            raise ValueError('Comm GPU IDs and hardware IDs must be nonempty and trimmed')
        if self.inter=='inter_surrogate':
            p=self.inter_parameters
            if p.recovery_mode!='none' or any((p.feedback_extra_us,p.feedback_time_equivalent_bytes,p.source_reload_per_round_us)):
                raise ValueError('whole-path inter surrogate supports no ARQ/feedback/reload; use recovery_mode=none and zero feedback/reload')
        return self


# Preserve public serialization identities across the file-only move.
CellCommPolicy.__module__ = 'astermind.engine.comm.policy'

# Preserve public serialization identities across the file-only move.
CommPolicy.__module__ = 'astermind.engine.comm.frame_surrogate'
