"""Restricted overrides for the existing Python intra surrogate factories."""
import hashlib
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Parameters(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    ring_bw_bytes_per_us: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    startup_us: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    phase_fixed_us: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    wire_efficiency: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)


class SUParameters(Parameters):
    tree_bw_per_level_bytes_per_us: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fullmesh_bw_bytes_per_us: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    credit_bdp_ring_packets: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    credit_bdp_tree_packets: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class SUDomain(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    overrides: SUParameters = Field(default_factory=SUParameters)


class SODomain(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    overrides: Parameters = Field(default_factory=Parameters)


class IntraProfile(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    name: str = Field(default='registered-reference', min_length=1)
    base: Literal['nvlink_rocev2_h200_reference'] = 'nvlink_rocev2_h200_reference'
    source: str = Field(default='ProtoBridge Python Surrogate', min_length=1)
    su: SUDomain = Field(default_factory=SUDomain)
    so: SODomain = Field(default_factory=SODomain)

    @model_validator(mode='after')
    def nonblank(self):
        if not self.name.strip() or not self.source.strip():
            raise ValueError('profile name and source must not be blank')
        return self

    @classmethod
    def load(cls, path):
        return cls.model_validate_json(Path(path).read_text(encoding='utf-8'))

    def summary(self, *, custom=False):
        content = self.model_dump(mode='json', exclude_none=True)
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return dict(name=self.name, source=self.source, base=self.base, sha256=digest,
                    upstream_commit='408d4703a86535dc6d2242a88fa64036b6ea92bd',
                    labels=['SU 域内设备互联', '同星 SU 域间互联'],
                    status='custom_unverified' if custom else 'reference_approximation',
                    overrides=dict(su=self.su.overrides.model_dump(exclude_none=True),
                                   so=self.so.overrides.model_dump(exclude_none=True)),
                    limitations=['参考 profile 的参数只适用于相应硬件、协议和拓扑。',
                                 'RoCEv2 域间参数属于规格推导模型，不代表实测硬件校准。',
                                 '自定义参数不改变模型适用拓扑，也不代表所选平台已经验证。'])
