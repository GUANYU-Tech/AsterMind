"""Compute interface to the surrogate using only the public Network State Frame."""
from types import SimpleNamespace
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ._vendor.inter_mean.settings import InterMeanSettings
from .intra_hierarchical import IntraHierarchy
from .policy import CellCommPolicy


class CommPolicy(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)
    mode: Literal["surrogate"]="surrogate"
    execution: Literal["runtime"]="runtime"
    ground_processing_ns: float=Field(default=0,ge=0,allow_inf_nan=False)
    intra: Literal["protobridge_h200"]="protobridge_h200"
    _intra_hierarchy: IntraHierarchy | None = PrivateAttr(default=None)

    @property
    def intra_hierarchy(self):
        """Resolved physical metadata; never a user policy field."""
        return self._intra_hierarchy

    intra_mode: Literal["surrogate", "IntraSatelliteRoofline"]="surrogate"
    cell_policies: dict[str, CellCommPolicy]=Field(default_factory=dict)
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
        if self.inter=='inter_surrogate':
            p=self.inter_parameters
            if p.recovery_mode!='none' or any((p.feedback_extra_us,p.feedback_time_equivalent_bytes,p.source_reload_per_round_us)):
                raise ValueError('whole-path inter surrogate supports no ARQ/feedback/reload; use recovery_mode=none and zero feedback/reload')
        return self


class NoRouteError(ValueError):
    pass


class CommunicationRouting:
    """Route and candidate queries delegated to the retained target network."""
    def route(self, source, target, size_bytes=0, *, first_hop=None):
        routes = self._refresh()
        result = routes.route(source, target, size_bytes, first_hop=first_hop)
        return {**result, 'frame_revision': routes.frame.revision,
                'frame_time_s': routes.frame.start_time_s}

    def candidate_path(self, source, target):
        return self.network.candidate_path(source, target)


class FrameRoutes:

    def __init__(self,frame):
        self.frame=frame
        self.routing_policy="earliest_arrival"
        self.ground_owners={v.satellite_id for v in frame.node_states.values() if v.role=="ground"}
        self.topology=SimpleNamespace(devices={n:SimpleNamespace(kind=v.attributes["device_kind"],
            satellite=v.satellite_id) for n,v in frame.node_states.items()})
        self.adj={n:[] for n in frame.node_states}
        self.edges={}
        for link in frame.established_topology_graph:
            a,b=link.source_node_id,link.target_node_id
            if any(k in link.attributes for k in ("reverse_bandwidth_bps","reverse_bandwidth_gbps","reverse_latency_ns","reverse_propagation_delay_s")):
                raise ValueError(f"{link.link_id}: asymmetric parameters require a directional Frame contract")
            pairs=[(a,b),(b,a)] if link.attributes.get("bidirectional",True) else [(a,b)]
            for src,dst in pairs:
                key=(link.link_id,src,dst)
                self.edges[key]=dict(index=len(self.edges),source=src,target=dst,
                    source_owner=frame.node_states[src].satellite_id,target_owner=frame.node_states[dst].satellite_id,
                    bandwidth_gbps=link.bandwidth_bps/1e9,latency_ns=link.propagation_delay_s*1e9)
                self.edges[key]["resource_key"]=key
                directional=link.attributes.get('forward_shared_capacity_pools' if src==a else 'reverse_shared_capacity_pools',())
                self.edges[key]["shared_pools"]=tuple(dict.fromkeys((*getattr(link,"shared_capacity_pools",()),*directional)))
                self.edges[key]["attributes"]=dict(link.attributes)
                direction=link.attributes.get("duplex", "full_duplex")
                if direction not in ("full_duplex", "shared"):
                    raise ValueError(f"unknown duplex mode: {direction}")
                if direction=="shared":self.edges[key]["resource_key"]=(link.link_id,"shared")
                self.adj[src].append(key)

    def _result(self,path,keys,size):
        bandwidth=min((self.edges[k]["bandwidth_gbps"] for k in keys),default=float("inf"))
        # With intra contention disabled this is a per-transfer ceiling, not
        # a claim that concurrent transfers share the switch aggregate.
        if len({self.topology.devices[n].satellite for n in path}) == 1:
            switch_caps = [self.frame.node_states[n].attributes.get('capacity_gbps') for n in path
                           if self.topology.devices[n].kind == 'switch']
            bandwidth = min([bandwidth] + [c for c in switch_caps if c is not None])
        latency=sum(self.edges[k]["latency_ns"] for k in keys)
        return dict(path=list(path),resources=keys,latency_ns=latency,bandwidth_gbps=bandwidth,
                    isolated_estimate_ns=latency+(0. if bandwidth==float("inf") else 8*size/bandwidth),
                    routing_policy=self.routing_policy)





    def route_resources(self,route):
        return [dict(self.edges[key],reverse=self.edges.get((key[0],key[2],key[1])))
                for key in route["resources"]]
