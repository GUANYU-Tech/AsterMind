"""Lossless JSON views of the original compute contracts, without timing edits."""
from collections.abc import Mapping
from dataclasses import fields,is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path


def contract(value):
    if isinstance(value,Enum):return value.value
    if is_dataclass(value):return {field.name:contract(getattr(value,field.name)) for field in fields(value)}
    if isinstance(value,Mapping):return {str(key):contract(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [contract(item) for item in value]
    if isinstance(value,Path):return str(value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'non-JSON compute provenance: {type(value).__name__}')


def inputs(root,cells):
    snapshot=root/'.orbitserve_snapshot.json'
    manifest=json.loads(snapshot.read_text())
    identities={}
    for cell in cells:
        files={}
        for field in ('model','hardware','memory_policy_file','host_launch_profile'):
            name=getattr(cell,field,None)
            if name is not None:
                path=root/name
                files[field]=dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        identities[cell.id]=dict(files=files,parallel=cell.parallel.model_dump(),
            placements=[placement.model_dump() for placement in cell.placements],
            deployment=cell.deployment,execution_mode=cell.execution_mode,
            scheduler_adapter=cell.scheduler_adapter,expert_routing_seed=cell.expert_routing_seed)
        if cell.fast_glm_dsa_score is not None:
            identities[cell.id]["fast_glm_dsa_score"]=cell.fast_glm_dsa_score
    return dict(snapshot_manifest_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        native_source_commit=manifest['source_commit'],native_source_config_tree_sha256=manifest['source_config_tree_sha256'],
        cells=identities,assumptions=[
            'Length-only requests use the upstream synthetic token IDs and explicit MoE routing seed.',
            'DP input distribution assumes each replica leader already has its input; only intra-replica broadcast is charged.',
            'Intra communication predictions do not compete globally or reprice mid-flight.',
            'Cross-satellite communication requires all selected path resources simultaneously; no store-and-forward.',
        ])
