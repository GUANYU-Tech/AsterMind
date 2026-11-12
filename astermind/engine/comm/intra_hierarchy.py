"""Physical su declarations and communication-scope classification."""
from pydantic import BaseModel, ConfigDict, model_validator

class IntraHierarchy(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    su_groups: tuple[tuple[str, ...], ...]
    su_switches: tuple[str, ...]

    @model_validator(mode='after')
    def validate_groups(self):
        if len(self.su_groups)<2 or len(self.su_groups)!=len(self.su_switches):
            raise ValueError('at least two su groups and one switch per group required')
        if min(map(len,self.su_groups))<2 or len(set(map(len,self.su_groups)))!=1:
            raise ValueError('equal su groups of at least two GPUs required')
        ids = sum(self.su_groups,()) + self.su_switches
        if len(set(ids)) != len(ids):
            raise ValueError('su GPU and switch IDs must be distinct')
        return self


def classify_participants(frame, participants):
    nodes = frame.node_states
    if not participants or any(p not in nodes for p in participants):
        raise ValueError('communication participants must exist in Network Frame')
    owners = {nodes[p].satellite_id for p in participants}
    if len(owners)>1:
        return 'inter_satellite', None, False, 'intra_topology'
    definitions = [n.attributes.get('intra_hierarchy') for n in nodes.values()
                   if n.satellite_id in owners and n.attributes.get('intra_hierarchy')]
    if not definitions:
        return 'single_su', None, False, 'satellite_only'
    if any(d != definitions[0] for d in definitions):
        raise ValueError('conflicting physical su declarations in Frame')
    hierarchy = IntraHierarchy.model_validate(definitions[0])
    members = set(participants)
    known = set().union(*map(set,hierarchy.su_groups))
    if members-known:
        raise ValueError('participants outside declared physical su groups')
    touched = [set(g) for g in hierarchy.su_groups if members.intersection(g)]
    return ('intra_cross_su' if len(touched)>1 else 'single_su', hierarchy,
            members == set().union(*touched), 'intra_topology')
