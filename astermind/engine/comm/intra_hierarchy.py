"""Physical su declarations and communication-scope classification."""
# Compatibility import: physical declarations belong to Builder, not a model.
from astermind.builder.config.intra_topology import IntraHierarchy


class ScaleUpPairIndex:
    """Deployed GPU membership for Fast's same-satellite P2P transfers."""

    def __init__(self, topology):
        self.hierarchies = {owner: IntraHierarchy.model_validate(value)
                            for owner, value in topology.intra_hierarchies.items()}
        self.membership = {gpu: group_id for hierarchy in self.hierarchies.values()
                           for group_id, group in enumerate(hierarchy.su_groups)
                           for gpu in group}
        self.owners = {device_id: device.satellite for device_id, device in topology.devices.items()
                       if device.kind == 'gpu'}

    def classify(self, source, target):
        owner = self.owners[source]
        if self.owners[target] != owner:
            return 'inter_satellite', None
        hierarchy = self.hierarchies.get(owner)
        if hierarchy is None:
            return 'single_su', None
        if source not in self.membership or target not in self.membership:
            raise ValueError('participants outside declared physical su groups')
        scope = 'single_su' if self.membership[source] == self.membership[target] else 'intra_cross_su'
        return scope, hierarchy


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
