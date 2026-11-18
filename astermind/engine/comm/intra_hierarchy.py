"""Physical su declarations and communication-scope classification."""
# Compatibility import: physical declarations belong to Builder, not a model.
from astermind.builder.config.intra_topology import IntraHierarchy


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
