"""Whole-model eligibility before any collective expansion."""
from dataclasses import replace
from .intra_hierarchical import estimate_hierarchical,HierarchyNotApplicable
from .policy import DEFAULT_ALGORITHMS

def select_whole(task,policy,selection,routes,network):
    if policy.intra_mode=='IntraSatelliteRoofline' and selection['scope']!='inter_satellite':
        from .native_intra import estimate_native
        return task,estimate_native(task,routes,network)
    try:return task,estimate_hierarchical(task,routes,policy.intra_hierarchy)
    except HierarchyNotApplicable as error:
        if selection['algorithm_source']!='comm_default':raise
        algorithm=DEFAULT_ALGORITHMS[task.operation]
        selection.update(algorithm=algorithm,algorithm_source='comm_default_topology_fallback',
                         whole_rejection=str(error))
        return replace(task,algorithm=algorithm),None
