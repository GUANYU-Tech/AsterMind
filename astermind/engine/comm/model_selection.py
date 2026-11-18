"""Check registered models before retaining the configured algorithm for P2P."""
from .intra_hierarchical import estimate_hierarchical, HierarchyNotApplicable
from .registered_profiles import ProfileMiss, whole_registered


def select_whole(task, policy, selection, routes, network):
    if policy.intra_mode == 'IntraSatelliteRoofline' and selection['scope'] != 'inter_satellite':
        if selection['scope'] == 'intra_cross_su':
            raise ValueError('ArchPrism native whole prediction across local SU domains requires a bound scale-out fabric')
        from .native_intra import estimate_native
        return task, estimate_native(task, routes, network)

    try:
        if task.algorithm == 'hierarchical':
            return task, estimate_hierarchical(task, routes, policy.intra_hierarchy,
                                               policy=policy, selection=selection)
        if task.operation != 'send_recv':
            return task, whole_registered(task, routes, policy, selection)
    except (HierarchyNotApplicable, ProfileMiss) as error:
        selection['whole_rejection'] = str(error)
    return task, None
