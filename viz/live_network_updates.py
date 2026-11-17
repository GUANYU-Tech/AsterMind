"""Avoid rebuilding an unchanged native supply frame for each link notification.

The native host logs each link notification separately. A frame already reads
all physical links at the current exact tick, so notifications at that same
tick and topology revision expose the same supply. Traffic events, native
clocks, fault revisions and manual interventions are left in the host queue.
"""
from functools import wraps


def coalesce_network_updates(host):
    policy = host.communication_policy
    previous = policy.network_changed
    if getattr(previous, '_live_supply_gate', False):
        return
    last = None

    @wraps(previous)
    def network_changed():
        nonlocal last
        ticks = getattr(host.now, 'ticks', None)
        # Never merge distinct attosecond events through a rounded float key.
        if ticks is None:
            return previous()
        key = ticks, host.net.revision
        if key == last:
            return
        result = previous()
        last = key
        return result

    network_changed._live_supply_gate = True
    policy.network_changed = network_changed
