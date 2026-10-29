"""Check cell clearances before preparing any physical link budgets."""

import math
from dataclasses import replace
from itertools import combinations

from orbitserve.orbit import load_sgp4_element_bundle, propagate_sgp4_teme


def validate_formation_input(manifest, times):
    """Distances are rotation invariant, so a compute-only TEME pass suffices.

    This checks the same physical time grid as the full ITRS validation. It is
    a sampled clearance check, not a continuous-time station-keeping guarantee.
    """
    keepouts = {row['name']: row.get('keepout_m', 0.)
                for row in manifest.get('initialization_provenance', ())}
    cells = {}
    for node in manifest['network']['nodes']:
        if node.get('cell_id') and keepouts.get(node.get('shell_id'), 0.) > 0:
            cells.setdefault(node['cell_id'], []).append(node)
    cells = {key: rows for key, rows in cells.items() if len(rows) > 1}
    if not cells:
        return {}
    member_ids = {node['node_id'] for rows in cells.values() for node in rows}
    bundle = load_sgp4_element_bundle(manifest['elements'])
    bundle = replace(bundle, satellites=tuple(s for s in bundle.satellites if s.satellite_id in member_ids))
    state = propagate_sgp4_teme(bundle, simulation_epoch_utc=manifest['time']['epoch_utc'], sample_times_s=times)
    positions = {}
    for sample in state.samples:
        positions.setdefault(sample.time_s, {})[sample.satellite_id] = sample.position_teme_km
    results, errors = {}, []
    for cell, nodes in sorted(cells.items()):
        closest = (math.inf, None, None)
        pairs = tuple(combinations(sorted(n['node_id'] for n in nodes), 2))
        for time, frame in positions.items():
            for a, b in pairs:
                distance = math.dist(frame[a], frame[b]) * 1000
                if distance < closest[0]:
                    closest = distance, time, (a, b)
        distance, time, pair = closest
        required = keepouts[nodes[0]['shell_id']]
        results[cell] = dict(min_separation_m=distance, required_m=required,
                             time_s=time, member_pair=list(pair), sample_count=len(positions))
        if distance < required:
            errors.append(f'cell {cell}: {distance:.3f} < {required:.3f} m at t={time:g} s '
                          f'between {pair[0]} and {pair[1]}')
    if errors:
        raise ValueError('formation keepout preflight failed: ' + '; '.join(errors[:5]))
    return results
