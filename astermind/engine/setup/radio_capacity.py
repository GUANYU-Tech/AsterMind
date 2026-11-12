"""Versioned public satellite capacities and directional RF sharing resources.

These are satellite aggregates, never beam/link rates or constellation geometry.
Unknown values remain unknown; callers must label scenario assumptions explicitly.
All values use decimal SI bits/s. Profiles are frozen source editions, not aliases
for whatever the vendor publishes most recently.
"""
from copy import deepcopy
from astermind.engine.setup.starlink_defaults import simulation_defaults


STARLINK_RADIO_PROFILES = {
    'starlink_v2_mini_2024': {
        'generation': 'Starlink V2 Mini',
        'capacities_bps': {'user_downlink_bps': 96e9},
        'terminals': {'optical_count': 3, 'optical_advertised_bps': 200e9,
                      'rf_feeder_antennas': 3, 'user_phased_arrays': 5,
                      'user_downlink_beams': 192, 'user_uplink_beams': 144},
        'terminal_sources': ['https://starlink.com/technology',
            'https://starlink.com/en-ps/updates/starlink-version-3-satellites'],
        'sources': [
            'https://starlink.com/public-files/starlinkProgressReport_2024.pdf#page=26',
            'https://content.spacex.com/cms-assets/FINAL_Documents%20and%20Updates/SpaceX%20-%20EU%20Prospectus%20%28Approved%20by%20Bafin%29%20-%20June%205%2C%202026.pdf',
        ],
        'notes': ['96 Gbps is per-satellite user downlink, not per GSL.',
                  'Uplink and RF feeder limits are unspecified in this edition.',
                  'No optical duplex interpretation or orbital shell is inferred.'],
    },
    'starlink_v3_design_160_20261021': {
        'generation': 'Starlink V3 design',
        'capacities_bps': {'user_downlink_bps': 1e12, 'user_uplink_bps': 160e9,
                           'feeder_combined_bps': 1.2e12},
        'terminals': {'optical_count': 6, 'optical_advertised_bps': 400e9,
                      'rf_feeder_antennas': 4, 'user_phased_arrays': None,
                      'user_downlink_beams': 2048, 'user_uplink_beams': 2048},
        'terminal_sources': ['https://starlink.com/en-ps/updates/starlink-version-3-satellites'],
        'sources': ['https://starlink.com/en-ps/updates/starlink-version-3-satellites'],
        'notes': ['Source edition retrieved 2026-10-21: 1 Tbps DL / 160 Gbps UL.',
                  '1.2 Tbps RF feeder is conservatively shared across both directions; '
                  'the source does not establish separate directional feeder budgets.',
                  'Other official editions report >200 Gbps UL or 1024 Gbps DL; '
                  'these are not merged into this profile.',
                  'Six advertised 400-gigabit lasers do not establish per-direction '
                  'optical rates; optical link budgets remain explicit scenario inputs.',
                  'Design target, not measured operational throughput or a 72x22 shell.'],
    },
}

# Keep conflicting source editions separate and immutable by identifier.
STARLINK_RADIO_PROFILES['starlink_v3_progress_2025'] = {
    **deepcopy(STARLINK_RADIO_PROFILES['starlink_v3_design_160_20261021']),
    'generation': 'Starlink V3 · 2025 report lower-bound scenario',
    'capacities_bps': {'feeder_combined_bps': 1.2e12},
    'sources': ['https://starlink.com/public-files/starlinkProgressReport_2025.pdf#page=72'],
    'notes': ['Report says >1000 Gbps DL / >200 Gbps UL; exact ceilings unknown.',
              'Runnable lower-bound assumptions are explicit overrides, not exact vendor specifications.',
              'Optical terminals / RF backhaul references come from the separate V3 hardware specification.'],
}
for _identity, _profile in STARLINK_RADIO_PROFILES.items():
    _profile['simulation_defaults'] = simulation_defaults(2 if 'v2_' in _identity else 3,
                                                         report_2025=_identity == 'starlink_v3_progress_2025')
    if _identity == 'starlink_v3_progress_2025':
        _profile['simulation_defaults']['radio_capacity_bps'].update(user_downlink_bps=1e12, user_uplink_bps=200e9)

CAPACITY_FIELDS = {'user_uplink_bps', 'user_downlink_bps', 'feeder_uplink_bps',
                   'feeder_downlink_bps', 'feeder_combined_bps'}


def radio_parameters(node):
    result = deepcopy({k: v for k, v in STARLINK_RADIO_PROFILES[node.radio_profile].items()
                       if k != 'simulation_defaults'}) if node.radio_profile else {
        'generation': 'scenario', 'capacities_bps': {}, 'sources': [], 'notes': []}
    result['profile_id'] = node.radio_profile
    result['capacities_bps'].update(node.radio_capacity_bps)
    result['scenario_overrides_bps'] = dict(node.radio_capacity_bps)
    result['assumptions'] = list(node.capacity_assumptions)
    result['allocated_terminal_limits'] = {'isl': node.isl_terminals, 'sgl': node.sgl_terminals,
        'optical_bandwidth_bps_per_direction': node.optical_terminal_bandwidth_bps}
    result['terminal_assumptions'] = list(node.terminal_assumptions)
    result['terminal_reference_checked'] = '2026-10-21' if node.radio_profile else None
    return result


def capacity_pools(nodes):
    return {f'radio:{n.node_id}:{name}': rate for n in nodes.values()
            for name, rate in _capacity_values(n).items()}


def _capacity_values(node):
    # Hot routing/budget paths need numeric limits, not copied source documents.
    values = dict(STARLINK_RADIO_PROFILES[node.radio_profile]['capacities_bps']) if node.radio_profile else {}
    values.update(node.radio_capacity_bps)
    return values


def hop_pools(nodes, source, target):
    a, b = nodes[source], nodes[target]
    if (a.role == 'ground') == (b.role == 'ground'):
        return ()
    ground, satellite = (a, b) if a.role == 'ground' else (b, a)
    params = _capacity_values(satellite)
    if not params and not satellite.radio_profile:
        return ()  # Existing unversioned engineering scenarios retain their inputs.
    service = 'user' if ground.ground_kind == 'user' else 'feeder'
    direction = 'uplink' if a.role == 'ground' else 'downlink'
    field = f'{service}_{direction}_bps'
    names = [field] if field in params else []
    if service == 'feeder' and 'feeder_combined_bps' in params:
        names.append('feeder_combined_bps')
    if not names:
        raise ValueError(f'{satellite.node_id}: {field} is unpublished in '
                         f'{satellite.radio_profile}; provide radio_capacity_bps '
                         'and capacity_assumptions for this scenario')
    return tuple(f'radio:{satellite.node_id}:{name}' for name in names)


def directional_profile(nodes, profiles, edge, source):
    profile = profiles[edge.get('profile_id', edge['kind'])]
    if edge['kind'] == 'sgl':
        ground = nodes[edge['endpoint_a']] if nodes[edge['endpoint_a']].role == 'ground' else nodes[edge['endpoint_b']]
        satellite = nodes[edge['endpoint_b']] if nodes[edge['endpoint_a']].role == 'ground' else nodes[edge['endpoint_a']]
        service = 'user' if ground.ground_kind == 'user' else 'feeder'
        direction = 'uplink' if nodes[source].role == 'ground' else 'downlink'
        selected = satellite.sgl_profiles.get(f'{service}_{direction}')
        if selected:
            return profiles[selected]
        name = profile.uplink_profile if nodes[source].role == 'ground' else profile.downlink_profile
        if name:
            return profiles[name]
    return profile


def max_min_rates(requirements, capacities):
    """Progressive filling over physical links and satellite aggregate pools."""
    rates = dict.fromkeys(requirements, 0.0)
    pending = set(requirements)
    residual = dict(capacities)
    while pending:
        counts = {}
        for identity in pending:
            for key in requirements[identity]:
                counts[key] = counts.get(key, 0) + 1
        step = min(residual.get(k, 0) / n for k, n in counts.items())
        for identity in pending:
            rates[identity] += max(0, step)
        for key, count in counts.items():
            residual[key] = max(0, residual.get(key, 0) - step * count)
        saturated = {k for k in counts if residual[k] <= max(1e-9, capacities.get(k, 0) * 1e-12)}
        pending = {i for i in pending if not saturated.intersection(requirements[i])}
    return rates
