"""Materialize multi-shell and physical formation inputs using public Orbit APIs."""

from copy import deepcopy
from pathlib import Path
import math

from astermind.builder.orbit import walker_delta_elements, tle_elements, dawn_dusk_elements


def _wrap_degrees(value):
    # Tiny negative offsets can round to exactly 360 after modulo.
    result = value % 360
    return 0. if result == 360 else result


def _hardware_profiles(defaults, overrides):
    """Layer scenario fields over the selected generation's hardware inputs."""
    result = deepcopy(defaults)
    for kind, profile in overrides.items():
        inherited = result.get(kind, {})
        merged = {**inherited, **deepcopy(profile)}
        if profile.get('islsim_config'):
            native = deepcopy(inherited.get('islsim_config', {}))
            for section, fields in profile['islsim_config'].items():
                native[section] = {**native.get(section, {}), **fields}
            merged['islsim_config'] = native
        result[kind] = merged
    return result


def snapshot_network_input(manifest, base_dir=".", *, bind_islsim=False):
    """Resolve source files without running constellation or orbit calculations."""
    result = deepcopy(manifest)
    if isinstance(result.get('elements'), str):
        import json
        result['elements'] = json.loads((Path(base_dir) / result['elements']).read_text(encoding='utf-8'))
    for collection in ('communication_shells', 'compute_formations'):
        for shell in result.get('constellations', {}).get(collection, ()):
            source = shell['source']
            if source.get('type') == 'tle' and 'tle_file' in source:
                source['tle_text'] = (Path(base_dir) / source.pop('tle_file')).read_text(encoding='utf-8')
    from astermind.engine import freeze_network_profiles
    if 'network' in result:
        result['network'] = freeze_network_profiles(result['network'], bind_islsim=bind_islsim)
    # Include generated hardware parameters in the input identity. Only local
    # physical preparation requests bind_islsim and attaches implementation hashes.
    from astermind.engine import STARLINK_RADIO_PROFILES
    for shell in result.get('constellations', {}).get('communication_shells', ()):
        defaults = STARLINK_RADIO_PROFILES.get(shell.get('radio_profile'), {}).get('simulation_defaults', {})
        profiles = _hardware_profiles(defaults.get('hardware_link_profiles', {}),
                                      shell.get('hardware_link_profiles', {}))
        if profiles:
            shell['hardware_link_profiles'] = freeze_network_profiles(
                {'link_profiles': profiles}, bind_islsim=bind_islsim)['link_profiles']
    return result


def materialize_network_input(manifest, base_dir="."):
    result = snapshot_network_input(manifest, base_dir)
    result.setdefault("time", {}).setdefault("step_s", 1.0)
    result["time"].setdefault("duration_s", 600.0)
    if "constellations" not in result:
        requested = result.pop('scenario_type', None)
        if requested is not None:
            roles = {n['role'] for n in result.get('network', {}).get('nodes', ())} - {'ground'}
            actual = {'compute_only': {'compute'}, 'communication_only': {'communication'},
                      'mixed': {'compute','communication'}}
            if requested not in actual or actual[requested] != roles:
                raise ValueError('scenario_type must match explicit network satellite roles')
        return result
    if "elements" in result:
        raise ValueError("choose constellations or explicit elements, not both")
    config = result.pop("constellations")
    network = result["network"]
    has_comm = bool(config.get('communication_shells'))
    has_compute = bool(config.get('compute_formations'))
    actual_scenario = 'mixed' if has_comm and has_compute else 'compute_only' if has_compute else 'communication_only'
    requested_scenario = result.pop('scenario_type', actual_scenario)
    if not (has_comm or has_compute) or requested_scenario != actual_scenario:
        raise ValueError('scenario_type must match nonempty communication_shells/compute_formations')
    if actual_scenario == 'compute_only':
        network['ground_roles'] = ['compute']
    if network.get("nodes"):
        raise ValueError(
            "constellation inputs generate physical nodes; do not also provide nodes"
        )
    nodes = []
    elements = []
    shells = []
    cell_topologies = {}
    cell_egress_policies = dict(network.get('cell_egress_policies', {}))
    sources = []
    epoch = result["time"]["epoch_utc"]

    def source_rows(source, name, compute=False, topology='grid'):
        kind = source["type"]
        if kind in {"walker_delta", "dawn_dusk"}:
            params = {
                k: source[k] for k in ("altitude_km", "planes")
            }
            params["satellites_per_plane"] = source[
                "clusters_per_plane" if compute else "satellites_per_plane"
            ]
            for k in (("walker_phase", "raan0_deg", "u0_deg", "inclination_deg")
                      if kind == 'walker_delta' else ('ltan_hours', 'plane_spread_deg', 'u0_deg')):
                if k in source:
                    params[k] = source[k]
            initializer = walker_delta_elements if kind == 'walker_delta' else dawn_dusk_elements
            bundle = initializer(
                name=name, epoch_utc=source.get("epoch_utc", epoch), **params
            )
            rows = [row.to_dict() for row in bundle.satellites]
            slots = params["satellites_per_plane"]
            layout = [(p, s) for p in range(params["planes"]) for s in range(slots)]
        elif kind == "tle":
            if compute:
                raise ValueError(
                    "compute TLE formations require explicit per-member element input"
                )
            if "tle_text" in source:
                text = source["tle_text"]
            else:
                text = (Path(base_dir) / source["tle_file"]).read_text(encoding="utf-8")
            bundle = tle_elements(text, name=name)
            if topology == 'explicit':
                rows = [row.to_dict() for row in bundle.satellites]
                layout = [(None, None)] * len(rows)
                indexing = 'explicit_catalog_pairs'
            else:
                from astermind.builder.physical.tle_layout import sparse_tle_layout
                rows, layout, indexing = sparse_tle_layout(bundle, source, epoch)
        else:
            raise ValueError("source.type must be walker_delta, dawn_dusk, or tle")
        sources.append(
            dict(
                name=name,
                type=kind,
                observationally_fitted=kind == "tle",
                indexing="plane_slot"
                if kind in {'walker_delta', 'dawn_dusk'}
                else indexing,
            )
        )
        return rows, layout

    for shell in config.get("communication_shells", ()):
        from astermind.engine import STARLINK_RADIO_PROFILES
        reference = STARLINK_RADIO_PROFILES.get(shell.get('radio_profile'), {})
        hardware = reference.get('terminals', {})
        defaults = reference.get('simulation_defaults', {})
        if defaults:
            shell['radio_capacity_bps'] = {**defaults['radio_capacity_bps'], **shell.get('radio_capacity_bps', {})}
            shell['capacity_assumptions'] = [*defaults['capacity_assumptions'], *shell.get('capacity_assumptions', ())]
            for key in ('optical_terminal_bandwidth_bps', 'terminal_assumptions'):
                shell.setdefault(key, deepcopy(defaults[key]))
        physical_profiles = _hardware_profiles(defaults.get('hardware_link_profiles', {}), shell.get('hardware_link_profiles', {}))
        directional_ids = {}
        for kind, profile in physical_profiles.items():
            if kind not in {'communication_isl', 'user_uplink', 'user_downlink', 'feeder_uplink', 'feeder_downlink'}:
                raise ValueError('unsupported hardware_link_profiles kind')
            identity = f"{shell['shell_id']}__hardware__{kind}"
            if identity in network['link_profiles']:
                raise ValueError('generated hardware profile conflicts with explicit link_profiles')
            network['link_profiles'][identity] = deepcopy(profile)
            if kind == 'communication_isl':
                shell['isl_profile'] = identity
            else:
                directional_ids[kind] = identity
        name = shell["shell_id"]
        shell.setdefault('topology', 'grid')
        rows, layout = source_rows(shell["source"], name, topology=shell['topology'])
        if shell['topology'] in {'grid', 'honeycomb'}:
            shell['grid_shape'] = [shell['source']['planes'], shell['source']['satellites_per_plane']]
            sources[-1].update(
                placed_satellites=sum(p is not None for p, s in layout),
                unplaced_satellites=sum(p is None for p, s in layout),
                missing_grid_positions=math.prod(shell['grid_shape']) - sum(p is not None for p, s in layout),
                topology_repair=shell.get('topology_repair', network.get('topology_repair', 'none')))
        if shell['source']['type'] == 'dawn_dusk':
            sources[-1]['design_model'] = 'first_order_j2_mean_sun_ltan_no_station_keeping'
        shells.append(
            {
                k: shell[k]
                for k in (
                    "shell_id",
                    "topology",
                    "wrap_planes",
                    "wrap_slots",
                    "cross_plane_offset",
                    "isl_profile",
                    "explicit_pairs",
                    "grid_shape",
                    "topology_repair",
                )
                if k in shell
            }
        )
        for row, (plane, slot) in zip(rows, layout):
            elements.append(row)
            nodes.append(
                dict(
                    node_id=row["satellite_id"],
                    role="communication",
                    shell_id=name,
                    plane=plane,
                    slot=slot,
                    isl_terminals=shell.get("isl_terminals", hardware.get('optical_count', 6)),
                    sgl_terminals=shell.get("sgl_terminals", 1),
                    sgl_profiles=directional_ids,
                    **{k: shell[k] for k in ('radio_profile', 'radio_capacity_bps', 'capacity_assumptions',
                        'optical_terminal_bandwidth_bps', 'terminal_assumptions') if k in shell},
                    **{k: shell[k] for k in ('max_user_links', 'max_gateway_links') if k in shell},
                )
            )
    for formation in config.get("compute_formations", ()):
        name = formation["name"]
        cluster = formation["cluster"]
        count = cluster["satellites_per_cluster"]
        if type(count) is not int or count < 1:
            raise ValueError("satellites_per_cluster must be a positive integer")
        egress = formation.get('egress_policy', 'multi')
        if egress not in {'single', 'multi', 'free'}:
            raise ValueError('egress_policy must be single, multi, or free')
        ingress_members = formation.get('ingress_members', [0])
        if egress != 'free' and (not isinstance(ingress_members, (list, tuple)) or not ingress_members
            or any(type(i) is not int or not 0 <= i < count for i in ingress_members)
            or len(set(ingress_members)) != len(ingress_members)
            or (egress == 'single' and len(ingress_members) != 1)):
            raise ValueError('ingress_members must identify unique physical cell members; single requires one')
        if egress == 'free':
            ingress_members = list(range(count))
        if formation['source']['type'] == 'tle':
            physical, _ = source_rows(formation['source'], name, topology='explicit')
            by_catalog = {str(row['satnum']): row for row in physical}
            groups = formation['source'].get('cell_members')
            if not isinstance(groups, dict) or not groups:
                raise ValueError('compute TLE requires source.cell_members: cell ID -> ordered catalog numbers')
            membership = []
            for cell_name, catalog in groups.items():
                if (not isinstance(cell_name, str) or not cell_name or not isinstance(catalog, (list, tuple))
                    or len(catalog) != count or any(str(k) not in by_catalog for k in catalog)):
                    raise ValueError('compute TLE cell_members must match satellites_per_cluster and known catalog numbers')
                cell = f'{name}-{cell_name}'
                cell_topologies[cell] = formation.get('topology', 'ring')
                cell_egress_policies[cell] = egress
                for member, number in enumerate(catalog):
                    membership.append(str(number))
                    row = by_catalog[str(number)]
                    elements.append(row)
                    nodes.append(dict(node_id=row['satellite_id'], role='compute', shell_id=name,
                        cell_id=cell, cell_ingress=member in ingress_members, compute_target=member == 0,
                        isl_terminals=formation.get('isl_terminals', 4),
                        sgl_terminals=formation.get('sgl_terminals', 1 if actual_scenario == 'compute_only' else 0),
                        **{k:formation[k] for k in ('max_user_links','max_gateway_links') if k in formation}))
            if len(membership) != len(set(membership)) or set(membership) != set(by_catalog):
                raise ValueError('compute TLE cell_members must partition every imported satellite exactly once')
            keepout = float(cluster.get('d_keepout_m', 0.))
            if not math.isfinite(keepout) or keepout < 0:
                raise ValueError('compute TLE keepout must be finite and nonnegative')
            sources[-1].update(formation_motion='observed_per_member_sgp4_no_synthetic_offsets', keepout_m=keepout)
            continue
        hubs, layout = source_rows(formation["source"], name, True)
        shape = cluster.get("grid_shape", f"{count}x1").lower().split("x")
        if (
            len(shape) != 2
            or any(not x.isdigit() or int(x) < 1 for x in shape)
            or math.prod(map(int, shape)) != count
        ):
            raise ValueError(
                "grid_shape dimensions must multiply to satellites_per_cluster"
            )
        columns, rows = map(int, shape)
        pitch = float(cluster.get("formation_pitch_m", 500.0))
        keepout = float(cluster.get("d_keepout_m", 0.0))
        if (
            not math.isfinite(pitch)
            or pitch <= 0
            or not math.isfinite(keepout)
            or keepout < 0
            or (count > 1 and pitch <= keepout)
        ):
            raise ValueError(
                "formation pitch must be positive and greater than keepout"
            )
        if cluster.get("formation_frame", "lvlh") != "lvlh":
            raise ValueError("formation_frame must be lvlh")
        for hub, (plane, slot) in zip(hubs, layout):
            cell = hub["satellite_id"]
            cell_topologies[cell] = formation.get("topology", "ring")
            cell_egress_policies[cell] = egress
            n = hub["mean_motion_rev_per_day"] * 2 * math.pi / 86400
            radius = (398600.8 / n**2) ** (1 / 3)
            inclination = math.radians(hub["inclination_deg"])
            u = math.radians(hub["mean_anomaly_deg"] + hub["argument_of_perigee_deg"])
            for member in range(count):
                row = deepcopy(hub)
                along = ((member % columns) - (columns - 1) / 2) * pitch / 1000
                across = ((member // columns) - (rows - 1) / 2) * pitch / 1000
                if abs(math.sin(inclination)) < 1e-8 and abs(across) > 0:
                    raise ValueError(
                        "equatorial cross-track formation requires explicit member elements"
                    )
                di = across / radius * math.sin(u)
                draan = (
                    -across / radius * math.cos(u) / math.sin(inclination)
                    if across
                    else 0.0
                )
                # Parallel relative eccentricity/inclination vectors keep
                # radial separation when the cross-track separation vanishes.
                # Compensate the eccentricity-induced along-track offset so
                # the requested initial along/cross grid is preserved.
                du = (along - 2 * across) / radius - math.cos(inclination) * draan
                row["satellite_id"] = f"{cell}-m{member:03d}"
                row["raan_deg"] = _wrap_degrees(hub["raan_deg"] + math.degrees(draan))
                row["inclination_deg"] = hub["inclination_deg"] + math.degrees(di)
                omega = math.radians(hub['argument_of_perigee_deg'])
                if across:
                    ex = hub['eccentricity'] * math.cos(omega) + di
                    ey = hub['eccentricity'] * math.sin(omega) + math.sin(inclination) * draan
                    row['eccentricity'] = math.hypot(ex, ey)
                    omega = math.atan2(ey, ex)
                    row['argument_of_perigee_deg'] = _wrap_degrees(math.degrees(omega))
                row["mean_anomaly_deg"] = _wrap_degrees(hub['mean_anomaly_deg'] + hub['argument_of_perigee_deg']
                                                       + math.degrees(du - omega))
                if not 0 <= row['eccentricity'] < 1:
                    raise ValueError('formation offsets exceed eccentricity limits')
                if not 0 <= row["inclination_deg"] <= 180:
                    raise ValueError("formation offsets exceed inclination limits")
                elements.append(row)
                if (
                    any(
                        type(i) is not int or not 0 <= i < count
                        for i in ingress_members
                    )
                    or not ingress_members
                ):
                    raise ValueError(
                        "ingress_members must identify physical cell members"
                    )
                nodes.append(
                    dict(
                        node_id=row["satellite_id"],
                        role="compute",
                        shell_id=name,
                        cell_id=cell,
                        cell_ingress=member in ingress_members,
                        compute_target=member == 0,
                        isl_terminals=formation.get("isl_terminals", 4),
                        sgl_terminals=formation.get("sgl_terminals", 1 if actual_scenario == 'compute_only' else 0),
                        **{k: formation[k] for k in ('max_user_links', 'max_gateway_links') if k in formation},
                    )
                )
        sigmas = [
            float(cluster.get(key, 0.0))
            for key in ("sigma_navigation_m", "sigma_control_m")
        ]
        if any(not math.isfinite(value) or value < 0 for value in sigmas):
            raise ValueError(
                "formation uncertainty parameters must be finite and nonnegative"
            )
        uncertainty = math.hypot(*sigmas)
        sources[-1].update(
            formation_motion="free_sgp4_ei_separated_mean_element_offsets_v2",
            separation_design="parallel_relative_eccentricity_and_inclination_vectors",
            separation_reference="https://doi.org/10.2514/1.23015",
            initial_target_pitch_m=pitch,
            keepout_m=keepout + 3 * uncertainty,
            uncertainty_model="three_sigma_clearance_margin_no_stochastic_displacement",
        )
    for station in result["ground_ingresses"]:
        nodes.append(
            dict(
                node_id=station["ingress_id"],
                role="ground",
                isl_terminals=0,
                sgl_terminals=station.get("terminal_count", 1),
                ground_kind=station.get('ground_kind', 'gateway'),
            )
        )
    # Operational fields stay outside the geometry-only ground contract.
    result["ground_ingresses"] = [
        {k: v for k, v in station.items() if k not in {'terminal_count', 'ground_kind'}}
        for station in result["ground_ingresses"]
    ]
    observed = {
        r["satnum"]
        for r in elements
        if any(
            s["type"] == "tle" and r["satellite_id"].startswith(s["name"] + "-")
            for s in sources
        )
    }
    used = set(observed)
    next_number = 1
    for row in elements:
        if any(
            s["type"] == "tle" and row["satellite_id"].startswith(s["name"] + "-")
            for s in sources
        ):
            continue
        while next_number in used:
            next_number += 1
        row["satnum"] = next_number
        used.add(next_number)
    fitted = bool(sources) and all(s["observationally_fitted"] for s in sources)
    result["elements"] = dict(
        schema_version="orbitserve.sgp4_element_bundle.v1",
        element_semantics="sgp4_mean_elements",
        gravity_model="wgs72",
        ops_mode="improved",
        time_scale="utc",
        satellites=elements,
        provenance=dict(
            source_kind="observationally_fitted_general_perturbations"
            if fitted
            else "synthetic_sgp4_mean_elements",
            source_name="multi-shell-and-physical-formations",
            source_version="1",
            observationally_fitted=fitted,
        ),
    )
    network["nodes"] = nodes
    network["communication_shells"] = shells
    network["cell_topologies"] = cell_topologies
    network['cell_egress_policies'] = cell_egress_policies
    network["communication_topology"] = "explicit"
    result["initialization_provenance"] = sources
    from astermind.engine import freeze_network_profiles
    result['network'] = freeze_network_profiles(result['network'])
    return result
