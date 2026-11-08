"""Reuse OrbitServe geometry and ISLSim; keep orbit assumptions visible."""
from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import sys


def _islsim():
    # Upstream link_model uses an absolute `from config import ...`. Load that
    # dependency temporarily without leaving a global `config` module behind.
    root = Path(__file__).parent / "_vendor/islsim"
    name = "network_sim.physical._vendor.islsim.config"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "config.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    config = sys.modules[name]
    model_name = "network_sim.physical._vendor.islsim.link_model"
    if model_name not in sys.modules:
        old = sys.modules.get("config")
        sys.modules["config"] = config
        try:
            spec = importlib.util.spec_from_file_location(model_name, root / "link_model.py")
            model = importlib.util.module_from_spec(spec)
            sys.modules[model_name] = model
            spec.loader.exec_module(model)
        finally:
            if old is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = old
    def make_config(**values):
        values = dict(values)
        for name, constructor in (("tx", config.TXConfig), ("rx", config.RXConfig), ("channel", config.ChannelConfig)):
            if isinstance(values.get(name), dict):
                values[name] = constructor(**values[name])
        return config.FSOConfig(**values)
    return make_config, sys.modules[model_name].compute_link


def orbit_grid(physical, *, orbit, ground_owners, horizon_ns):
    if orbit is None:
        return None
    if orbit.trace_manifest:
        from orbitserve.orbit import load_orbit_trace
        bundle = load_orbit_trace(orbit.trace_manifest)
        samples = bundle.ephemeris_samples
    elif orbit.elements_file:
        from orbitserve.orbit import propagate_sgp4_teme, transform_sgp4_teme_to_itrs
        times = list(range(0, horizon_ns, orbit.sample_step_ns)) + [horizon_ns]
        propagation = propagate_sgp4_teme(orbit.elements_file, simulation_epoch_utc=orbit.simulation_epoch_utc,
                                        sample_times_s=[at / 1e9 for at in times])
        samples = [sample.to_dict() for sample in transform_sgp4_teme_to_itrs(propagation).samples]
    else:
        samples = orbit.samples
    by_time = {}
    for sample in samples:
        if set(sample) - {"satellite_id", "time_s", "position_itrs_km", "velocity_itrs_km_s",
                          "position_teme_km", "velocity_teme_km_s", "source_reference_frame", "earth_fixed_reference_frame"}:
            raise ValueError("orbit sample contains unsupported fields")
        if sample.get("earth_fixed_reference_frame", "ITRS") != "ITRS":
            raise ValueError("orbit input must use ITRS positions")
        at = round(sample["time_s"] * 1e9)
        if at < 0:
            raise ValueError("orbit sample precedes scenario origin")
        position = tuple(sample["position_itrs_km"])
        if len(position) != 3 or not all(math.isfinite(x) for x in position):
            raise ValueError("invalid ITRS position")
        here = by_time.setdefault(at, {})
        if sample["satellite_id"] in here:
            raise ValueError("duplicate orbit sample")
        here[sample["satellite_id"]] = position
    owners = {s["id"] for s in physical["satellites"]} - set(ground_owners)
    for at, positions in by_time.items():
        if orbit.mode == "reference_offsets":
            reference = positions.get(orbit.reference_satellite)
            if reference is None:
                raise ValueError("reference position missing from orbit grid")
            if set(orbit.offsets_itrs_km) != owners - {orbit.reference_satellite}:
                raise ValueError("offsets must cover every non-reference member")
            positions = {orbit.reference_satellite: reference, **{
                member: tuple(a + b for a, b in zip(reference, offset))
                for member, offset in orbit.offsets_itrs_km.items()}}
            by_time[at] = positions
        if set(positions) != owners:
            raise ValueError("orbit grid must cover every satellite at every sample time")
    times = sorted(by_time)
    if not times or times[0] != 0 or times[-1] < horizon_ns:
        raise ValueError("orbit samples must cover [0, simulation horizon]")
    return [(at, by_time[at]) for at in times]


def prepare_environment(physical, *, orbit, ground_owners, horizon_ns, link_parameters):
    grid = orbit_grid(physical, orbit=orbit, ground_owners=ground_owners, horizon_ns=horizon_ns)
    devices = {d["id"]: d for d in physical["devices"]}
    fixed = {s["id"]: s.get("ecef_position_km") for s in physical["satellites"]}
    links = {link["id"]: link for link in physical.get("links", [])}
    if set(link_parameters) - set(links):
        raise ValueError("link parameter binding references unknown explicit link")
    report = dict(orbit_mode=orbit.mode if orbit else "unspecified_static",
                  sample_times_ns=[at for at, _ in grid] if grid else [], links={},
                  formation_assumption="fixed ITRS offsets; no orbital error bound" if orbit and orbit.mode == "reference_offsets" else None)
    if grid:
        from orbitserve.orbit.sgp4.geometry import _ellipsoid_segment_clearance
        for satellite in physical["satellites"]:
            if satellite["id"] in grid[0][1]:
                satellite["ecef_position_km"] = grid[0][1][satellite["id"]]
    for identifier, link in links.items():
        first, second = (devices[link[k]]["satellite"] for k in ("source", "target"))
        if first == second:
            continue
        samples = grid if grid else [(0, fixed)]
        states, budgets = [], []
        params = link_parameters.get(identifier)
        for at, positions in samples:
            a = positions.get(first, fixed.get(first))
            b = positions.get(second, fixed.get(second))
            distance = math.dist(a, b) if a is not None and b is not None else None
            feasible = link.get("enabled", True)
            if grid:
                if a is None or b is None:
                    raise ValueError("ground access with orbit needs ground ECEF position")
                if first in ground_owners or second in ground_owners:
                    ground, satellite = (a, b) if first in ground_owners else (b, a)
                    # WGS84 ellipsoid surface normal, not geocentric latitude.
                    up = (ground[0] / 6378.137**2, ground[1] / 6378.137**2, ground[2] / 6356.752314245**2)
                    delta = tuple(x - y for x, y in zip(satellite, ground))
                    sine = sum(x * y for x, y in zip(delta, up)) / (math.sqrt(sum(x*x for x in delta)) * math.sqrt(sum(x*x for x in up)))
                    elevation = math.degrees(math.asin(max(-1, min(1, sine))))
                    feasible &= elevation >= orbit.min_ground_elevation_deg
                else:
                    feasible &= _ellipsoid_segment_clearance(a, b, 0)[0] > 0 and distance <= orbit.max_isl_range_km
            if params and params.kind == "islsim":
                if first in ground_owners or second in ground_owners:
                    raise ValueError("ISLSim is an inter-satellite link model, not an atmospheric ground-link model")
                value = distance if distance is not None else params.distance_km
                if value is None or value <= 0:
                    raise ValueError("ISLSim requires a positive distance or orbit positions")
                FSOConfig, compute_link = _islsim()
                budget = compute_link(FSOConfig(**params.fso), value)
                feasible &= budget.feasible
                budgets.append(asdict(budget))
            states.append((at, feasible))
        if budgets:
            feasible_budgets = [b for b in budgets if b["feasible"]]
            # Explicit conservative sampled envelope. No fabricated continuous
            # rate changes on the older externally clocked FlowSim API.
            link["bandwidth_gbps"] = min((b["total_bw_gbps"] for b in feasible_budgets), default=1e-6)
            link["latency_ns"] = math.ceil(max((b["latency_ms"] for b in budgets), default=0) * 1e6)
            if not feasible_budgets:
                link["enabled"] = False
            report["links"][identifier] = dict(source="ISLSim", budgets=budgets,
                timing_policy="minimum feasible bandwidth and maximum delay over supplied samples")
        if grid:
            windows = []
            for (start, enabled), (end, _) in zip(states, states[1:]):
                if not enabled:
                    continue
                end = min(end, horizon_ns)
                if start >= end:
                    continue
                if windows and windows[-1]["end_ns"] == start:
                    windows[-1]["end_ns"] = end
                else:
                    windows.append(dict(start_ns=start, end_ns=end))
            link["availability_windows"] = windows
            report["links"].setdefault(identifier, {})["availability_windows"] = windows
    if grid and physical.get("link_policy") is None:
        intervals = [b[0] - a[0] for a, b in zip(grid, grid[1:])]
        physical["link_policy"] = dict(mode="simple_keep", check_interval_ns=min(intervals),
                                        initial_acquisition_ns=0, reacquisition_ns=0)
    return physical, report, grid
