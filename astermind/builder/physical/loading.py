"""Use OrbitServe's existing prepared physics, identity checks and fault model."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory


def load_physics(config, output_dir=None):
    """Return a live NetworkState and its lifetime owner; never regenerate an explicit dataset."""
    from astermind.builder.physical.physical_dataset import load_dataset

    from astermind.builder.orbit import load_orbit_trace
    from astermind.engine import NetworkSpec
    from astermind.engine.topology import candidate_links
    from astermind.engine.setup.faults import realize_runtime_faults
    from astermind.engine.state.network import NetworkState

    lifetime = None
    reference = config.physical_dataset
    if reference is None and config.physical_network is not None:
        from astermind.builder.physical.network import run_network_experiment

        value = config.physical_network
        base = Path(".")
        if isinstance(value, str):
            path = Path(value)
            base, value = path.parent, json.loads(path.read_text(encoding="utf8"))
        if output_dir is None:
            lifetime = TemporaryDirectory(prefix="orbitserve-cells-physics-")
        destination = Path(lifetime.name) if lifetime is not None else Path(output_dir)
        try:
            run_network_experiment(
                value, destination, base_dir=base, _workers=1, export_mode="core"
            )
        except BaseException:
            if lifetime is not None:
                lifetime.cleanup()
            raise
        reference = destination
    if reference is None:
        if config.additional_cell_links or config.disabled_physical_links:
            raise ValueError("Cell 链路编辑需要绑定物理数据")
        if config.runtime_network:
            raise ValueError("runtime physical faults require canonical physical data")
        return None, None
    try:
        root, metadata, _ = load_dataset(reference)
        concrete = json.loads(
            (root / "materialized_input.json").read_text(encoding="utf8")
        )
        spec = NetworkSpec.from_value(
            {
                **concrete["network"],
                **config.runtime_network,
                "transport_model": "runtime_comm",
            }
        )
        trace = load_orbit_trace(root / "orbit_trace/trace_manifest.json")
        boundaries = sorted({row["time_s"] for row in trace.ephemeris_samples})
        spec = realize_runtime_faults(spec, boundaries, candidate_links(spec))
        physical = NetworkState(
            trace,
            spec,
            concrete["ground_ingresses"],
            physical_state=root / "physical_state.sqlite",
        )
        apply_cell_topology(physical, config)
        physical.dataset_identity = metadata["input_sha256"]
        physical.dataset_root = str(root)
        return physical, lifetime
    except BaseException:
        if lifetime is not None:
            lifetime.cleanup()
        raise


def apply_cell_topology(physical, config):
    """Apply explicit Cell connections using real orbit geometry and allocator."""
    from astermind.engine.state.interventions import _candidate, _add_physics

    if not config.additional_cell_links and not config.disabled_physical_links:
        return
    for pair in config.additional_cell_links:
        a, b = physical.nodes.get(pair.source), physical.nodes.get(pair.target)
        if (
            not a
            or not b
            or a.role != "compute"
            or b.role != "compute"
            or not a.cell_id
            or a.cell_id != b.cell_id
        ):
            raise ValueError("新增 Cell 链路必须连接同一 Cell 的两颗计算卫星")
        edge, added = _candidate(physical, pair.source, pair.target)
        if not added:
            raise ValueError("新增 Cell 链路与现有候选链路重复")
        _add_physics(physical, edge, physical.start_s)
    known = {e["link_id"] for e in physical.edges}
    disabled = set(config.disabled_physical_links)
    if len(disabled) != len(config.disabled_physical_links) or not disabled <= known:
        raise ValueError("禁用链路必须是存在且不重复的物理链路")
    physical.manual_disabled_links.update(disabled)
    physical.revision += 1
    physical._compile(from_s=physical.start_s)
    physical.frame_times = [f["time_s"] for f in physical.frames]
