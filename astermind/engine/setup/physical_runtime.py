"""Create live Engine network state from Builder-prepared physical data."""

def create_physical_runtime(config, root, metadata, concrete, spec, trace):
    from astermind.engine.topology import candidate_links
    from astermind.engine.setup.faults import realize_runtime_faults
    from astermind.engine.state.network import NetworkState

    if config.fast_orbital_route_frames is not None:
        from astermind.builder.physical.snapshot_store import SnapshotSource
        from astermind.engine.state.link_state_store import INDEX_DIRECTORY
        from astermind.engine.setup.faults import random_faults_enabled

        if not (root / INDEX_DIRECTORY).is_dir():
            raise ValueError("Fast orbital routing requires the indexed physical dataset")
        if spec.faults or random_faults_enabled(spec.random_faults) or config.additional_cell_links or config.disabled_physical_links:
            raise ValueError("Fast orbital routing requires the original prepared topology")
        physical = SnapshotSource(root, initial_spec=spec)
        physical.dataset_identity = metadata["input_sha256"]
        physical.dataset_root = str(root)
        return physical

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
    return physical


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
