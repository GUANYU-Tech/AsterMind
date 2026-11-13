"""Use OrbitServe's existing prepared physics, identity checks and fault model."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory


def prepare_physics(config, output_dir=None):
    """Read/precompute physical inputs and return them with their lifetime owner."""
    from astermind.builder.physical.physical_dataset import load_dataset

    from astermind.builder.orbit import load_orbit_trace
    from astermind.builder.config.network import NetworkSpec

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
        return (root, metadata, concrete, spec, trace), lifetime
    except BaseException:
        if lifetime is not None:
            lifetime.cleanup()
        raise


def load_physics(config, output_dir=None):
    """Compatibility entry: return a live NetworkState and its lifetime owner."""
    prepared, lifetime = prepare_physics(config, output_dir)
    if prepared is None:
        return None, None
    try:
        from astermind.engine.setup.physical_runtime import create_physical_runtime
        return create_physical_runtime(config, *prepared), lifetime
    except BaseException:
        if lifetime is not None:
            lifetime.cleanup()
        raise


def apply_cell_topology(physical, config):
    """Compatibility import for explicit runtime Cell topology edits."""
    from astermind.engine.setup.physical_runtime import apply_cell_topology as apply
    return apply(physical, config)
