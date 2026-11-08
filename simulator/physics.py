"""Use OrbitServe's existing prepared physics, identity checks and fault model."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory


def load_physics(config, output_dir=None):
    """Return a live NetworkState and its lifetime owner; never regenerate an explicit dataset."""
    from orbitserve.experiments.physical_dataset import load_dataset

    from orbitserve.orbit import load_orbit_trace
    from orbitserve.runtime import NetworkSpec
    from orbitserve.runtime.engine.topology import candidate_links
    from orbitserve.runtime.setup.faults import realize_runtime_faults
    from orbitserve.runtime.state.network import NetworkState

    lifetime = None
    reference = config.physical_dataset
    if reference is None and config.physical_network is not None:
        from orbitserve.experiments.network import run_network_experiment

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
                "transport_model": config.backend,
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
        physical.dataset_identity = metadata["input_sha256"]
        physical.dataset_root = str(root)
        return physical, lifetime
    except BaseException:
        if lifetime is not None:
            lifetime.cleanup()
        raise
