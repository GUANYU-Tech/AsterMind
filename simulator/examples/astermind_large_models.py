"""Make the reproducible two-Cell Kimi + DeepSeek smoke scenario."""

import argparse
import json
from pathlib import Path


def scenario(models=("kimi-k2.6", "deepseek-v4-flash")):
    hardware_profiles = {
        "kimi-k2.6": ("nvidia_a800", 8, 2, True),
        "kimi-k2.6-h200": ("nvidia_h200", 8, 1, True),
        "kimi-k2.6-pd": ("nvidia_h200", 8, 1, True),
        "deepseek-v3": ("nvidia_h200", 8, 1, False),
        "deepseek-v4-flash": ("nvidia_h200", 4, 1, False),
        "deepseek-v4-pro": ("nvidia_h200", 8, 1, False),
    }
    physical = {
        "cells": [dict(id="ground-resources", satellites=["ground"])],
        "satellites": [dict(id="ground")],
        "devices": [dict(id="user", satellite="ground", kind="terminal",
                         external_access=True, capacity_gbps=1, max_external_links=len(models))],
        "links": [],
    }
    cells, requests, satellite_roles = [], [], {}
    for index, model in enumerate(models):
        hardware, tp, pp, ep = hardware_profiles[model]
        pd = model == "kimi-k2.6-pd"
        stages = 2 if pd else pp
        cell_id = f"cell-{model}"
        owners, placements = [], []
        for stage in range(stages):
            sat = f"sat-{index}-{stage}"
            fabric = f"fabric-{index}-{stage}"
            terminal = f"terminal-{index}-{stage}"
            owners.append(sat)
            satellite_roles[sat] = "mixed"
            physical["satellites"].append(dict(id=sat))
            physical["devices"].extend([
                dict(id=fabric, satellite=sat, kind="switch", capacity_gbps=4000),
                dict(id=terminal, satellite=sat, kind="terminal", external_access=True,
                     max_external_links=2 if stage == 0 and stages > 1 else 1),
            ])
            physical["links"].append(dict(id=f"terminal-local-{index}-{stage}", source=fabric,
                                          target=terminal, bandwidth_gbps=50))
            for lane in range(tp):
                gpu = f"gpu-{index}-{stage}-{lane}"
                pool = ("prefill" if stage == 0 else "decode") if pd else "main"
                placements.append(dict(rank=stage * tp + lane, device=gpu, pool=pool))
                physical["devices"].append(dict(id=gpu, satellite=sat, kind="gpu"))
                physical["links"].append(dict(id=f"local-{index}-{stage}-{lane}", source=gpu,
                                              target=fabric, bandwidth_gbps=1000, latency_ns=100))
            if stage:
                physical["links"].append(dict(id=f"isl-{index}-{stage}", source=f"terminal-{index}-{stage-1}",
                                              target=terminal, bandwidth_gbps=10, latency_ns=100_000))
        physical["cells"].append(dict(id=cell_id, satellites=owners))
        physical["links"].append(dict(id=f"ground-link-{index}", source="user", target=f"terminal-{index}-0",
                                      bandwidth_gbps=1, latency_ns=150_000))
        model_config = model.removesuffix("-h200").removesuffix("-pd")
        cells.append(dict(id=cell_id, model=f"configs/model/{model_config}.json",
                          hardware=f"configs/hardware/{hardware}.yaml", hardware_id=hardware,
                          parallel=dict(tp=tp, pp=pp, ep=tp if ep else 1), expert_routing_seed=42,
                          deployment="pd_disagg" if pd else "colocated",
                          placements=placements, ingress_device=placements[0]["device"],
                          output_device=placements[(stages - 1) * tp]["device"],
                          external_terminals=[f"terminal-{index}-0"], max_batched_tokens=16))
        for wave, (prompt, output) in enumerate(((8, 2), (20, 2))):
            requests.append(dict(id=f"{model}-{wave}", cell=cell_id, arrival_ns=wave * 400_000,
                                 user_device="user", input_tokens=prompt, output_tokens=output,
                                 input_bytes=10_000, output_bytes_per_token=1000))
    return dict(physical=physical, ground_owners=["ground"], satellite_roles=satellite_roles,
                cells=cells, requests=requests, horizon_ns=10_000_000_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("models", nargs="*")
    args = parser.parse_args()
    data = scenario(tuple(args.models)) if args.models else scenario()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
