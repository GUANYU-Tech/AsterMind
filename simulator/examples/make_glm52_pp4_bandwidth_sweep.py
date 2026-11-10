"""Generate the six four-satellite GLM-5.2 A800 bandwidth scenarios."""

import argparse
import json
import math
from pathlib import Path


RANKS_PER_SATELLITE = (8, 16, 32)
ISL_BANDWIDTH_GBPS = (400, 1000)
ISL_LATENCY_NS = math.ceil(1000 / 299_792_458 * 1e9)


def make_scenario(tp, bandwidth_gbps, *, expert_parallel=True):
    template = Path(__file__).with_name("glm52_single_cell_32ranks.json")
    scenario = json.loads(template.read_text(encoding="utf-8"))
    physical = scenario["physical"]
    physical["devices"] = [device for device in physical["devices"] if device["kind"] != "gpu"]
    physical["links"] = [link for link in physical["links"] if not link["id"].startswith("local-")]
    for link in physical["links"]:
        if link["id"].startswith("isl-"):
            link.update(bandwidth_gbps=bandwidth_gbps, latency_ns=ISL_LATENCY_NS)
        elif link["id"].startswith("terminal-local-"):
            link["bandwidth_gbps"] = max(ISL_BANDWIDTH_GBPS)

    cell, = scenario["cells"]
    placements = []
    for stage in range(4):
        for gpu in range(tp):
            device_id = f"gpu-0-{stage}-{gpu}"
            physical["devices"].append({"id": device_id, "satellite": f"sat-0-{stage}", "kind": "gpu"})
            physical["links"].append({"id": f"local-0-{stage}-{gpu}", "source": device_id,
                                      "target": f"fabric-0-{stage}", "bandwidth_gbps": 1000,
                                      "latency_ns": 100})
            placements.append({"rank": stage * tp + gpu, "device": device_id, "pool": "main"})
    cell.update(parallel={"tp": tp, "pp": 4, "ep": tp if expert_parallel else 1}, placements=placements,
                scheduler_adapter="vllm_0_30",
                max_model_len=1100, max_batched_tokens=2048, max_requests=32,
                num_kv_blocks=2304, kv_block_size_tokens=16)

    scenario["requests"] = [
        {"id": f"R{index + 1}", "cell": cell["id"], "user_device": "user", "arrival_ns": 0,
         "input_tokens": 1000, "output_tokens": 100, "input_bytes": 4000, "output_bytes_per_token": 4}
        for index in range(32)
    ]
    scenario["horizon_ns"] = 300_000_000_000
    return scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-expert-parallel", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tp in RANKS_PER_SATELLITE:
        for bandwidth in ISL_BANDWIDTH_GBPS:
            path = args.output_dir / f"glm52_pp4_{4 * tp}gpu_{bandwidth}gbps.json"
            scenario = make_scenario(tp, bandwidth, expert_parallel=not args.no_expert_parallel)
            path.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
