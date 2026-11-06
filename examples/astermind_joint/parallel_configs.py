"""Validated serving deployments for the full communication-shell scenario."""

import json
from dataclasses import asdict
from pathlib import Path

from astermind.config import AsterMindConfig
from midlayer.serving.workload.runtime_inputs import ExpertRoutingInput


def serving_config(repository, parallel):
    repository = Path(repository)
    payload = json.loads(
        (repository / "examples/midlayer-qwen3-4b-tp2.json").read_text()
    )
    pool = payload["deployment"]["worker_pools"][0]
    pp = 2 if parallel == "tp_pp" else 1
    count = 2 * pp
    pool["parallelism"] = {"tensor_parallel_size": 2, "pipeline_parallel_size": pp}
    if pp == 2:
        pool["pipeline_p2p"] = [
            {
                "source_stage": 0,
                "destination_stage": 1,
                "algorithm": "send_recv",
                "link": "infiniband",
                "topology": "point_to_point",
                "resource_id": "pipeline-network",
                "backend": "flowsim",
            }
        ]
    if parallel == "ep":
        payload["model"] = "qwen3-30b-a3b"
        pool["parallelism"]["enable_expert_parallel"] = True
        payload["runtime_inputs"]["expert_routing"] = asdict(
            ExpertRoutingInput.seeded_assumption("full-serving-ep-demo", root_seed=17)
        )
    payload["hardware"] = {
        "satellite_count": count,
        "accelerators_per_satellite": 1,
        "satellite_ids": [f"rank-{i}" for i in range(count)],
        "default_hardware_id": "nvidia_a800",
    }
    payload["deployment"]["placements"] = [
        {
            "worker_pool": "main",
            "parallel_coordinate": {"dp": 0, "pp": p, "cp": 0, "tp": t},
            "satellite_id": f"rank-{p * 2 + t}",
            "local_index": 0,
        }
        for p in range(pp)
        for t in range(2)
    ]
    payload["runtime"]["timeout_seconds"] = 600
    return AsterMindConfig.model_validate(payload)
