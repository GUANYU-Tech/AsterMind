"""Run an L2 scenario and save compact, per-batch and per-request results."""
import argparse
from itertools import chain, combinations
import json
import math
import os
from pathlib import Path
from time import perf_counter

from astermind.__main__ import json_default
from astermind.builder.config import SimulationConfig

HERE = Path(__file__).resolve().parent
DEPLOYMENTS = {
    "tp8-pp4": {"tp": 8, "dp": 1, "pp": 4, "ep": 1},
    "tp16-pp2": {"tp": 16, "dp": 1, "pp": 2, "ep": 1},
    "wideep": {"tp": 8, "dp": 4, "pp": 1, "ep": 32},
}
SATELLITES = ("a", "b", "c", "d")


def scenario(case, *, smoke=False, prefill_chunk_tokens=None):
    parallel = DEPLOYMENTS[case["deployment"]]
    model = case["model"]
    wideep = parallel["ep"] > 1
    devices, links, groups, ranks = [], [], {}, []
    for satellite in SATELLITES:
        gpus = [f"{satellite}g{i}" for i in range(8)]
        terminal = satellite + "t"
        ranks.extend(gpus)
        groups[satellite] = {"su_groups": [gpus]}
        devices.extend(dict(id=gpu, satellite=satellite, kind="gpu") for gpu in gpus)
        devices.append(dict(id=terminal, satellite=satellite, kind="terminal",
                            external_access=True, max_external_links=3, capacity_gbps=3600))
        for src, dst in chain(combinations(gpus, 2), ((gpu, terminal) for gpu in gpus)):
            links.append(dict(id=f"{src}-{dst}", source=src, target=dst,
                              bandwidth_gbps=3600, latency_ns=200))

    neighbors = (combinations(SATELLITES, 2) if case["topology"] == "fc" else
                 zip(SATELLITES, SATELLITES[1:] + SATELLITES[:1]))
    distance = case["distance_km"]
    for src, dst in neighbors:
        links.append(dict(id=f"isl-{src}{dst}", source=src + "t", target=dst + "t",
                          bandwidth_gbps=case["bandwidth_gbps"], distance_km=distance,
                          latency_ns=math.ceil(distance * 1e9 / 299792.458)))

    devices.extend((
        dict(id="gateway", satellite="a", kind="terminal", external_access=True),
        dict(id="user", satellite="ground", kind="terminal", external_access=True),
    ))
    links.extend((
        dict(id="local-gateway", source=ranks[0], target="gateway", bandwidth_gbps=3600),
        dict(id="ground-ingress", source="gateway", target="user", bandwidth_gbps=3600),
    ))

    cell = dict(
        id="l2", model=f"configs/model/{model}.json",
        hardware="configs/hardware/nvidia_h200.yaml", hardware_id="nvidia_h200",
        parallel=parallel, placements=[dict(rank=i, device=gpu) for i, gpu in enumerate(ranks)],
        ingress_device=ranks[0], output_device=ranks[parallel["tp"] * (parallel["pp"] - 1)],
        external_terminals=["gateway"], max_model_len=1152, max_requests=32,
        scheduler_adapter="vllm_0_30", wide_ep_enabled=wideep, expert_routing_seed=0,
        enable_chunked_prefill=False, prefix_caching=False,
    )
    if prefill_chunk_tokens is not None:
        if not 0 < prefill_chunk_tokens <= cell["max_model_len"]:
            raise ValueError("prefill chunk tokens must be between 1 and the Cell's max_model_len")
        cell.update(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=prefill_chunk_tokens,
            max_batched_tokens=cell["max_model_len"] * cell["max_requests"],
        )
    communication = dict(
        mode="surrogate", intra_mode="surrogate", execution="runtime", trace_service=False,
        default_hardware_id="nvidia_h200", runtime_protocol="inter_reference",
        inter_parameters=dict(hop_processing_us=0, startup_us=0),
    )
    if model == "deepseek-v4-pro":
        cell["kv_block_size_tokens"] = 256
        if wideep:
            communication["ep_strategy"] = "direct"

    count, prompt, output = (4, 96, 2) if smoke else (128, 1000, 100)
    return dict(
        physical=dict(cells=[dict(id="l2", satellites=SATELLITES),
                             dict(id="ground-resources", satellites=["ground"])],
                      satellites=[dict(id=sat) for sat in (*SATELLITES, "ground")],
                      devices=devices, links=links, intra_hierarchies=groups),
        ground_owners=["ground"], cells=[cell],
        requests=[dict(id=f"r{i:03}", cell="l2", user_device="user",
                       input_tokens=prompt, output_tokens=output, arrival_ns=0) for i in range(count)],
        compute_mode="midlayer", horizon_ns=300_000_000_000,
        routing=dict(policy="min_propagation"), communication=communication,
    )


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


class BatchProgress(list):
    def __init__(self, path, started):
        super().__init__()
        self.path = path
        self.started = started
        self.previous = perf_counter()

    def append(self, row):
        now = perf_counter()
        record = dict(
            step_id=row["step_id"], start_ns=row["start_time_ns"], end_ns=row["end_time_ns"],
            status=row.get("status", "completed"), failure=row.get("failure"), calls=len(row["calls"]),
            wall_interval_s=now - self.previous, elapsed_wall_s=now - self.started,
            simulated_batch_ms=(row["end_time_ns"] - row["start_time_ns"]) / 1e6,
        )
        self.previous = now
        super().append(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=json_default) + "\n")
        print(f"Batch {record['step_id']} {record['status']} | "
              f"wall interval {record['wall_interval_s']:.3f}s | "
              f"cumulative wall {record['elapsed_wall_s']:.3f}s | "
              f"simulated batch {record['simulated_batch_ms']:.3f}ms", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_commit")
    parser.add_argument("--prefill-chunk-tokens", type=int, metavar="TOKENS",
                        help="Enable chunked prefill with this per-request token cap.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Run four requests with 96 input and 2 output tokens.")
    mode.add_argument("--validate", action="store_true", help="Validate the production workload and topology.")
    args = parser.parse_args()
    case = json.loads(args.case.read_bytes())
    try:
        raw = scenario(case, smoke=args.smoke, prefill_chunk_tokens=args.prefill_chunk_tokens)
    except ValueError as error:
        parser.error(str(error))
    config = SimulationConfig.model_validate(raw)
    config.topology()
    save(args.output / "config.json", raw)
    if args.validate:
        print(f"Validated {args.case.parent.name}: {len(config.requests)} requests", flush=True)
        return

    if case["model"] == "deepseek-v4-pro":
        os.environ["ORBITServe_V4_EXPERIMENTAL_MEMORY_PROJECTION"] = str(HERE / "deepseek-v4-h200-memory.json")

    from astermind.common.bootstrap import activate
    from astermind.engine.record_store import CompressedRecords
    from astermind.engine.simulation import Simulation

    activate()
    from src.serving.backend import vllm_scheduler_process

    vllm_scheduler_process._STARTUP_TIMEOUT_S = 240.0
    started = perf_counter()
    result = dict(case=case, source_commit=args.source_commit, smoke=args.smoke, status="failed")
    try:
        with Simulation(config, startup_progress=print) as sim:
            runtime = sim.cells["l2"].backend.runtime_info
            result["effective_scheduler"] = [
                {k: info[k] for k in (
                    "effective_chunked_prefill", "effective_max_num_batched_tokens",
                    "effective_long_prefill_token_threshold",
                )}
                for info in runtime.get("replicas", [runtime])
            ]
            print(f"Effective scheduler: {result['effective_scheduler']}", flush=True)
            sim.network.communication_policy.trace = CompressedRecords()
            for executor in sim.executors.values():
                executor.records = BatchProgress(args.output / "batches.jsonl", started)
            report = sim.run()
            result.update({k: report[k] for k in (
                "status", "requests", "measurement", "batches", "communication_runtime_stats")})
        if result["status"] != "completed":
            raise RuntimeError("Some requests failed. See the saved result.")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["total_wall_s"] = perf_counter() - started
        save(args.output / "result.json", result)
    print(f"Completed in {result['total_wall_s']:.1f}s wall. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
