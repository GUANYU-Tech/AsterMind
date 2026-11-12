"""Reproducible four-satellite serving-capacity experiments.

Planning and analysis need no native dependency. Execution uses the separately
prepared AsterMind checkout; every load/seed runs in an isolated process.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Workload(Contract):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    input_tokens: int = Field(strict=True, gt=0)
    output_tokens: int = Field(strict=True, gt=1)


class ServiceTarget(Contract):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    ttft_s: float = Field(gt=0)
    tpot_s: float = Field(gt=0)
    attainment: float = Field(default=0.95, gt=0, le=1)


class StudySpec(Contract):
    schema_version: str = "orbitserve.capacity_study.v1"
    models: tuple[str, ...] = ("all",)
    hardware: tuple[str, ...] = (
        "nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b",
    )
    cards_per_satellite: tuple[StrictInt, ...] = (1, 2, 4, 6, 8)
    link_bandwidth_gbps: tuple[float, ...] = (400, 1000)
    bandwidth_accounting: str = "bidirectional_aggregate"
    layouts: tuple[str, ...] = (
        "local_replicas", "cross_pairs", "cross_replicas", "cross_pipeline", "cross_tp", "target",
    )
    workloads: tuple[Workload, ...] = (
        Workload(name="short", input_tokens=512, output_tokens=128),
        Workload(name="medium", input_tokens=2048, output_tokens=256),
        Workload(name="long-input", input_tokens=8192, output_tokens=256),
    )
    targets: tuple[ServiceTarget, ...] = (
        ServiceTarget(name="interactive", ttft_s=2, tpot_s=0.05),
        ServiceTarget(name="relaxed", ttft_s=5, tpot_s=0.1),
    )
    request_rates: tuple[float, ...] = (0.5, 2, 8, 32, 128)
    seeds: tuple[StrictInt, ...] = (17, 29, 43)
    arrival_distribution: str = "poisson"
    warmup_s: float = Field(default=5, ge=0)
    measurement_s: float = Field(default=30, gt=0)
    drain_s: float = Field(default=30, ge=0)
    min_measured_requests: int = Field(default=20, strict=True, gt=0)
    max_requests_per_trial: int = Field(default=20000, strict=True, gt=0)
    max_concurrency_per_replica: int = Field(default=32, strict=True, gt=0)
    max_batched_tokens: int = Field(default=4096, strict=True, gt=0)
    kv_block_size_tokens: int = Field(default=16, strict=True, gt=0)
    memory_utilization: float = Field(default=0.9, gt=0, le=1)
    non_kv_reserve_fraction: float = Field(default=0.1, ge=0, lt=1)
    link_distance_km: float = Field(default=100, gt=0)
    local_gpu_bandwidth_gbps: float = Field(default=1800, gt=0)
    local_switch_bandwidth_gbps: float = Field(default=14400, gt=0)
    ground_bandwidth_gbps: float = Field(default=100, gt=0)
    ground_latency_s: float = Field(default=0.002, ge=0)
    local_latency_ns: int = Field(default=1000, strict=True, ge=0)
    acknowledgements_enabled: bool = Field(default=True, strict=True)
    ack_packet_payload_bytes: int = Field(default=1000, strict=True, gt=0)
    ack_every_packets: int = Field(default=1, strict=True, gt=0)
    ack_size_bytes: int = Field(default=64, strict=True, gt=0)
    timeout_s: int = Field(default=600, strict=True, gt=0)
    scheduler_startup_timeout_s: int = Field(default=180, strict=True, gt=0)
    max_consecutive_worker_failures: int = Field(default=3, strict=True, gt=0)
    recommendation_fraction: float = Field(default=0.95, gt=0, le=1)
    queue_growth_fraction: float = Field(default=0, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_study(self):
        if self.schema_version != "orbitserve.capacity_study.v1":
            raise ValueError("unsupported capacity study version")
        if self.bandwidth_accounting not in ("bidirectional_aggregate", "per_direction"):
            raise ValueError("declare bidirectional_aggregate or per_direction")
        if self.arrival_distribution not in ("poisson", "constant"):
            raise ValueError("arrival_distribution must be poisson or constant")
        allowed = {"local_replicas", "cross_pairs", "cross_replicas", "cross_pipeline", "cross_tp", "target"}
        if not self.layouts or set(self.layouts) - allowed:
            raise ValueError("unknown or empty deployment layouts")
        for name in ("models", "hardware", "cards_per_satellite", "link_bandwidth_gbps",
                     "request_rates", "seeds", "layouts", "workloads", "targets"):
            values = getattr(self, name)
            identities = [getattr(v, "name", v) for v in values]
            if not values or len(set(identities)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
        if any(type(n) is not int or not 1 <= n <= 8 for n in self.cards_per_satellite):
            raise ValueError("four-satellite main study permits 1..8 cards per satellite")
        if any(type(n) is not int or n < 0 for n in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if any(not math.isfinite(v) or v <= 0 for v in (*self.request_rates, *self.link_bandwidth_gbps)):
            raise ValueError("rates and bandwidths must be finite and positive")
        if self.memory_utilization <= self.non_kv_reserve_fraction:
            raise ValueError("memory utilization must exceed the non-KV reserve")
        if self.scheduler_startup_timeout_s > self.timeout_s:
            raise ValueError("Scheduler startup timeout must fit inside the trial wall-clock timeout")
        if "all" in self.models and self.models != ("all",):
            raise ValueError("all cannot be mixed with individual models")
        return self


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def directional_bandwidth(gbps, accounting):
    if accounting == "bidirectional_aggregate":
        return gbps / 2
    if accounting == "per_direction":
        return gbps
    raise ValueError("unknown bandwidth accounting")


def model_catalog(directory):
    result = {}
    for path in sorted(Path(directory).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        text = value.get("text_config", value)
        result[path.stem] = {
            "model": path.stem, "file": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model_type": text.get("model_type"),
            "compute_dtype": text.get("dtype", text.get("torch_dtype")),
            "quantization": value.get("quantization_config", text.get("quantization_config", {})).get("quant_method"),
            "layers": text.get("num_hidden_layers"),
            "reduced_model": path.stem.endswith("-l32"),
        }
    return result


def source_identity(repository, spec):
    """Bind resume to executable Python/config inputs without distributing them."""
    import importlib.util
    vendor = repository / "third_party/midlayer"
    paths = [*sorted((repository / "src/astermind").rglob("*.py")),
             *sorted((vendor / "midlayer").rglob("*.py")),
             *(vendor / "configs/hardware" / f"{hardware}.yaml" for hardware in spec.hardware),
             *(Path(__file__).parent / name for name in
               ("capacity_study.py", "capacity_native.py", "capacity_metrics.py"))]
    native = importlib.util.find_spec("_flowsim")
    if native and native.origin:
        paths.append(Path(native.origin))
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def plan_cases(spec, catalog):
    models = sorted(catalog) if spec.models == ("all",) else spec.models
    cases = []
    for model, hardware, cards, bandwidth, layout, workload in itertools.product(
        models, spec.hardware, spec.cards_per_satellite, spec.link_bandwidth_gbps,
        spec.layouts, spec.workloads,
    ):
        row = dict(model=model, hardware=hardware, cards_per_satellite=cards,
                   total_cards=4 * cards, link_bandwidth_gbps=bandwidth,
                   bandwidth_accounting=spec.bandwidth_accounting,
                   per_direction_gbps=directional_bandwidth(bandwidth, spec.bandwidth_accounting),
                   layout=layout, workload=workload.model_dump())
        row["case_id"] = digest(row)[:20]
        row["status"] = "planned" if model in catalog else "missing_model"
        cases.append(row)
    return cases


def placements(case, model_type):
    """Return replicas as rank->(satellite, local GPU); never share a GPU."""
    n, layout = case["cards_per_satellite"], case["layout"]
    if layout == "local_replicas":
        return n, 1, False, [[(s, t) for t in range(n)] for s in range(4)]
    if layout == "cross_replicas":
        return 2, 2, False, [[(s, slot) for s in range(4)] for slot in range(n)]
    if layout == "cross_pairs":
        return 2, 1, False, [[(s, slot), (s + 1, slot)] for slot in range(n) for s in (0, 2)]
    if layout == "cross_pipeline":
        return n, 4, False, [[(s, t) for s in range(4) for t in range(n)]]
    if layout == "cross_tp":
        return 4 * n, 1, False, [[(s, t) for s in range(4) for t in range(n)]]
    if layout == "target":
        axes = {"glm_moe_dsa": (8, 4), "kimi_k2": (8, 2), "qwen3_moe": (2, 1)}
        if model_type not in axes:
            raise NotImplementedError("no retained target layout for this model")
        tp, pp = axes[model_type]
        width = tp * pp
        if (4 * n) % width:
            raise NotImplementedError("retained target workers do not tile all allocated GPUs")
        devices = [(s, t) for s in range(4) for t in range(n)]
        return tp, pp, True, [devices[i:i + width] for i in range(0, 4 * n, width)]
    raise ValueError("unknown layout")


def replica_resources(group):
    """Describe one replica locally; host rank mapping retains global GPU IDs."""
    counts, local_indices = {}, []
    if len(set(group)) != len(group):
        raise ValueError("a replica cannot share a physical GPU between ranks")
    for satellite, slot in group:
        local_indices.append(counts.get(satellite, 0))
        counts[satellite] = counts.get(satellite, 0) + 1
    if len(set(counts.values())) != 1:
        raise NotImplementedError("native replica resource contract requires uniform cards per owner")
    return dict(satellite_count=len(counts), accelerators_per_satellite=next(iter(counts.values())),
                satellite_ids=[f"sat{s}" for s in counts]), local_indices


def arrivals(spec, rate, seed, workload):
    # A shared seed/rate gives identical arrivals across hardware, card count and bandwidth.
    rng = random.Random(seed)
    end, time_s, result = spec.warmup_s + spec.measurement_s, 0.0, []
    while True:
        time_s += rng.expovariate(rate) if spec.arrival_distribution == "poisson" else 1 / rate
        if time_s >= end:
            break
        if len(result) >= spec.max_requests_per_trial:
            raise ValueError("request cap would truncate the offered load; shorten or split the trial")
        result.append(dict(request_id=f"r{len(result):07d}", arrival_s=time_s,
                           input_tokens=workload["input_tokens"], output_tokens=workload["output_tokens"]))
    return result


def _isolated(payload, destination, timeout_s):
    write_json(destination / "input.json", payload)
    command = [sys.executable, "-m", "orbitserve.experiments.capacity_study", "_worker",
               str(destination / "input.json"), "--out", str(destination / "result.json")]
    with (destination / "execution.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == "posix")
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            write_json(destination / "result.json", {
                "status": "timeout", "input_sha256": digest(payload),
                "reason": "trial wall-clock timeout; no capacity inferred",
            })
            return
    if code and not (destination / "result.json").exists():
        write_json(destination / "result.json", {
            "status": "worker_failed", "input_sha256": digest(payload), "returncode": code,
        })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("catalog", "plan", "check", "run", "analyze", "_worker", "_check"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--astermind-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum trials this invocation; progress remains incomplete")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.action == "_check":
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        from .capacity_native import execute
        cache, rows = {}, []
        for index, case in enumerate(payload["cases"]):
            key = (case["model"], case["hardware"], case["cards_per_satellite"],
                   case["layout"], case["workload"]["input_tokens"], case["workload"]["output_tokens"])
            if key not in cache:
                request = dict(spec=payload["spec"], case=case, seed=payload["spec"]["seeds"][0],
                               rate=1, astermind_root=payload["astermind_root"],
                               model=payload["catalog"].get(case["model"]), preflight_only=True)
                try:
                    cache[key] = execute(request, args.out.parent)
                except Exception as error:
                    cache[key] = dict(status="invalid_configuration", reason=f"{type(error).__name__}: {error}")
            rows.append(dict(case_id=case["case_id"], **cache[key]))
            if (index + 1) % 100 == 0:
                print(f"capability checks: {index + 1}/{len(payload['cases'])}", flush=True)
        write_json(args.out, dict(schema_version="orbitserve.capacity_capabilities.v1",
                                 plan_sha256=digest(payload), cases=rows))
        from .capacity_metrics import capability_report
        capability_report(payload, rows, args.out.parent)
        return 0
    if args.action == "_worker":
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        try:
            from .capacity_native import execute
            result = execute(payload, args.out.parent)
        except Exception as error:
            import traceback
            traceback.print_exc()
            result = dict(status="worker_failed", reason=f"{type(error).__name__}: {error}")
        result["input_sha256"] = digest(payload)
        write_json(args.out, result)
        return int(result["status"] == "worker_failed")
    if args.action == "analyze":
        from .capacity_metrics import analyze
        analyze(args.input, args.out)
        return 0
    spec = StudySpec.model_validate_json(args.input.read_text(encoding="utf-8"))
    if args.astermind_root is None:
        parser.error("--astermind-root is required for catalog, plan and run")
    repository = args.astermind_root.resolve()
    directory = args.model_dir or repository / "third_party/midlayer/configs/model"
    catalog = model_catalog(directory)
    if args.action == "catalog":
        write_json(args.out, dict(models=catalog, spec=spec.model_dump(mode="json")))
        return 0
    cases = plan_cases(spec, catalog)
    metadata = dict(schema_version="orbitserve.capacity_plan.v1", spec=spec.model_dump(mode="json"),
                    catalog=catalog, astermind_root=str(repository), cases=cases,
                    source_identity=source_identity(repository, spec),
                    assumptions=["four compute satellites in a static ring; no orbital availability claim",
                                 "symmetric independent full-duplex directions",
                                 "non-KV reserve is an explicit assumption, not a measurement",
                                 "device-only timing; native kernel energy is not evaluated"])
    if args.action == "plan":
        write_json(args.out, metadata)
        return 0
    if args.action == "check":
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "plan.json", metadata)
        subprocess.run([sys.executable, "-m", "orbitserve.experiments.capacity_study", "_check",
                        str(args.out / "plan.json"), "--out", str(args.out / "capabilities.json")], check=True)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    identity = digest(metadata)
    plan_path = args.out / "plan.json"
    if plan_path.exists():
        if not args.resume or digest(json.loads(plan_path.read_text(encoding="utf-8"))) != identity:
            raise ValueError("existing study requires --resume with an identical plan")
    write_json(plan_path, metadata)
    capability_path = args.out / "capabilities.json"
    if capability_path.exists():
        capabilities = json.loads(capability_path.read_text(encoding="utf-8"))
        if capabilities.get("plan_sha256") != identity:
            raise ValueError("capability check identity differs from the study")
    else:
        subprocess.run([sys.executable, "-m", "orbitserve.experiments.capacity_study", "_check",
                        str(plan_path), "--out", str(capability_path)], check=True, timeout=spec.timeout_s)
        capabilities = json.loads(capability_path.read_text(encoding="utf-8"))
    eligible = {row["case_id"] for row in capabilities["cases"] if row["status"] == "ready"}
    runnable_cases = [case for case in cases if case["case_id"] in eligible]
    completed, consecutive_failures = 0, 0
    stopped_reason = None
    total = len(runnable_cases) * len(spec.request_rates) * len(spec.seeds)
    print(f"capability gate: {len(runnable_cases)}/{len(cases)} configurations runnable; {total} trials", flush=True)
    for case, rate, seed in itertools.product(runnable_cases, sorted(spec.request_rates), spec.seeds):
        trial_id = f"{case['case_id']}-r{rate:g}-s{seed}"
        destination = args.out / "trials" / trial_id
        payload = dict(spec=metadata["spec"], case=case, rate=rate, seed=seed,
                       astermind_root=str(repository), model=catalog.get(case["model"]),
                       source_identity_sha256=digest(metadata["source_identity"]))
        expected = digest(payload)
        path = destination / "result.json"
        if path.exists() and args.resume:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("input_sha256") != expected:
                raise ValueError(f"trial identity mismatch: {trial_id}")
            continue
        if args.limit is not None and completed >= args.limit:
            break
        print(f"[{completed + 1}/{total}] {case['model']} {case['hardware']} "
              f"{case['cards_per_satellite']} GPU/sat {case['layout']} "
              f"{case['link_bandwidth_gbps']:g}G rate={rate:g} seed={seed}", flush=True)
        destination.mkdir(parents=True, exist_ok=False)
        _isolated(payload, destination, spec.timeout_s)
        completed += 1
        outcome = json.loads(path.read_text(encoding="utf-8"))
        if outcome["status"] in ("worker_failed", "timeout", "environment_error"):
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        from .capacity_metrics import analyze
        analyze(args.out, args.out / "analysis", plots=False)
        if consecutive_failures >= spec.max_consecutive_worker_failures:
            stopped_reason = f"{consecutive_failures} consecutive worker failures; inspect execution logs"
            print(stopped_reason, flush=True)
            break
    from .capacity_metrics import analyze
    analyze(args.out, args.out / "analysis")
    write_json(args.out / "progress.json", dict(stopped_reason=stopped_reason,
                                               trials_run_this_invocation=completed))
    return int(stopped_reason is not None)


if __name__ == "__main__":
    raise SystemExit(main())
