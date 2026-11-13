#!/usr/bin/env python3
"""Summarize scheduled resource occupancy for a real first decode step."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.serving.metrics.flash_decode_steady import scenario_hardware, source_identity
from src.kernel.kernel_base import predict
from src.serving.kernel_lowering.execution_timeline import build_execution_timeline, with_earliest_ready_resource_order
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.workload.model_config import load_model_config
from src.serving.workload.traced_flash_decode_scenario import FlashDecodeScenario

p = argparse.ArgumentParser()
p.add_argument("--scheme", choices=("dp8_ep8", "dp8_fpga"), required=True)
p.add_argument("--bw216", action="store_true")
p.add_argument("--output", required=True)
a = p.parse_args()
t = time.perf_counter()
identity = source_identity()
model = load_model_config("configs/model/deepseek-v4-pro.json")
hardware = scenario_hardware(a.bw216)
source = FlashDecodeScenario(model, hardware_id="hygon_bw1100", scheme=a.scheme, max_steps=1)
graph, state = source.step()
plan = lower_workload(graph, source.deployment, hardware)
predictions = {c.call_id: predict(c.kernel_spec, hardware[c.kernel_spec.hardware_id], "roofline") for c in plan.calls}
plan = with_earliest_ready_resource_order(plan, predictions)
timeline = build_execution_timeline(plan, predictions)
calls = {c.call_id: c for c in plan.calls}
by_rank_kind = defaultdict(float)
for interval in timeline.intervals:
    key = (interval.pool_name, interval.rank, calls[interval.call_id].kernel_spec.kind.value)
    by_rank_kind[key] += interval.end_time_s - interval.start_time_s
rows = [{"pool": pool, "rank": rank, "kind": kind, "busy_s": seconds} for (pool, rank, kind), seconds in sorted(by_rank_kind.items())]
if source_identity() != identity:
    raise RuntimeError("source changed during step")
Path(a.output).write_text(json.dumps({"scope": "first actual decode step only; sum of per-rank services is not critical-path attribution", "scheme": a.scheme, "bw216": a.bw216, "layers": model.num_hidden_layers, "timeline_s": timeline.duration_s, "cpu_wall_s": time.perf_counter()-t, "priced_calls": len(predictions), "per_rank_kind_service_s": rows, "source_identity": identity}, indent=2)+"\n")
