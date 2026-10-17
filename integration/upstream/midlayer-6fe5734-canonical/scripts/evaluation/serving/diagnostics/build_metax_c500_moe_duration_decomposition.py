#!/usr/bin/env python3
"""Build the C500 TP2/EP2 decode MoE same-launch duration decomposition.

The predicted side is a fresh independent-eager selected-step replay at the
current working tree; the measured side is the retained mcTracer device
timeline of the same structurally validated injected step.  Measured
durations are evaluation evidence only: no value here selects, fits or
corrects a Forward parameter.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SCHEMA_VERSION = 1
REPORT_KIND = "c500_decode_moe_same_launch_duration_decomposition"

# One measured mcTracer kernel group maps onto one or more predicted call-id
# patterns.  Every pair below is the same public operation at the same public
# launch extent; pairs that are not launch-comparable are recorded separately.
_MOE_REGION = "moe"
_CLASSES = (
    ("w1_grouped_gemm", _MOE_REGION, "fused_moe_kernel", (3192, 1, 1),
     ("moe:ep-rank-{rank}:3",)),
    ("w2_grouped_gemm", _MOE_REGION, "fused_moe_kernel", (4256, 1, 1),
     ("moe:ep-rank-{rank}:5",)),
    ("swiglu_act_and_mul", _MOE_REGION, "act_and_mul_kernel", (280, 1, 1),
     ("moe:ep-rank-{rank}:4",)),
    ("moe_sum_reduce", _MOE_REGION, "thrust::pair", (35, 1, 1),
     ("moe:ep-rank-{rank}:6",)),
    ("routing_align_block_size", _MOE_REGION,
     "moe_align_block_size_opt_128_expert_kernel", (1, 1, 1),
     ("moe:ep-rank-{rank}:0", "moe:ep-rank-{rank}:1",
      "moe:ep-rank-{rank}:2")),
    ("router_topk_softmax", _MOE_REGION, "topkGatingSoftmaxDecode", (35, 1, 1),
     ("router:tp-rank-{rank}:1",)),
    ("router_gate_gemm", _MOE_REGION, "sb_0_2", (2, 2, 8),
     ("router:tp-rank-{rank}:0",)),
    ("ep_and_tp_all_reduce", _MOE_REGION,
     "mcclKernel_AllReduce_FC2_Sum_mccl_bfloat16", (16, 1, 1),
     ("o-proj:tp-all-reduce", "moe:ep-output-all-reduce",
      "token-embedding:tp-all-reduce")),
)


def _measured_groups(analysis: dict, rank: int) -> list[dict]:
    for entry in analysis["per_rank"]:
        if entry["rank"] == rank:
            return entry["kernel_groups"]
    raise KeyError(f"mcTracer analysis has no rank {rank}")


def _measured_class(groups, symbol_fragment, grid) -> dict:
    matches = [
        group for group in groups
        if symbol_fragment in group["name"]
        and tuple(group["launch"]["grid"]) == tuple(grid)
    ]
    if not matches:
        raise LookupError(
            f"no measured kernel group for {symbol_fragment!r} at grid {grid}")
    return {
        "additive_duration_ns": sum(m["additive_duration_ns"] for m in matches),
        "calls": sum(m["count"] for m in matches),
        "symbols": sorted({m["name"] for m in matches}),
    }


def _predicted_ranks(contributions) -> set:
    return {contribution["rank"] for contribution in contributions}


def _predicted_class(contributions, rank, suffixes) -> dict:
    total = 0.0
    calls = 0
    for contribution in contributions:
        if contribution["rank"] != rank:
            continue
        call_id = contribution["call_id"]
        for suffix in suffixes:
            if call_id.endswith(suffix.format(rank=rank)):
                total += contribution["predicted_additive_s"]
                calls += contribution["occurrence_count"]
                break
    if calls == 0:
        raise LookupError(f"no predicted contribution for {suffixes}")

    return {"additive_duration_ns": total * 1e9, "calls": calls}


def build_report(replay: dict, analysis: dict, rank: int) -> dict:
    comparison = replay["step_comparison"][0]
    contributions = comparison["predicted_kernel_contributions"]
    groups = _measured_groups(analysis, rank)
    measured_total_ns = sum(g["additive_duration_ns"] for g in groups)
    predicted_total_ns = sum(
        c["predicted_additive_s"] for c in contributions
        if c["rank"] == rank) * 1e9
    rows = []
    for name, region, symbol, grid, suffixes in _CLASSES:
        measured = _measured_class(groups, symbol, grid)
        # The lowered TP/EP collective is a single cross-rank operation that
        # the workload attributes to one rank; compare it against that rank's
        # contribution for every measured rank timeline.
        predicted_rank = rank
        if name == "ep_and_tp_all_reduce":
            predicted_rank = min(_predicted_ranks(contributions))
        predicted = _predicted_class(contributions, predicted_rank, suffixes)
        gap = measured["additive_duration_ns"] - predicted["additive_duration_ns"]
        rows.append({
            "kernel_class": name,
            "region": region,
            "measured_symbols": measured["symbols"],
            "measured_calls": measured["calls"],
            "predicted_calls": predicted["calls"],
            "measured_ms": measured["additive_duration_ns"] / 1e6,
            "predicted_ms": predicted["additive_duration_ns"] / 1e6,
            "measured_per_call_us": (
                measured["additive_duration_ns"] / measured["calls"] / 1e3),
            "predicted_per_call_us": (
                predicted["additive_duration_ns"] / predicted["calls"] / 1e3),
            "signed_gap_ms": gap / 1e6,
            "predicted_over_measured": (
                predicted["additive_duration_ns"]
                / measured["additive_duration_ns"]),
        })
    moe_measured = sum(r["measured_ms"] for r in rows)
    moe_predicted = sum(r["predicted_ms"] for r in rows)
    moe_gap = moe_measured - moe_predicted
    for row in rows:
        row["share_of_moe_region_gap"] = (
            row["signed_gap_ms"] / moe_gap if moe_gap else None)
    rows.sort(key=lambda row: -abs(row["signed_gap_ms"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "cell_id": replay["cell_id"],
        "rank": rank,
        "sign_convention": (
            "signed_gap_ms = measured - predicted; positive means Forward "
            "underpredicts"),
        "evidence_level": "evaluation_only_measured_side",
        "step": {
            "canonical_step_id": replay["step_id_mapping"][
                "canonical_workload_step_id"],
            "injected_step_id": replay["step_id_mapping"][
                "injected_trace_step_id"],
            "phase": comparison["phase"],
            "logical_tokens": 35,
            "graph_capture_bucket_tokens": 40,
            "replay_graph_mode": "NONE",
        },
        "step_closure": {
            "measured_device_union_ms": measured_total_ns / 1e6,
            "predicted_rank_additive_ms": predicted_total_ns / 1e6,
            "signed_gap_ms": (measured_total_ns - predicted_total_ns) / 1e6,
        },
        "moe_region": {
            "measured_ms": moe_measured,
            "predicted_ms": moe_predicted,
            "signed_gap_ms": moe_gap,
        },
        "classes": rows,
        "zero_fit_statement": (
            "Measured durations are evaluation evidence only; no value in "
            "this report enters Forward, HWSpec, launch selection, a LUT or "
            "a residual correction."),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--injection-replay", type=Path, required=True)
    parser.add_argument("--mctracer-analysis", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    replay = json.loads(arguments.injection_replay.read_text(encoding="utf-8"))
    analysis = json.loads(
        arguments.mctracer_analysis.read_text(encoding="utf-8"))
    report = build_report(replay, analysis, arguments.rank)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        arguments.output_dir / f"moe_region_decomposition_rank{arguments.rank}.json")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = (
        arguments.output_dir / f"moe_region_decomposition_rank{arguments.rank}.csv")
    fields = (
        "kernel_class", "region", "measured_calls", "predicted_calls",
        "measured_ms", "predicted_ms", "measured_per_call_us",
        "predicted_per_call_us", "signed_gap_ms", "predicted_over_measured",
        "share_of_moe_region_gap")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["classes"]:
            writer.writerow(row)
    print(json.dumps({
        "json": str(json_path), "csv": str(csv_path),
        "moe_region_signed_gap_ms": report["moe_region"]["signed_gap_ms"],
        "step_signed_gap_ms": report["step_closure"]["signed_gap_ms"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
