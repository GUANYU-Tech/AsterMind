#!/usr/bin/env python3
"""Select high-contribution Serving kernels for extrapolation checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.metrics.diagnostics import (  # noqa: E402
    metax_c500_kernel_coverage as coverage,
)


HARDWARE_IDS = ("nvidia_a800", "metax_c500", "ascend_910b")


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"coverage report must be a mapping: {path}")
    return payload


def _maximum(rows, predicate):
    matches = [row for row in rows if predicate(row)]
    if not matches:
        raise ValueError("candidate selection matched no logical operation")
    return max(matches, key=lambda row: row["predicted_device_time_s"])


def _select(report):
    rows = report["logical_ops"]
    selected = [
        _maximum(rows, lambda row: (
            row["canonical_mapping"]["status"] == "missing"
            and row["op_key"]["workload"]["kind"] == "linear"
            and row["op_key"]["workload"]["shape"]["m"] == 1
            and row["op_key"]["workload"]["shape"]["n"] == 24576)),
        _maximum(rows, lambda row: (
            row["canonical_mapping"]["status"] == "missing"
            and row["op_key"]["workload"]["kind"] == "linear"
            and row["op_key"]["workload"]["shape"]["m"] > 1000
            and row["op_key"]["workload"]["shape"]["n"] == 24576)),
        _maximum(rows, lambda row: (
            row["canonical_mapping"]["status"] == "missing"
            and row["op_key"]["workload"]["kind"] == "flash_attn")),
    ]
    if report["hardware_id"] == "ascend_910b":
        selected.append(_maximum(rows, lambda row: (
            row["canonical_mapping"]["status"] == "missing"
            and row["op_key"]["workload"]["kind"] == "norm")))
    return selected


def _nearest_case(query, candidates):
    scalar_query = {
        name: value for name, value in query.items()
        if isinstance(value, (int, float, str, bool))
    }
    ranked = []
    for candidate in candidates:
        attributes = candidate["attributes"]
        distance = 0.0
        mismatches = {}
        for name, value in scalar_query.items():
            if name not in attributes:
                distance += 100.0
                mismatches[name] = {"serving": value, "dataset": None}
                continue
            other = attributes[name]
            if isinstance(value, bool) or isinstance(value, str):
                if other != value:
                    distance += 100.0
                    mismatches[name] = {"serving": value, "dataset": other}
            elif isinstance(other, (int, float)) and other > 0 and value > 0:
                difference = abs(math.log2(float(value) / float(other)))
                distance += difference
                if other != value:
                    mismatches[name] = {"serving": value, "dataset": other}
            elif other != value:
                distance += 100.0
                mismatches[name] = {"serving": value, "dataset": other}
        ranked.append((distance, candidate["case_id"], mismatches))
    if not ranked:
        return None
    distance, case_id, mismatches = min(ranked, key=lambda row: row[:2])
    return {"case_id": case_id, "distance": distance,
            "scalar_field_differences": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for value in args.coverage:
        hardware_id, separator, raw_path = value.partition("=")
        if not separator or hardware_id not in HARDWARE_IDS:
            raise ValueError(f"invalid --coverage value: {value!r}")
        paths[hardware_id] = Path(raw_path).resolve()
    if set(paths) != set(HARDWARE_IDS):
        raise ValueError("coverage inputs must contain all three hardware IDs")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for hardware_id in HARDWARE_IDS:
        report = _load(paths[hardware_id])
        if report["hardware_id"] != hardware_id:
            raise ValueError(f"coverage hardware mismatch: {paths[hardware_id]}")
        physical = {
            row["physical_key_id"]: row
            for row in report["physical_kernels"]
        }
        canonical = coverage.load_canonical_timing_index(
            REPOSITORY_ROOT, hardware_id)
        for ordinal, row in enumerate(_select(report)):
            workload = row["op_key"]["workload"]
            family, reason, query = coverage._candidate_query(row["op_key"])
            if family is None or query is None:
                raise ValueError(f"selected candidate is unmapped: {reason}")
            physical_rows = [physical[value] for value in row["physical_key_ids"]]
            case_id = f"{hardware_id}-{ordinal:02d}-{family}"
            result = {
                "case_id": case_id,
                "hardware_id": hardware_id,
                "family": family,
                "serving_case_id": report["cases"][0]["cell_id"],
                "serving_workload": workload,
                "serving_predicted_total_s": row["predicted_device_time_s"],
                "serving_predicted_time_share": row[
                    "global_predicted_device_time_share"],
                "logical_occurrences": row["logical_occurrences"],
                "predicted_ns_per_occurrence": (
                    row["predicted_device_time_s"] * 1e9
                    / row["logical_occurrences"]),
                "dataset_coverage": row["canonical_mapping"],
                "nearest_dataset_case": _nearest_case(
                    query, canonical[family]),
                "physical_kernels": physical_rows,
            }
            if family == "flash_attn":
                sidecar = output_dir / f"{case_id}-workload.json"
                sidecar.write_text(json.dumps({
                    "q_lens": workload["attributes"]["q_lens"],
                    "kv_lens": workload["attributes"]["kv_lens"],
                }, indent=2) + "\n", encoding="utf-8")
                result["workload_sidecar"] = str(sidecar)
            results.append(result)
    if len(results) != 10:
        raise AssertionError(f"expected 10 candidates, got {len(results)}")
    output = output_dir / "candidates.json"
    output.write_text(json.dumps({
        "schema_version": 1,
        "selection_policy": (
            "per hardware: top uncovered gate-up GEMM at M=1, top uncovered "
            "gate-up GEMM at M>1000, top uncovered FlashAttention; plus top "
            "uncovered 910B Norm"),
        "candidate_count": len(results),
        "candidates": results,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
