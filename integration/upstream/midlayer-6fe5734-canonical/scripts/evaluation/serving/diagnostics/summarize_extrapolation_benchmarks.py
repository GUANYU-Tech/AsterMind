#!/usr/bin/env python3
"""Compare selected Serving predictions with independent Event results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _measurement_index(root: Path):
    index = {}
    for path in root.glob("*/*.json"):
        payload = _load(path)
        hardware_id = payload["hardware_id"]
        family = payload["family"]
        workload = payload["workload"]
        if family == "gemm":
            if "B" in workload:
                shape = (
                    workload["B"], workload["M"],
                    workload["K"], workload["N"],
                )
            else:
                shape = (
                    workload["batch"], workload["seq_len"],
                    workload["in_features"], workload["out_features"],
                )
            key = (hardware_id, family, shape)
        else:
            key = (hardware_id, family, payload["case_id"])
        if key in index:
            raise ValueError(f"duplicate benchmark result: {key}")
        index[key] = (path, payload)
    return index


def _result_key(candidate):
    if candidate["family"] == "gemm":
        shape = candidate["serving_workload"]["shape"]
        return (
            candidate["hardware_id"], "gemm",
            (shape["b"], shape["m"], shape["k"], shape["n"]),
        )
    return (
        candidate["hardware_id"], candidate["family"],
        candidate["case_id"],
    )


def _label(candidate):
    shape = candidate["serving_workload"]["shape"]
    if candidate["family"] == "gemm":
        return f"GEMM M{shape['m']} K{shape['k']} N{shape['n']}"
    if candidate["family"] == "flash_attn":
        return (
            f"FA B{shape['batch']} Q{shape['q_len']} "
            f"KV{shape['kv_len']}")
    return (
        f"{candidate['serving_workload']['variant']} "
        f"rows{shape['rows']} hidden{shape['hidden']}")


def _format_differences(differences):
    return ", ".join(
        f"{name}: {values['dataset']} -> {values['serving']}"
        for name, values in differences.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = _load(args.candidates)["candidates"]
    measurements = _measurement_index(args.measurements)
    rows = []
    for candidate in candidates:
        key = _result_key(candidate)
        if key not in measurements:
            raise ValueError(f"missing measurement for {key}")
        path, measurement = measurements[key]
        measured_ns = measurement["measurement"]["statistics"]["median_ns"]
        predicted_ns = candidate["predicted_ns_per_occurrence"]
        signed_error = (predicted_ns - measured_ns) / measured_ns * 100.0
        predicted_total_s = candidate["serving_predicted_total_s"]
        predicted_case_total_s = (
            predicted_total_s / candidate["serving_predicted_time_share"])
        measured_total_s = (
            measured_ns * candidate["logical_occurrences"] / 1e9)
        rows.append({
            "case_id": candidate["case_id"],
            "hardware_id": candidate["hardware_id"],
            "family": candidate["family"],
            "label": _label(candidate),
            "serving_time_share_pct": (
                candidate["serving_predicted_time_share"] * 100.0),
            "logical_occurrences": candidate["logical_occurrences"],
            "predicted_ns": predicted_ns,
            "measured_median_ns": measured_ns,
            "measured_p10_ns": measurement[
                "measurement"]["statistics"]["p10_ns"],
            "measured_p90_ns": measurement[
                "measurement"]["statistics"]["p90_ns"],
            "signed_error_pct": signed_error,
            "ape_pct": abs(signed_error),
            "replacement_case_total_delta_pct": (
                (measured_total_s - predicted_total_s)
                / predicted_case_total_s * 100.0),
            "dataset_status": candidate["dataset_coverage"]["status"],
            "nearest_dataset_case": candidate[
                "nearest_dataset_case"]["case_id"],
            "nearest_dataset_scalar_field_differences": candidate[
                "nearest_dataset_case"]["scalar_field_differences"],
            "measurement_path": str(path.resolve()),
        })
    if len(rows) != 10:
        raise AssertionError(f"expected 10 benchmark rows, got {len(rows)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_comparison.json"
    json_path.write_text(json.dumps({
        "schema_version": 1,
        "error_definition": "100 * (prediction - measurement) / measurement",
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output_dir / "benchmark_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Serving kernel extrapolation benchmark",
        "",
        "Error is `100 × (prediction − measurement) / measurement`; positive means overprediction.",
        "",
        "| Hardware | Candidate | Serving share | Predicted | Measured | Error | Nearest Dataset difference |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['hardware_id']} | {row['label']} | "
            f"{row['serving_time_share_pct']:.3f}% | "
            f"{row['predicted_ns'] / 1000:.2f} us | "
            f"{row['measured_median_ns'] / 1000:.2f} us | "
            f"{row['signed_error_pct']:+.2f}% | "
            f"{row['nearest_dataset_case']}: "
            f"{_format_differences(row['nearest_dataset_scalar_field_differences'])} |")
    lines.extend((
        "",
        "`replacement_case_total_delta_pct` in the JSON/CSV is the change in the full predicted device-time total if only that selected occurrence class is replaced by its measured median. It is a bounded contribution estimate, not a claim that one sampled shape represents its whole family.",
    ))
    (output_dir / "benchmark_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt
    markers = {"gemm": "o", "flash_attn": "s", "vector": "^"}
    colors = {
        "nvidia_a800": "#1f77b4",
        "metax_c500": "#d62728",
        "ascend_910b": "#2ca02c",
    }
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    values = []
    for row in rows:
        x_value = row["measured_median_ns"] / 1000.0
        y_value = row["predicted_ns"] / 1000.0
        values.extend((x_value, y_value))
        axis.scatter(
            x_value, y_value, s=70,
            marker=markers[row["family"]],
            color=colors[row["hardware_id"]],
            edgecolor="black", linewidth=0.5,
        )
        axis.annotate(
            row["case_id"].replace("ascend_", "").replace("nvidia_", ""),
            (x_value, y_value), xytext=(4, 4), textcoords="offset points",
            fontsize=7,
        )
    lower = min(values) * 0.8
    upper = max(values) * 1.25
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Measured public-op median (us)")
    axis.set_ylabel("Serving Forward prediction (us)")
    axis.set_title("Serving-only kernel extrapolation\nQwen3-8B TP1 QPS64")
    axis.grid(True, which="both", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_vs_measurement.png", dpi=130)
    plt.close(figure)
    print(output_dir)


if __name__ == "__main__":
    main()
