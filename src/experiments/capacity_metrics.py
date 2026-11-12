"""Capacity statistics include unfinished requests and finite search bounds."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

from .capacity_study import StudySpec, write_json


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * q
    low = int(at)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (at - low)


def trial_metrics(records, spec, rate):
    start, end = spec.warmup_s, spec.warmup_s + spec.measurement_s
    cohort = [r for r in records if start <= r["arrival_s"] < end]

    def latencies(record):
        first, last = record.get("first_received_s"), record.get("completed_s")
        if first is None or last is None:
            return None
        return (first - record["arrival_s"],
                (last - first) / (record["output_tokens"] - 1),
                last - record["arrival_s"])

    observed = [v for r in cohort if (v := latencies(r)) is not None]
    midpoint = (start + end) / 2

    def backlog(at):
        return sum(r["arrival_s"] < at and (r.get("completed_s") is None or r["completed_s"] >= at)
                   for r in records)

    growth = (backlog(end) - backlog(midpoint)) / (spec.measurement_s / 2)
    tolerance = max(rate * spec.queue_growth_fraction, 1 / spec.measurement_s)
    enough = len(cohort) >= spec.min_measured_requests
    # Compare both halves: a low completed-latency percentile cannot hide a growing queue.
    stable = growth <= tolerance
    completions = [r for r in records if r.get("completed_s") is not None and start <= r["completed_s"] < end]
    targets = {}
    for target in spec.targets:
        def passes(record):
            value = latencies(record)
            return value is not None and value[0] <= target.ttft_s and value[1] <= target.tpot_s

        passed = sum(passes(r) for r in cohort)
        attainment = passed / len(cohort) if cohort else None
        targets[target.name] = dict(
            attainment=attainment, cohort_good_requests=passed,
            goodput_requests_per_s=sum(passes(r) for r in completions) / spec.measurement_s,
            sustainable=enough and stable and attainment is not None and attainment >= target.attainment,
        )
    return dict(
        measured_arrivals=len(cohort), completed_cohort=len(observed),
        incomplete_cohort=len(cohort) - len(observed), sufficient_population=enough,
        completed_requests_per_s=len(completions) / spec.measurement_s,
        output_tokens_per_s=sum(r["output_tokens"] for r in completions) / spec.measurement_s,
        ttft_p95_s=percentile([v[0] for v in observed], 0.95),
        tpot_p95_s=percentile([v[1] for v in observed], 0.95),
        e2e_p95_s=percentile([v[2] for v in observed], 0.95),
        backlog_start=backlog(start), backlog_midpoint=backlog(midpoint), backlog_end=backlog(end),
        backlog_growth_per_s=growth, backlog_growth_tolerance_per_s=tolerance,
        queue_stable=stable, targets=targets,
    )


def _csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capability_report(plan, capabilities, output):
    output = Path(output)
    by_id = {case["case_id"]: case for case in plan["cases"]}
    rows, models, totals = [], {}, {}
    for result in capabilities:
        case = by_id[result["case_id"]]
        status = result["status"]
        totals[status] = totals.get(status, 0) + 1
        entry = models.setdefault(case["model"], {})
        entry[status] = entry.get(status, 0) + 1
        rows.append({**{k: v for k, v in case.items() if k not in ("workload", "status")},
                     "workload": case["workload"]["name"], "status": status,
                     "reason": result.get("reason"),
                     "minimum_rank_kv_blocks": result.get("memory", {}).get("physical_block_limit")})
    _csv(output / "capabilities.csv", rows)
    write_json(output / "capability-summary.json", dict(
        configurations=len(rows), status_counts=totals, models=models,
        native_support_is_not_accuracy_validation=True))


def analyze(directory, output, *, plots=True):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    spec = StudySpec.model_validate(plan["spec"])
    capability_path = directory / "capabilities.json"
    capabilities = (json.loads(capability_path.read_text(encoding="utf-8"))["cases"]
                    if capability_path.exists() else [])
    capability = {r["case_id"]: r for r in capabilities}
    trials, flat = {}, []
    for path in sorted((directory / "trials").glob("*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        request = json.loads((path.parent / "input.json").read_text(encoding="utf-8"))
        case, rate, seed = request["case"], request["rate"], request["seed"]
        trials[case["case_id"], rate, seed] = value
        metrics = value.get("metrics", {})
        for target in spec.targets:
            flat.append({**{k: v for k, v in case.items() if k not in ("workload", "status")},
                         "workload": case["workload"]["name"], "rate": rate, "seed": seed,
                         "status": value["status"], "reason": value.get("reason"),
                         "target": target.name,
                         **{k: v for k, v in metrics.items() if k != "targets"},
                         **metrics.get("targets", {}).get(target.name, {})})
    curves = []
    for case, target in itertools.product(plan["cases"], spec.targets):
        passing, failing, missing, inconclusive, statuses = [], [], [], [], set()
        gate = capability.get(case["case_id"], {}).get("status")
        if gate is not None and gate != "ready":
            curves.append({**{k: v for k, v in case.items() if k not in ("workload", "status")},
                           "workload": case["workload"]["name"], "target": target.name,
                           "max_tested_sustainable_rps": None, "next_tested_failing_rps": None,
                           "capacity_is_lower_bound": False, "missing_rates": [], "statuses": [gate],
                           "inconclusive_rates": [],
                           "complete_load_seed_grid": False, "selection_scope": "capability_excluded"})
            continue
        for rate in sorted(spec.request_rates):
            values = [trials.get((case["case_id"], rate, seed)) for seed in spec.seeds]
            statuses.update(v["status"] for v in values if v)
            if any(v is None for v in values):
                missing.append(rate)
            elif all(v["status"] == "evaluated" for v in values):
                if not all(v["metrics"].get("sufficient_population", False) for v in values):
                    inconclusive.append(rate)
                elif all(v["metrics"]["targets"][target.name]["sustainable"] for v in values):
                    passing.append(rate)
                else:
                    failing.append(rate)
            else:
                inconclusive.append(rate)
        best = max(passing) if passing else None
        upper = min((r for r in failing if best is None or r > best), default=None)
        curves.append({**{k: v for k, v in case.items() if k not in ("workload", "status")},
                       "workload": case["workload"]["name"], "target": target.name,
                       "max_tested_sustainable_rps": best,
                       "next_tested_failing_rps": upper,
                       "capacity_is_lower_bound": best is not None and upper is None,
                       "missing_rates": missing,
                       "inconclusive_rates": inconclusive,
                       "statuses": sorted(statuses),
                       "complete_load_seed_grid": not missing,
                       "selection_scope": "exploratory_model_prediction"})
    keys = ("model", "workload", "target", "link_bandwidth_gbps", "bandwidth_accounting")
    groups = {}
    for row in curves:
        if row["max_tested_sustainable_rps"] is not None:
            groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    recommendations = []
    for key, rows in groups.items():
        best = max(r["max_tested_sustainable_rps"] for r in rows)
        near = [r for r in rows if r["max_tested_sustainable_rps"] >= best * spec.recommendation_fraction]
        count = min(r["total_cards"] for r in near)
        # Keep hardware/layout ties; price, power and hardware accuracy are not inferred.
        recommendations.append({**dict(zip(keys, key)), "best_tested_rps": best,
                                "minimum_tested_total_cards_near_best": count,
                                "candidates": [r for r in near if r["total_cards"] == count],
                                "final_hardware_recommendation": False,
                                "reason": "minimum card count near best tested capacity; calibration, coverage and cost remain explicit"})
    eligible_count = sum(not capabilities or capability.get(c["case_id"], {}).get("status") == "ready"
                         for c in plan["cases"])
    expected = eligible_count * len(spec.request_rates) * len(spec.seeds)
    evaluated = sum(v["status"] == "evaluated" for v in trials.values())
    summary = dict(schema_version="orbitserve.capacity_analysis.v1", expected_trials=expected,
                   planned_configurations=len(plan["cases"]), runnable_configurations=eligible_count,
                   excluded_configurations=len(plan["cases"]) - eligible_count,
                   recorded_trials=len(trials), evaluated_trials=evaluated,
                   complete=expected > 0 and evaluated == expected,
                   recommendations=recommendations, curves=curves,
                   limitations=plan["assumptions"])
    _csv(output / "trials.csv", flat)
    _csv(output / "curves.csv", curves)
    write_json(output / "summary.json", summary)
    if plots and groups:
        plot_curves(curves, output)
    return summary


def plot_curves(curves, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    facets = {}
    for row in curves:
        if row["max_tested_sustainable_rps"] is not None:
            facets.setdefault((row["model"], row["workload"], row["target"]), []).append(row)
    for (model, workload, target), rows in facets.items():
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        series = {}
        # Show best tested layout at each card count for each hardware/bandwidth.
        for row in rows:
            key = row["hardware"], row["link_bandwidth_gbps"], row["bandwidth_accounting"]
            points = series.setdefault(key, {})
            x = row["cards_per_satellite"]
            old = points.get(x)
            if old is None or row["max_tested_sustainable_rps"] > old["max_tested_sustainable_rps"]:
                points[x] = row
        colors = {h: f"C{i}" for i, h in enumerate(sorted({k[0] for k in series}))}
        for (hardware, bandwidth, accounting), points in sorted(series.items()):
            ordered = sorted(points.items())
            ax.plot([x for x, _ in ordered], [r["max_tested_sustainable_rps"] for _, r in ordered],
                    marker="o", linestyle="-" if bandwidth == min(k[1] for k in series) else "--",
                    color=colors[hardware], label=f"{hardware} / {bandwidth:g}G {accounting}")
            for x, row in ordered:
                if row["capacity_is_lower_bound"]:
                    ax.annotate("≥", (x, row["max_tested_sustainable_rps"]), xytext=(3, 5),
                                textcoords="offset points")
        ax.set(xlabel="GPUs per satellite (4 satellites)", ylabel="Max tested sustainable requests/s",
               title=f"{model} / {workload} / {target}\nExploratory prediction; ≥ means load ceiling not found")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.savefig(output / f"{model}-{workload}-{target}.png", dpi=160)
        fig.savefig(output / f"{model}-{workload}-{target}.svg")
        plt.close(fig)
