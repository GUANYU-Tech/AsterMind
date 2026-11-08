#!/usr/bin/env python3
"""Drive the UniNDP HBM-PIM simulator over a controlled shape sweep.

The simulator is the HBM-PIM ground-truth instrument (the role ncu plays for
A800).  This harness only *measures*: it submits one primitive per shape and
records the cycle-accurate latency the simulator reports, so a mechanism can be
isolated by varying one geometric variable at a time.  It never reads the
simulator's internal timing algorithm.

Usage:
  python3 microbench/samsung_hbm_pim/unindp_probe.py <plan.json> <out.csv> [jobs]

`plan.json` is a list of {"name", "workload", "sizes"} records, where `sizes`
is the UniNDP `-S` 4-tuple.  Invocation flags mirror the canonical dataset
pipeline (configs/dataset/samsung_hbm_pim.yaml): arch hbm-pim, topk 1,
quicksearch on.
"""

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
UNINDP_ROOT = os.environ.get(
    "UNINDP_ROOT", "/share_data/yangxinhao/chip_modeling/UniNDP")
# compile.py resolves -WS against its own cwd, so the simulator scratch is
# addressed relative to the UniNDP checkout but lands in this repository's
# profiler_output/, never inside the external checkout.
WORKSPACE = os.path.join(
    REPOSITORY_ROOT, "profiler_output", "samsung_hbm_pim", "unindp_probe_ws")
WORKSPACE_REL = os.path.relpath(WORKSPACE, UNINDP_ROOT)
SIM_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", **os.environ}


def run_one(record):
    name = record["name"]
    workload = record["workload"]
    sizes = record["sizes"]
    outdir = f"out_{name}"
    cmd = ["python3", "compile.py", "-A", "hbm-pim", "-W", workload,
           "-N", name, "-S", *[str(s) for s in sizes],
           "-O", outdir, "-WS", WORKSPACE_REL, "-K", "1", "-Q"]
    proc = subprocess.run(cmd, cwd=UNINDP_ROOT, capture_output=True,
                          text=True, timeout=3600, env=SIM_ENV)
    csv_path = os.path.join(UNINDP_ROOT, WORKSPACE_REL, outdir, "csv",
                            f"_{name}.csv")
    if proc.returncode != 0 or not os.path.exists(csv_path):
        return {"name": name, "workload": workload, "sizes": sizes,
                "latency_ns": "", "error": proc.stderr[-200:].replace("\n", " ")}
    with open(csv_path) as stream:
        row = next(csv.reader(stream))
    return {"name": name, "workload": workload, "sizes": sizes,
            "latency_ns": float(row[2]), "inst_num": row[3],
            "pu_dram_num": row[4], "host_dram_num": row[5],
            "row_change_num": row[6], "error": ""}


def main():
    plan_path, out_path = sys.argv[1], sys.argv[2]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    with open(plan_path) as stream:
        plan = json.load(stream)
    os.makedirs(WORKSPACE, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(run_one, plan))
    fields = ["name", "workload", "sizes", "latency_ns", "inst_num",
              "pu_dram_num", "host_dram_num", "row_change_num", "error"]
    with open(out_path, "w", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in fields})
    failures = [r for r in results if r["error"]]
    print(f"{len(results)} probes, {len(failures)} failed -> {out_path}")


if __name__ == "__main__":
    main()
