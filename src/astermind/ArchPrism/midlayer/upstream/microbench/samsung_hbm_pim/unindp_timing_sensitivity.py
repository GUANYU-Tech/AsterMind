#!/usr/bin/env python3
"""Attribute each HBM-PIM bank-stream cost to a named DDR command timing.

The bank-stream model charges a column access, a bus turnaround, a row cycle
and a pipeline entry.  This harness proves which vendor timing each of those
is made of, by perturbing one timing at a time in an isolated copy of the
UniNDP architecture configuration and re-measuring the *differential* cost of
one column, one operand-buffer boundary and one bank-row boundary.  A claim
survives only if the measured step moves by the predicted multiple of the
perturbation.

It never reads the simulator's scheduling algorithm; it only submits shapes
and reads back latencies, the way a microbenchmark treats a profiler.

Usage:
  UNINDP_ROOT=<path> python3 microbench/samsung_hbm_pim/unindp_timing_sensitivity.py [out.txt]

The isolated copy is built under a scratch directory; the real UniNDP
checkout is never modified.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

if "UNINDP_ROOT" not in os.environ:
    raise SystemExit("UNINDP_ROOT must name the external UniNDP checkout")
UNINDP_ROOT = os.environ["UNINDP_ROOT"]
TIMINGS = ("tCCDL", "tCCDS", "tRCDWR", "tRCDRD", "tRP", "tWR", "tRTPL",
           "tWTRL", "tRTRS", "RL", "WL", "BL")
DELTA = 10
# One work column is `lanes x units` = (co_w/data_pr) x (ch x de x de_pu)
# elements; the probes below sit either side of a plain column (c=7,8),
# an operand-buffer boundary (c=9, buffer holds 8 columns) and a bank-row
# boundary (c=33, a row holds 32 columns).
COLUMN_ELEMENTS = 8192
PROBE_COLUMNS = (7, 8, 9, 10, 32, 33, 34)


def _build_scratch():
    scratch = tempfile.mkdtemp(prefix="unindp_sensitivity_")
    subprocess.run(
        f"tar -cf - --exclude=.git --exclude='w[0-9]*' --exclude='ws_*' "
        f"--exclude='out_*' . | (cd {scratch} && tar -xf -)",
        cwd=UNINDP_ROOT, shell=True, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return scratch


def _write_config(scratch, pristine, timing, delta):
    lines = []
    for line in pristine.split("\n"):
        match = (re.match(r"^(%s):\s*(-?\d+)" % re.escape(timing), line)
                 if timing else None)
        lines.append("%s: %d" % (timing, int(match.group(2)) + delta)
                     if match else line)
    with open(os.path.join(scratch, "config", "hbm-pim.yaml"), "w") as stream:
        stream.write("\n".join(lines))


def _simulate(scratch, workload, elements, tag):
    name = f"sens_{tag}"
    subprocess.run(
        ["python3", "compile.py", "-A", "hbm-pim", "-W", workload,
         "-N", name, "-S", str(elements), "1", "1", "1",
         "-O", f"out_{tag}", "-WS", "sens_ws", "-K", "1", "-Q"],
        cwd=scratch, capture_output=True, text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"})
    path = os.path.join(scratch, "sens_ws", f"out_{tag}", "csv", f"_{name}.csv")
    if not os.path.exists(path):
        return None
    return float(open(path).read().split(",")[2])


def _steps(scratch, pristine, workload, timing, delta):
    _write_config(scratch, pristine, timing, delta)
    probes = [(workload, COLUMN_ELEMENTS * c, f"{workload[:2]}{c}")
              for c in PROBE_COLUMNS]
    with ThreadPoolExecutor(len(probes)) as pool:
        values = dict(zip(
            PROBE_COLUMNS,
            pool.map(lambda p: _simulate(scratch, *p), probes)))
    return {
        "plain column": values[8] - values[7],
        "buffer boundary": values[9] - values[8],
        "row boundary": values[33] - values[32],
        "one work column (c=1)": _simulate(
            scratch, workload, COLUMN_ELEMENTS, f"{workload[:2]}1"),
    }


def main():
    out = open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stdout
    scratch = _build_scratch()
    try:
        config = os.path.join(scratch, "config", "hbm-pim.yaml")
        pristine = open(config).read()
        for workload in ("elewise", "layernorm", "softmax", "mm"):
            if workload == "mm":
                continue
            print(f"\n=== {workload} (+{DELTA} on one timing at a time) ===",
                  file=out)
            reference = _steps(scratch, pristine, workload, None, 0)
            columns = list(reference)
            print("%-10s %s" % ("timing", " ".join(
                "%22s" % c for c in columns)), file=out)
            print("%-10s %s" % ("(baseline)", " ".join(
                "%22.1f" % reference[c] for c in columns)), file=out)
            for timing in TIMINGS:
                moved = _steps(scratch, pristine, workload, timing, DELTA)
                print("%-10s %s" % (f"{timing}+{DELTA}", " ".join(
                    "%+22.1f" % (moved[c] - reference[c]) for c in columns)),
                    file=out)
        _write_config(scratch, pristine, None, 0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
