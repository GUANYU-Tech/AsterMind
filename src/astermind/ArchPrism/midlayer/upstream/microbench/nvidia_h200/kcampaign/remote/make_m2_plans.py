"""Machine-2 plans for the 2026-09-27 H200 (runs on h200-tmp, stdlib only).

The 8-GPU host of 2026-09-27 is a different machine (new GPU UUIDs, driver 580.95.05) than the
4-GPU host of 2026-09-26, so no case may mix windows from both. For each listed plan this writes
`<base>_m2__v030` holding the cases that do not yet have their three regular windows
(window_0[123]) in the original run directory; those cases are measured afresh, all three
windows on machine 2, in the new run directory. Cases complete on machine 1 stay there.
`--retry plan:case_file` writes `<base>_retrym2__v030` with exactly the listed cases (cases that
failed assembly on machine 1 are remeasured whole on machine 2). `--whole plan` copies a plan's
cases unchanged as `<base>_m2__v030`. Each file carries `machine_split_of` provenance.

usage: make_m2_plans.py [--pending p ...] [--retry p:file ...] [--whole p ...] [--write]
"""
import argparse
import glob
import hashlib
import json
import os

K = "/mnt/public/yangxinhao/h200_microbench/kcampaign"
ap = argparse.ArgumentParser()
ap.add_argument("--pending", nargs="*", default=[])
ap.add_argument("--retry", nargs="*", default=[])
ap.add_argument("--whole", nargs="*", default=[])
ap.add_argument("--write", action="store_true")
a = ap.parse_args()


def load(p):
    path = f"{K}/plans/{p}.json"
    return json.load(open(path)), hashlib.sha256(open(path, "rb").read()).hexdigest()


def complete_cases(p):
    cases = [f.split("/")[-2]
             for f in glob.glob(f"{K}/runs/{p}/cases/*/window_0[123].launch.json")]
    return {c for c in set(cases) if cases.count(c) == 3}


def emit(src, name, cases, kind, extra):
    doc, sha = load(src)
    if not src.endswith("__v030"):
        raise SystemExit(f"{src}: only __v030 plans")
    out = {key: value for key, value in doc.items() if key != "plans"}
    out["case_count"] = len(cases)
    out["machine_split_of"] = {"plan": src, "plan_sha256": sha, "kind": kind,
                               "machine": "h200 8-GPU host of 2026-09-27", **extra}
    out["plans"] = cases
    path = f"{K}/plans/{name}.json"
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    print(f"{name}: {len(cases)} cases ({3 * len(cases)} windows) [{kind}]")
    if a.write:
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")


for p in a.pending:
    doc, _ = load(p)
    done = complete_cases(p)
    todo = [c for c in doc["plans"] if c["case_id"] not in done]
    if todo:
        emit(p, p[:-len("__v030")] + "_m2__v030", todo, "pending",
             {"cases_complete_on_machine1": len(doc["plans"]) - len(todo)})
for spec in a.retry:
    p, case_file = spec.split(":", 1)
    doc, _ = load(p)
    wanted = [line.strip() for line in open(case_file) if line.strip()]
    by_id = {c["case_id"]: c for c in doc["plans"]}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise SystemExit(f"{p}: retry cases not in plan: {missing[:5]}")
    emit(p, p[:-len("__v030")] + "_retrym2__v030", [by_id[w] for w in wanted], "retry",
         {"case_list": os.path.basename(case_file)})
for p in a.whole:
    doc, _ = load(p)
    emit(p, p[:-len("__v030")] + "_m2__v030", doc["plans"], "whole", {})
