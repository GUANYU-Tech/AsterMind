"""Compare legacy results after the additive prerequisite/report source port.

Only the named new observation fields and a null migration binding are removed.
Existing values, unknown fields and physical inputs still use the strict Native
comparator. Bound receiver scenarios have intentionally different execution and
must be tested separately, never normalized into legacy equivalence.
"""

import argparse
import copy
import json
from pathlib import Path

from tests.scripts.compare_migration import differences, normalize, verified_references


def legacy_projection(report):
    report = copy.deepcopy(report)
    report.pop("request_state_events", None)
    for request in report["requests"].values():
        for field in ("metrics", "input_ready_ns", "input_ready_ticks",
                      "parent_ready_ns", "parent_ready_ticks", "kv_ready_ns", "kv_ready_ticks"):
            request.pop(field, None)
    for cell in report["cells"].values():
        cell.pop("load_history", None)
    for migration in report["configuration"]["migrations"]:
        if migration.pop("before_request", None) is not None:
            raise ValueError("bound receiver scenarios are not legacy baselines")
    return report


def compare_folders(before, after):
    before, after = Path(before), Path(after)
    left_refs, left_inputs = verified_references(before)
    right_refs, right_inputs = verified_references(after)
    problems = list(differences(left_inputs, right_inputs, "inputs"))
    left_files = {p.name for p in before.glob("*.json")}
    right_files = {p.name for p in after.glob("*.json")}
    if left_files != right_files:
        problems.append("report file set differs")
    for name in sorted(left_files & right_files):
        left = normalize(legacy_projection(json.loads((before / name).read_text())), left_refs)
        right = normalize(legacy_projection(json.loads((after / name).read_text())), right_refs)
        found = list(differences(left, right, name))
        problems.extend(found)
        print(name, "PASS" if not found else f"FAIL ({len(found)} differences)")
    return dict(reports=len(left_files), input_sets=len(left_inputs), differences=problems)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    result = compare_folders(args.before, args.after)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["differences"]))
