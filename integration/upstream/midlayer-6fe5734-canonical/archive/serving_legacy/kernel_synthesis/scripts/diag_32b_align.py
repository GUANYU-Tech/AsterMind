#!/usr/bin/env python3
"""快速诊断 32B TP2 对齐状态（30s，不 re-predict，不扫 mctracer）。

复用已生成的 .mctracer_sync_ends_*.json cache + regen cache + nsys json。
改完 operator_records/analytic_mirror 后，re-predict 一次，然后反复用这个脚本看效果。

用法:
    python scripts/serve_predict/diag_32b_align.py [case_id] [n_first_steps]
    默认 case=ShareGPT-Qwen3-32B_TP2_n128_qps1, n=20

输出:
- best-offset（regen[i]↔nsys[i+off] 最大匹配时的 off）
- no-offset match%（真实 alignment 的 cat 匹配率）
- 前 N step 的逐位置对比（regen vs nsys，cat + name）
- cat 差异对 top（定位剩余错位源）
"""
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "serve_predict"))

from serve_predict.validate_trace_alignment import (
    load_nsys_all, load_regen_trace, _partition)
from run_full_eval import _kernel_cat


def cat(nm):
    return _kernel_cat(nm)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "ShareGPT-Qwen3-32B_TP2_n128_qps1"
    n_first = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    cd = ROOT / "serve_exp/eval_serving_sharegpt_c500" / case

    nk, iters, bounds = load_nsys_all(cd, 0)
    nby = _partition(nk, bounds, iters)
    regen = load_regen_trace(cd, rank=0)
    rby = defaultdict(list)
    for sid, nm, grid, op, blk, smem in regen:
        rby[sid].append(nm)

    # 稳态 step：跳过前 100（prefill/warmup），取 100..100+n_first
    steady = [it for it in iters[100:100 + n_first] if rby.get(it) and nby.get(it)]

    # best-offset 分布
    offs = Counter()
    nooff_match = 0; nooff_tot = 0
    mm_pairs = Counter()
    for it in steady:
        rs = [cat(n) for n in rby[it]]
        ns = [cat(k[1]) for k in nby[it]]
        best = (0, 0)
        for off in range(-40, 41):
            m = 0; t = 0
            for i in range(len(rs)):
                j = i + off
                if 0 <= j < len(ns):
                    t += 1
                    if rs[i] == ns[j]:
                        m += 1
            s = m / t if t else 0
            if s > best[1]:
                best = (off, s)
        offs[best[0]] += 1
        # no-offset
        m = min(len(rs), len(ns))
        for i in range(m):
            nooff_tot += 1
            if rs[i] == ns[i]:
                nooff_match += 1
            else:
                mm_pairs[(rs[i], ns[i])] += 1

    print(f"=== {case} (steady {len(steady)} steps) ===")
    print(f"best-offset dist: {offs.most_common(5)}")
    print(f"no-offset match:  {nooff_match}/{nooff_tot} = {100*nooff_match/nooff_tot:.1f}%")
    print(f"top mismatch pairs (regen->nsys): {mm_pairs.most_common(8)}")

    # 第一个稳态 step 的逐位置对比（前 30）
    if steady:
        it = steady[0]
        rs = rby[it]; ns = [k[1] for k in nby[it]]
        print(f"\n=== step {it} first 30 (regen={len(rs)} nsys={len(ns)}) ===")
        print(f"{'i':>3} {'match':>5} {'regen_cat':>9} {'nsys_cat':>9}  regen_name / nsys_name")
        for i in range(min(30, len(rs), len(ns))):
            rc = cat(rs[i]); nc = cat(ns[i])
            mc = "Y" if rc == nc else " "
            print(f"{i:3d} {mc:>5} {rc:>9} {nc:>9}  {rs[i][:30]:>30} | {ns[i][:30]}")


if __name__ == "__main__":
    main()
