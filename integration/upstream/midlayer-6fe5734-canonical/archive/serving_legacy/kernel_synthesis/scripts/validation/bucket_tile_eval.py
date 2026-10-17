"""Legacy per-kernel-category bucket eval over synthesized serving traces.

For each case:
  - load regen trace cache -> count synthetic emit invocations per kernel_name
  - load distilled op_profile.json -> count real msprof invocations per op_name
  - run check_tile_alignment -> per-kernel match/grid_mismatch/name_mismatch

Bucket both sides by kernel category (GEMM-dense / MoE-GEMM / FA / Norm / Rotary
/ KV-cache / SwiGlu / MoE-routing / other). Aggregate across 21 cases:
  - per category: synthetic count, real count, 1-to-1? (synthetic == real)
  - per category: matched vs mismatched emits (name+grid)
"""
import sys, json, io, re, gzip, pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

HW = "configs/hardware/huawei_910b.yaml"
EXP = Path("serve_exp/serve_exp_910b/eval_serving_sharegpt")

# ---- category mapping ----
def synth_cat(kn):
    k = kn.lower()
    if "groupedmatmul" in k: return "MoE-GEMM (grouped)"
    if k.startswith("aclnnmatmul"): return "GEMM (dense aclnnMatmul)"
    if "fusedinferattention" in k: return "FA"
    if "rmsnorm" in k or "addrmsnormbias" in k: return "Norm"
    if "rope" in k or "rotary" in k: return "Rotary"
    if "reshapeandcache" in k or "reshape_and_cache" in k: return "KV-cache"
    if "swiglu" in k or "act_and_mul" in k: return "Act (SwiGlu)"
    if "moegating" in k or "moeinitrouting" in k or "moetokenunpermute" in k or "moecomputeexpert" in k or "moefinalize" in k: return "MoE-routing"
    if "nccl" in k or "allreduce" in k: return "Comm (NCCL)"
    return "other-glue"

def real_cat(op):
    k = op.lower()
    if "groupedmatmul" in k or "batchmatmul" in k: return "MoE-GEMM (grouped)"
    if "fusedinferattention" in k: return "FA"
    if k.startswith("aclnnmatmul") or ("matmul" in k and "grouped" not in k and "batch" not in k): return "GEMM (dense aclnnMatmul)"
    if "rmsnorm" in k or "addrmsnormbias" in k: return "Norm"
    if "rope" in k or "rotary" in k: return "Rotary"
    if "reshapeandcache" in k or "reshape_and_cache" in k: return "KV-cache"
    if "swiglu" in k or "act_and_mul" in k: return "Act (SwiGlu)"
    if "moegating" in k or "moeinitrouting" in k or "moetokenunpermute" in k or "moecomputeexpert" in k or "moefinalize" in k: return "MoE-routing"
    if "nccl" in k or "allreduce" in k: return "Comm (NCCL)"
    return "other-glue"   # CANN per-launch small aclnn ops (cast/fill/slice/...)

def case_dict(cd):
    meta = json.load(open(cd / "meta.json"))
    cid = cd.name
    m = re.search(r"_TP(\d+)", cid); tp = int(meta.get("tensor_parallel_size") or meta.get("tp") or m.group(1))
    m = re.search(r"EP(\d+)", cid); ep = int(meta.get("expert_parallel_size") or meta.get("ep") or (m.group(1) if m and m.group(1) else 1))
    mq = re.search(r"qps(\d+)", cid); qps = int(meta.get("qps") or mq.group(1))
    return {"case_id": cid, "model": meta["model"], "tensor_parallel_size": tp,
            "expert_parallel_size": ep, "num_requests": int(meta.get("num_requests",128)),
            "qps": qps, "gpu_ids": list(range(tp)), "hw_config": HW,
            "workload": meta.get("workload","sharegpt"),
            "dataset_path": meta.get("dataset_path","/mnt/gpfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"),
            "sample_seed": int(meta.get("sample_seed", meta.get("seed",42))),
            "poisson_seed": int(meta.get("poisson_seed", meta.get("seed",42))),
            "timestamp": meta.get("timestamp","")}

def load_traces(cd, case):
    import scripts.serve_predict.build_eval_dataset as b
    from src.serving.workload.model_config import load_model_config
    mc = load_model_config(case["model"])
    tp = case["tensor_parallel_size"]; ep = case.get("expert_parallel_size",1)
    return b._regenerate_trace_with_step_id(cd, mc, tp, ep, HW)

def main():
    # per-category aggregates: synthetic count, real count, matched, grid_mismatch, name_mismatch
    agg_synth = defaultdict(int)
    agg_real  = defaultdict(int)
    agg_match = defaultdict(int)
    agg_grid  = defaultdict(int)
    agg_name  = defaultdict(int)
    per_case_rows = []

    for cd in sorted(EXP.glob("ShareGPT-*")):
        cid = cd.name
        try:
            case = case_dict(cd)
            traces = load_traces(cd, case)
        except Exception as e:
            print(f"[skip] {cid}: {e}", file=sys.stderr); continue
        if not traces:
            print(f"[skip] {cid}: no traces", file=sys.stderr); continue

        # count synthetic invocations per category
        synth = defaultdict(int)
        for rid, tr in traces.items():
            if not tr: continue
            for e in tr:
                kn = (e.get("kernel_name") or e.get("name") or "").strip()
                if kn: synth[synth_cat(kn)] += 1

        # count real invocations per category from op_profile.json
        real = defaultdict(int)
        opf = cd / "op_profile.json"
        if opf.exists():
            prof = json.load(open(opf))
            for key, v in prof.items():
                nm = v["op_name"]; cnt = v.get("count",0)
                real[real_cat(nm)] += cnt

        # run checker, capture per-kernel lines
        from src.serve_predict.check_910b_tile_alignment import check_tile_alignment
        buf = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(buf):
            check_tile_alignment(case, cd, traces, verbose=True)
        out = buf.getvalue()

        # parse per-kernel verdict lines (non-GEMM path)
        # [tile_match]    case=... kernel='X' actual_op='Y' count=N
        # [grid_mismatch] case=... kernel='X' count=N ...
        # [name_mismatch] case=... kernel='X' count=N reason=...
        mk = defaultdict(int); gk = defaultdict(int); nk = defaultdict(int)
        for line in out.splitlines():
            if line.startswith("[tile_match]    "):
                m = re.search(r"kernel='([^']+)'[^>]*count=(\d+)", line)
                if m: mk[synth_cat(m.group(1))] += int(m.group(2))
            elif line.startswith("[grid_mismatch] "):
                m = re.search(r"kernel='([^']+)'[^>]*count=(\d+)", line)
                if m: gk[synth_cat(m.group(1))] += int(m.group(2))
            elif line.startswith("[name_mismatch] "):
                m = re.search(r"kernel='([^']+)'[^>]*count=(\d+)", line)
                if m: nk[synth_cat(m.group(1))] += int(m.group(2))
        # GEMM per-(M,N,K) is not in per-kernel lines; GEMM synthetic count = synth['GEMM...']
        # and GEMM match count is in the GEMM summary line (g_match). We'll report
        # GEMM separately: match = (synth count - grid - name) for GEMM is wrong
        # because GEMM uses per-(M,N,K) verdict, not per-name. So for GEMM, the
        # per-name match/mismatch isn't captured. We'll mark GEMM verdict via the
        # GEMM summary (match/mismatch/unseen as shapes, not invocations).

        row = {"case": cid, "synth": dict(synth), "real": dict(real),
               "match": dict(mk), "grid": dict(gk), "name": dict(nk)}
        per_case_rows.append(row)
        # aggregate
        for c,v in synth.items(): agg_synth[c]+=v
        for c,v in real.items(): agg_real[c]+=v
        for c,v in mk.items(): agg_match[c]+=v
        for c,v in gk.items(): agg_grid[c]+=v
        for c,v in nk.items(): agg_name[c]+=v
        print(f"[done] {cid}: synth_total={sum(synth.values())} real_total={sum(real.values())}", file=sys.stderr)

    # ---- report ----
    cats = sorted(set(agg_synth)|set(agg_real)|set(agg_match)|set(agg_grid)|set(agg_name))
    print("\n=== Aggregate across 21 cases (per kernel category) ===\n")
    print(f"{'category':28} {'synth_emit':>12} {'real_msprof':>12} {'1-to-1?':>9} {'matched':>10} {'grid_mis':>9} {'name_mis':>9}")
    print("-"*100)
    for c in cats:
        s = agg_synth[c]; r = agg_real[c]
        one = "YES" if s==r else f"diff{s-r:+d}"
        print(f"{c:28} {s:12} {r:12} {one:>9} {agg_match[c]:10} {agg_grid[c]:9} {agg_name[c]:9}")
    print("-"*100)
    print(f"{'TOTAL':28} {sum(agg_synth.values()):12} {sum(agg_real.values()):12} {'':>9} {sum(agg_match.values()):10} {sum(agg_grid.values()):9} {sum(agg_name.values()):9}")

    print("\n=== Per-case (synth vs real total invocations) ===\n")
    print(f"{'case':44} {'synth':>12} {'real':>12} {'1-to-1?':>9}")
    print("-"*80)
    for row in per_case_rows:
        s=sum(row['synth'].values()); r=sum(row['real'].values())
        one="YES" if s==r else f"diff{s-r:+d}"
        print(f"{row['case']:44} {s:12} {r:12} {one:>9}")

    json.dump(per_case_rows, open("/tmp/bucket_eval.json","w"), indent=1)
    print("\n[dumped per-case detail to /tmp/bucket_eval.json]")

if __name__ == "__main__":
    main()
