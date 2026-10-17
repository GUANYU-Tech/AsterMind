#!/usr/bin/env python3
"""Legacy Huawei 910B trace-synthesis evaluation for one case.

Given a case (id/model/tp/ep), a completed analytic prediction dir, and a
completed msprof capture dir, assemble the A800-style experiment dir and run
build_eval_dataset.run_prediction + run_comparison → G1/G3 APEs.

  G1 = predicted GPU-only ms  vs  msprof timeline gpu_ns  (pure compute)
  G3 = predicted GPU + real CPU  vs  wall  (end-to-end)

The msprof timeline (ground truth) replaces nsys; parse_msprof_timeline.py emits
nsys_kernels.json in the exact schema run_comparison expects. Prediction side is
PLUGIN mode (real vLLM on device + real Scheduler/operator capture → recipe
operator→kernel); analytic mirror is forbidden per goal.
"""
import sys, os, json, glob, subprocess
from pathlib import Path

ROOT = Path("/mnt/gpfs/midlayer_yxh")
sys.path.insert(0, str(ROOT))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--hw-config", default="configs/hardware/huawei_910b.yaml")
    ap.add_argument("--analytic-dir", default=None,
                    help="dir with plugin_trace.json + scheduler sidecar")
    ap.add_argument("--msprof-dir", required=True,
                    help="msprof output dir (PROF_* + meta.json)")
    ap.add_argument("--out-dir", required=True, help="assembled experiment dir")
    ap.add_argument("--gpu-ids", default="0",
                    help="ASCEND_RT_VISIBLE_DEVICES for the plugin vLLM run")
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()

    case = {"id": a.case_id, "model": a.model,
            "tensor_parallel_size": a.tp, "expert_parallel_size": a.ep}
    case_dir = Path(a.out_dir); case_dir.mkdir(parents=True, exist_ok=True)

    # 1) bring in the analytic prediction trace + sidecar
    meta = Path(a.msprof_dir) / "meta.json"

    # 1) prediction: PLUGIN mode (real vLLM + real operator capture; analytic is
    #    forbidden per goal). Same ShareGPT sample as the msprof GT via identical
    #    --dataset/--seed/--num-requests (ShareGPTDataset(seed).sample is
    #    deterministic), so prediction and GT share one workload (valid G1).
    m = json.load(open(meta))
    nreq = int(m.get("num_requests", a.num_requests))
    seed = int(m.get("seed", 42))
    dataset = m.get("dataset", "/mnt/gpfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json")
    env = dict(os.environ)
    # Do NOT touch PYTHONPATH: on Ascend it carries the CANN toolkit path that
    # provides `acl` — popping/overriding it makes the vLLM spawn EngineCore
    # worker fail with `ModuleNotFoundError: No module named 'acl'`. The repo
    # root reaches the worker via trace_serving.py's .pth self-insert, not here.
    env["CUDA_TRACE_OUTPUT"] = str(case_dir / "plugin_trace.json")
    env["ASCEND_RT_VISIBLE_DEVICES"] = a.gpu_ids
    subprocess.run([sys.executable,
                    str(ROOT / "scripts/serve_trace/trace_serving.py"),
                    "--mode", "plugin", "--model", a.model,
                    "--tensor-parallel-size", str(a.tp),
                    "--expert-parallel-size", str(a.ep),
                    "--hw-config", a.hw_config,
                    "--dataset", dataset, "--seed", str(seed),
                    "--num-requests", str(nreq),
                    "--seq-len", str(a.max_model_len), "--max-tokens", str(a.max_tokens),
                    "--timeout", "2400",
                    "--trace-output", str(case_dir / "plugin_trace.json")],
                   env=env, check=True)

    # 2) ground truth: msprof timeline → nsys_kernels.json (inference window)
    subprocess.run([sys.executable,
                    str(ROOT / "scripts/serve_trace/parse_msprof_timeline.py"),
                    a.msprof_dir, "--meta", str(meta),
                    "--emit", str(case_dir / "nsys_kernels.json")], check=True)

    # 3) serving_metrics.json (real wall) from driver meta
    m = json.load(open(meta))
    json.dump({"wall_s": m["generate_s"], "num_requests": m["num_requests"],
               "total_out_tokens": m.get("total_out_tokens")},
              open(case_dir / "serving_metrics.json", "w"), indent=2)

    # 4) meta.json for eval
    json.dump({"case_id": a.case_id, "model": a.model,
               "tensor_parallel_size": a.tp, "expert_parallel_size": a.ep,
               "hw_config": a.hw_config}, open(case_dir / "meta.json", "w"), indent=2)

    # 5) run prediction + comparison
    from scripts.serve_predict.build_eval_dataset import run_prediction, run_comparison
    ok = run_prediction(case, case_dir, a.hw_config)
    print(f"[predict] ok={ok}")
    cmp = run_comparison(case, case_dir)
    print("=== COMPARISON ===")
    print(json.dumps(cmp, indent=2, ensure_ascii=False))
    json.dump(cmp, open(case_dir / "comparison.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
