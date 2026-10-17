#!/usr/bin/env bash
# Combined 910B serving capture for ONE case: msprof-wraps the PLUGIN serving
# run (trace_serving --mode plugin --qps) so a single run yields BOTH the msprof
# GT timeline AND the real operator records (prediction). Then parse GT + predict
# + compare → comparison.json. Mirrors A800's same-process nsys methodology.
# Args: id model tp ep qps gpu_ids
set -uo pipefail
ROOT=/mnt/gpfs/midlayer_yxh; cd "$ROOT"
PY=${PY:-/usr/local/python3.11.14/bin/python3}
HW=configs/hardware/huawei_910b.yaml
DS=/mnt/gpfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
N=${N:-128}
EXP=serve_exp/serve_exp_910b/eval_serving_sharegpt
id=$1; model=$2; tp=$3; ep=$4; qps=$5; gpus=$6
D=$EXP/$id
rm -rf "$D"; mkdir -p "$D/msprof"
export ASCEND_RT_VISIBLE_DEVICES=$gpus
# Async serving (_run_serving) trips a PyTorch OpenMP "Invalid thread pool" abort
# under fork whenever there is >1 worker process (tp>1 OR ep>1). spawn avoids it.
if [ "$tp" -gt 1 ] || [ "$ep" -gt 1 ]; then export VLLM_WORKER_MULTIPROC_METHOD=spawn; export OMP_NUM_THREADS=8; fi

echo "[$id] combined msprof+plugin capture (tp=$tp ep=$ep qps=$qps NPU=$gpus)"
# timeout: msprof's post-capture auto-export stalls ("There is no timeline data
# to export", then sleeps forever) on large 30B-A3B runs in CANN 8.5.1 — the raw
# op data IS captured (per-device sqlite), but the auto-export never produces
# op_summary CSV. Cap msprof so it can't hang the pipeline; the import+export
# block below rebuilds the summary + CSV from the captured sqlite regardless.
MSP_PY=/usr/local/Ascend/cann-8.5.1/tools/profiler/profiler_tool/analysis/msprof/msprof.py
timeout 1800 msprof --output="$D/msprof" --task-time=on --aic-metrics=PipeUtilization --ascendcl=on \
  --application="$PY scripts/serve_trace/trace_serving.py --mode plugin --qps $qps \
--model $model --tensor-parallel-size $tp --expert-parallel-size $ep --hw-config $HW \
--dataset $DS --seed 42 --num-requests $N --seq-len 8192 \
--timeout 3600 \
--trace-output $D/plugin_trace.json" > "$D/capture.log" 2>&1
rc=$?
if [ ! -f "$D/meta.json" ]; then echo "[$id] FAIL: no meta.json (capture rc=$rc)"; tail -5 "$D/capture.log"; exit 1; fi

echo "[$id] msprof import+export (auto-export stalls on large runs) + merge slices"
# For each device PROF dir, if msprof's auto-export didn't produce op_summary
# slices, run `msprof.py import` (parse raw -> build summary) then `export
# summary --format=csv` (extract op_summary to CSV). Idempotent (no-op if the
# auto-export already produced slices). This is the unblock for the CANN 8.5.1
# auto-export stall — the captured sqlite always has the data; only the export
# step is flaky.
$PY - "$D/msprof" "$MSP_PY" <<'EXPORT' >> "$D/capture.log" 2>&1
import sys, glob, os, subprocess
base, msp_py = sys.argv[1], sys.argv[2]
for prof in sorted(glob.glob(os.path.join(base, "PROF_*"))):
    od = os.path.join(prof, "mindstudio_profiler_output")
    have = glob.glob(os.path.join(od, "op_summary*.csv"))
    if have:
        continue  # auto-export already produced op_summary
    # import: parse raw sqlite -> build the summary the export reads.
    subprocess.run([sys.executable, msp_py, "import", "-dir", prof],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    # export summary -> op_summary_slice_*.csv (one per 1M rows).
    subprocess.run([sys.executable, msp_py, "export", "summary",
                    "-dir", prof, "--format=csv"],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    slices = glob.glob(os.path.join(od, "op_summary_slice_*.csv"))
    print(f"  {os.path.basename(prof)}: import+export -> {len(slices)} slices")
EXPORT
# Large runs (e.g. Llama-3.1-70B TP4, ~3.6M device ops) exceed msprof's 1M-row-
# per-file slice threshold → op_summary_slice_{0..N}.csv per device.
# parse_msprof_timeline reads op_summary_*.csv; concat each device's slices into
# one file (header once) and drop the slices so the parser sees the full set.
$PY - "$D/msprof" <<'MERGE' >> "$D/capture.log" 2>&1
import sys, glob, os, csv
base = sys.argv[1]
for prof in sorted(glob.glob(os.path.join(base, "PROF_*"))):
    od = os.path.join(prof, "mindstudio_profiler_output")
    slices = sorted(glob.glob(os.path.join(od, "op_summary_slice_*.csv")))
    if len(slices) <= 1:
        continue  # small model: single op_summary_*.csv — parse handles as-is
    merged = os.path.join(od, "op_summary_merged.csv")
    with open(merged, "w", newline="") as out:
        w = None
        for s in slices:
            with open(s, newline="") as f:
                rd = csv.DictReader(f)
                if w is None:
                    w = csv.DictWriter(out, fieldnames=rd.fieldnames)
                    w.writeheader()
                for row in rd:
                    w.writerow(row)
    for s in slices:
        os.remove(s)
    print(f"  merged {len(slices)} slices -> op_summary_merged.csv in {os.path.basename(prof)}")
MERGE
$PY scripts/serve_trace/parse_msprof_timeline.py "$D/msprof" --meta "$D/meta.json" \
  --emit "$D/nsys_kernels.json" >> "$D/capture.log" 2>&1
# Distill op_summary into small JSONs (gemm_grid_lut + op_grids + op_profile),
# the disk-cheap checker truth + future per-shape pipe times. Runs BEFORE the
# cleanup (reads the op_summary CSV still present at this point). The JSONs are
# a fallback source for the checker (csv -> sqlite -> json) so the grid LUT +
# per-shape pipe profile survive even if the sqlite is later deleted.
$PY scripts/serve_trace/distill_msprof_op_summary.py "$D" >> "$D/capture.log" 2>&1
if [ -f "$D/nsys_kernels.json" ]; then
  # Keep device sqlite (for _extract_msprof_per_step TTFT serial), delete
  # large raw data (host/, device_*/data/, mindstudio_profiler_output/).
  # The distilled JSONs (gemm_grid_lut/op_grids/op_profile) carry the
  # op_summary grid + pipe-profile in ~1.3MB (vs the 215MB CSV deleted here).
  for prof in "$D/msprof"/PROF_*; do
    rm -rf "$prof/host" "$prof"/device_*/data "$prof/mindstudio_profiler_output" 2>/dev/null
  done
fi
$PY - "$id" "$model" "$tp" "$ep" "$N" "$D" >> "$D/capture.log" 2>&1 <<'PYEOF'
import sys; sys.path.insert(0,'.')
from pathlib import Path; import json
import scripts.serve_predict.build_eval_dataset as b
id,model,tp,ep,N,D = sys.argv[1:7]
case=dict(id=id,model=model,tensor_parallel_size=int(tp),expert_parallel_size=int(ep),num_requests=int(N))
cd=Path(D); b.run_prediction(case,cd,'configs/hardware/huawei_910b.yaml')
c=b.run_comparison(case,cd); json.dump(c,open(cd/'comparison.json','w'),indent=2)
print('G1_APE=%s' % (c.get('group1_synth_gpu',{}).get('ape')))
PYEOF
g1=$(grep -o "G1_APE=[0-9.]*" "$D/capture.log" | tail -1)
echo "[$id] DONE  $g1"
