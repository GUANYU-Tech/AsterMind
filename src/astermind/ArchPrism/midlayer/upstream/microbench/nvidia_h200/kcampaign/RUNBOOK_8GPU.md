# H200 8-GPU slot runbook (2026-09-27)

Scope: everything the 2026-09-27 whole-machine H200 slot needs from branch `wt-h200-8gpu-prep` (worktree `/share_data/yangxinhao/chip_modeling/wt-h200-8gpu-prep`, based on `wt-h200-support` e1a1e3af1). All GPU work runs on vLLM 0.30.0 (`/mnt/public/yangxinhao/envs/vllm030`, torch 2.13.0+cu129, FlashInfer 0.6.18.post1). The H200 is gone after 2026-09-27, so nothing here may wait for a second attempt: run the smoke first, then the timed jobs in their slots.

Hard rules that every command below already enforces: one process per GPU; every GPU use holds `/mnt/public/yangxinhao/h200_microbench/kcampaign/runs/gpu<N>.proc.lock` (`flock -x`, ascending order, multi-GPU jobs take all their GPUs' locks) through `microbench/nvidia_h200/kcampaign/remote/with_gpu_locks.sh`; clocks are not locked, so each job samples `nvidia-smi` every 500 ms and `summarize_clock_log.py` flags busy samples not at 1980 MHz; ssh commands are wrapped in `timeout`; long jobs are started with `setsid nohup` and never stopped with `pkill -f` patterns that could match the controlling ssh.

## 0. Deployed code

The runbook assumes the snapshot `SRC=/mnt/public/yangxinhao/src/midlayer_h200_8gpu_prep/current` (symlink to `<commit>/`, which contains `DEPLOYED_COMMIT`). It was deployed on 2026-09-26; to redeploy after a new commit (local shell, worktree root; written through `h200-aicoder`, which shares `/mnt/public`, so nothing starts on `h200-tmp`):

```bash
C=$(git rev-parse --short=9 HEAD)
git archive --format=tar HEAD microbench/nvidia_h200 scripts configs | \
  timeout 300 ssh h200-aicoder "D=/mnt/public/yangxinhao/src/midlayer_h200_8gpu_prep; mkdir -p \$D/$C && tar -xf - -C \$D/$C && echo $C > \$D/$C/DEPLOYED_COMMIT && ln -sfn \$D/$C \$D/current && cat \$D/current/DEPLOYED_COMMIT"
```

FlashInfer comm JIT modules for sm_90a (`trtllm_comm`, `trtllm_mnnvl_comm`) were prebuilt on CPU on 2026-09-26 into `/mnt/public/yangxinhao/h200_microbench/home/.cache/flashinfer/0.6.18.post1/90a/cached_ops/`; every script below exports `HOME=/mnt/public/yangxinhao/h200_microbench/home` so the cache is hit and no nvcc build happens inside a timed slot.

## 1. First 10 minutes after the host shows 8 GPUs

All on `ssh h200-tmp`, `SRC=/mnt/public/yangxinhao/src/midlayer_h200_8gpu_prep/current`, `R=$SRC/microbench/nvidia_h200/kcampaign/remote`.

| Step | Command | Wall time | Success check |
|---|---|---|---|
| 1a machine / topology / driver / clocks (NVML only, no lock, no CUDA context) | `bash $R/check_8gpu_machine.sh /mnt/public/yangxinhao/h200_microbench/collectives_8gpu/check_$(date +%H%M)` | 10 s | last line `CHECK_8GPU: PASS` (8x "NVIDIA H200", driver 575.57.08, every GPU pair `NV18` in `topo -m`, fabric `Completed`/`Success`, max SM clock 1980 MHz); read the "index continuity" lines: a `WARN` means the four 2026-09-26 GPUs changed index or are absent (then the kernel chains' `gpu<N>` lock ids and `--physical-gpu` values refer to different physical GPUs — tell the user before starting any chain) |
| 1b one 8-rank all-reduce pass (all 8 locks) | `timeout 900 bash $R/run_8gpu_collectives.sh quick` (foreground is fine) | about 2-3 min incl. lock wait; lock wait is capped at 900 s (`KC_LOCK_WAIT_S`) | ends with `ALL DONE out=...`; in `<out>/job.log`: `world_size=8 selection={...}` shows `flashinfer_workspace_backend` (`mnnvl`, or `trtllm` if NVSwitch multicast is unavailable), `symm_mem_disabled: False`, `custom_ar_fully_connected: True`; row `tp_all_reduce_dispatch tokens=1` has backend `FLASHINFER` and `tokens=512` has `SYMM_MEM_MULTIMEM` (if torch symm-mem multicast is unavailable it is `CUSTOM` instead — record which); every time finite; `clock_summary.json` prints `OK` per GPU or lists flagged rows |
| 1c GLM-5.2 TP8 server start to first token (all 8 locks) | section 3, `smoke` command | 10-25 min (first start JIT-compiles kernels) | section 3 smoke row |

GPU 0 belongs to the DSR1 instance for most of the day. Steps 1b and 1c need all eight locks: if `gpu0.proc.lock` is held they wait up to `KC_LOCK_WAIT_S` seconds and then abort with exit 3 without touching any GPU. Coordinate a 15-20 minute window with the DSR1 instance for 1b+1c, or run them as the first thing of the 12:00 slot. Do not start single-GPU kernel chains on GPUs 1-7 before 1b/1c are done (the chains' `launch.sh` holds each `proc.lock` for a whole plan).

## 2. 12:00-12:30 — 8-rank collectives (vLLM 0.30)

Precondition: all single-GPU chains stopped (section 4) and the DSR1 TP8 job finished.

```bash
cd /mnt/public/yangxinhao/h200_microbench && mkdir -p logs && \
  setsid nohup bash $R/run_8gpu_collectives.sh full > logs/collectives_8gpu_full.log 2>&1 < /dev/null &
# follow: timeout 10 tail -n 20 /mnt/public/yangxinhao/h200_microbench/logs/collectives_8gpu_full.log
```

Expected wall time 10-15 min (the 2026-09-24 2+4-rank custom all-reduce run took 26 s; the new runner adds prefill sizes up to 8192 tokens and five series), hard budget 30 min. Output `/mnt/public/yangxinhao/h200_microbench/collectives_8gpu/full_<stamp>/`:

| Sub-run | What | Why |
|---|---|---|
| `car_ws248/` | `custom_allreduce_latency_runner.py --world-sizes 2 4 8`: vLLM `CustomAllreduce` standalone (8 MiB `max_size`), 64-node CUDA-graph chain, BF16 `tokens x 7168`, tokens 1-512, 30 trials, median per rank, slowest rank; plus NCCL all-reduce(+clone) | identical probe/derivation to the 4-rank reference `microbench/nvidia_h200/references/custom_allreduce_graph_node_sm90.json`; the `8:` row of `interconnect.custom_all_reduce_service` in `configs/hardware/nvidia_h200.yaml` is currently a copy of `4:` |
| `tpc_h7168_auto/` | `vllm_tp_collectives_runner.py --world-sizes 2 4 8 --hidden 7168`: vLLM's own distributed init and dispatch; series `tp_all_reduce_dispatch` (records the backend vLLM picks per payload), `flashinfer_fused_ar_residual_rmsnorm` (the `fuse_allreduce_rms` pass call, tokens <= 0.5 MiB/(hidden*2) = 36 at TP8), `flashinfer_fused_ar_residual_rmsnorm_model_op` (same kernel as called by the model-level op that DeepseekV32 / GLM-5.2 use under breakable CUDA graphs, separate `norm_out`), `pynccl_all_reduce`, `tp_all_gather_dim0`, `tp_reduce_scatter_dim0`, `tp_all_gather_logits` (`tokens x 154880/TP`); tokens 1-512 (64-node chain) and 1024-8192 (16-node chain) | what vLLM 0.30 actually runs at TP8 on H200 |
| `tpc_h7168_trtllm/` | same, TP8, `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` | cross-check against the 0.24 trtllm fused reference (`flashinfer_allreduce_fusion_graph_node_sm90.json`) if auto picked mnnvl |
| `tpc_h6144_auto/` | same, TP8, hidden 6144 (GLM-5.2; fusion limit 42 tokens) | GLM-5.2 shapes |
| `car_ws8_repeat/`, `tpc_h7168_auto_repeat/` | repeats of the TP8 runs | repeatability |
| `machine.txt`, `versions.txt`, `clocks.csv`, `clock_summary.json`, `job.log` | provenance and clock telemetry | |

Success checks: `job.log` ends with `ALL DONE`; each sub-run has `report.json` plus one `ws<W>_rank<r>.json` per rank; `report.json` `environment.vllm_version` is `0.30.0`; in `tpc_*` reports every rank agrees on `selection` (the runner fails otherwise); `clock_summary.json` has `busy_non_1980_samples == 0` for all GPUs or lists `flagged_rows` (re-run only the flagged sub-run if time allows); the repeat runs agree with the first runs to within their trial spread (compare medians row by row).

Expected vLLM 0.30 dispatch at TP8, SM90, BF16 (from source, to be confirmed by the `backend` column): payload <= 0.5 MiB -> FlashInfer all-reduce (`VLLM_ALLREDUCE_USE_FLASHINFER=1` default; backend `auto` -> mnnvl, falling back to trtllm without NVSwitch multicast); 0.5 MiB < payload <= 64 MiB -> torch symmetric-memory `multimem_all_reduce_` (custom AR `max_size` shrinks to 256 KiB when torch symm-mem is enabled, so the IPC custom AR is never reached for 2-D BF16); > 64 MiB -> PyNccl. NCCL symmetric memory is off (`VLLM_USE_NCCL_SYMM_MEM=0`). With compilation (O2 default) the decoder's all-reduce + residual RMSNorm pairs are replaced by FlashInfer `allreduce_fusion` (kARResidualRMSNorm, fp32 acc, PDL) for batches up to 36 tokens (hidden 7168) / 42 tokens (hidden 6144); DeepseekV32 / GlmMoeDsa run with compilation mode NONE (breakable CUDA graphs, auto-enabled) and get the same kernel from `vllm/models/common/ops/fused_allreduce_rms_norm.py` under the same token limit. All-gather / reduce-scatter go to PyNccl.

Post-processing (after the slot, not during it; no numbers are written before they are measured): the `8:` rows for `custom_all_reduce_service` come from `car_ws248/report.json` `world_sizes["8"].summary[*].custom_all_reduce_ns` keyed by `payload_bytes`, rounded to 0.1 ns as the existing rows are:

```bash
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); [print(f\"        {x['payload_bytes']}: {round(x['custom_all_reduce_ns'], 1)}\") for x in r['world_sizes']['8']['summary']]" car_ws248/report.json
```

That block's `software_revision` is vLLM 0.24.0 while the new run is 0.30.0 (it also re-measures 2 and 4 ranks on 0.30); whether to replace the whole block with the consistent 0.30 table, and whether the Serving model should switch to the 0.30 dispatch (FlashInfer / symm-mem instead of custom AR), is a decision for the main agent and user, and the reference JSON goes to `microbench/nvidia_h200/references/` with its sha256 in the YAML.

## 3. 12:30-15:00 — GLM-5.2 TP8 real serving (vLLM 0.30, dummy weights)

Scripts: `scripts/evaluation/serving/nvidia_h200_glm_5_2_tp8_capture_runner.py` (entry point with the GLM-5.2 model profile of the shared orchestrator `nvidia_h200_tp8_serving_capture.py`, run with `/usr/bin/python3.12`; modes `smoke` and `run --phases closedloop nsys`, `--dry-run` prints every command without starting anything) and `scripts/evaluation/serving/nvidia_h200_glm_5_2_tp8_serve.sh` (the `vllm serve` line). Modeled on the DSV3-L32 H200 captures (`nvidia_h200_nsys_pure_decode_capture.sh`, `nvidia_h200_nsys_prefill_capture.sh`, remote `sweep_closedloop.sh`). The orchestrator refuses to start unless it inherited the eight lock fds from `with_gpu_locks.sh`, nvidia-smi shows GPUs 0-7 all "NVIDIA H200", no compute process is on any GPU and port 8012 is free; it logs clocks for the whole job and flags busy non-1980 MHz samples per C and per window.

Model: config/tokenizer-only copy of ModelScope `ZhipuAI/GLM-5.2-FP8` at `/mnt/public/yangxinhao/models/GLM-5.2-FP8-config` (BF16 original at `GLM-5.2-config`), `--load-format dummy`. FP8 (e4m3, block [128,128], dynamic activations) is the only option: 743.4 B parameters are about 175.6 GiB per GPU in BF16 at TP8, above the 140.4 GiB usable HBM, versus about 88.2 GiB in FP8. KV cache `fp8_ds_mla` (53,940 B per token per GPU versus 92,628 B for BF16): C=64 x 9,216 tokens needs 29.6 GiB (BF16 KV would need 50.9 GiB of the ~38 GiB left at `--gpu-memory-utilization 0.90`). MoE routing `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random` (exists in 0.30; the run fails if the server log lacks "Simulating MoE routing"). Server flags: TP8, `--max-model-len 10240 --max-num-batched-tokens 32768 --max-num-seqs 128 --no-enable-prefix-caching --generation-config vllm`, MTP off.

vLLM 0.30 paths for GLM-5.2 on SM90 (read from source, confirmed by the startup log lines the smoke records): `GlmMoeDsaForCausalLM` -> `vllm/models/deepseek_v32` (CUDA DSA module); attention `FLASHMLA_SPARSE` with FP8 KV (`FLASH_ATTN_MLA_SPARSE` only with BF16 KV); indexer DeepGEMM `fp8_fp4_mqa_logits` + `top_k_per_row_prefill` (prefill) and `fp8_fp4_paged_mqa_logits` (decode), 21 of 78 layers run their own indexer; block-FP8 MoE under TP -> Triton fused MoE (tuned H200 E=256,N=256 block config present); dense FP8 linears FlashInfer SM90 block GEMM for M<32, DeepGEMM for M>=32; the architecture auto-enables breakable CUDA graphs, which sets compilation mode NONE (no torch.compile fusion passes), so all-reduce + residual RMSNorm is fused by the model-level op (`vllm/models/common/ops/fused_allreduce_rms_norm.py`, FlashInfer `allreduce_fusion`, up to 42 tokens at hidden 6144) and larger batches use the plain all-reduce dispatch of section 2. The FlashInfer `fp8_blockscale_gemm_sm90` JIT module was prebuilt on CPU on 2026-09-26 (76 s) into the shared cache; DeepGEMM / TensorRT-LLM per-shape kernels still compile on first use, which the smoke warms.

```bash
SRC=/mnt/public/yangxinhao/src/midlayer_h200_8gpu_prep/current; R=$SRC/microbench/nvidia_h200/kcampaign/remote
G=$SRC/scripts/evaluation/serving/nvidia_h200_glm_5_2_tp8_capture_runner.py; C=$(cat $SRC/DEPLOYED_COMMIT)
cd /mnt/public/yangxinhao/h200_microbench
# optional, no GPU: print the plan
/usr/bin/python3.12 $G run --tag dry --source-commit $C --routing uniform_random --kv-cache-dtype fp8_ds_mla --concurrency 1 4 16 64 --phases closedloop nsys --dry-run
# 1c smoke (first 10 minutes): server start -> one 16-token completion -> shutdown
T=$(date +%Y%m%d_%H%M); setsid nohup bash $R/with_gpu_locks.sh 0,1,2,3,4,5,6,7 900 -- /usr/bin/python3.12 $G smoke --tag $T --source-commit $C --routing uniform_random --kv-cache-dtype fp8_ds_mla --concurrency 64 > logs/glm52_tp8_smoke_$T.log 2>&1 < /dev/null &
# 12:30 full run
T=$(date +%Y%m%d_%H%M); setsid nohup bash $R/with_gpu_locks.sh 0,1,2,3,4,5,6,7 900 -- /usr/bin/python3.12 $G run --tag $T --source-commit $C --routing uniform_random --kv-cache-dtype fp8_ds_mla --concurrency 1 4 16 64 --phases closedloop nsys > logs/glm52_tp8_run_$T.log 2>&1 < /dev/null &
# follow: timeout 10 tail -n 30 logs/glm52_tp8_run_$T.log
```

Protocol: closed loop is `vllm bench serve` random dataset, input 7373 +-11.1% (6554-8192), output 922 +-11.1% (820-1024), `--ignore-eos`, `--max-concurrency C`, 10*C measured prompts after 2*C warm-ups, C in {1, 4, 16, 64}, no nsys. Then one server under `nsys --capture-range=cudaProfilerApi --cuda-graph-trace=node` (vLLM `--profiler-config '{"profiler": "cuda"}'`, windows via `/start_profile` / `/stop_profile`): per C one pure-decode window (exactly C requests, captured after all C first tokens are counted on the TTFT histogram and before the shortest request can finish) and one prefill-only window (fixed 7373-token prompts, 1 output token, starting 3 s into the load); windows are 1.5 s.

| | Wall time (estimate scaled from the 2026-09-25 DSV3-L32 TP4 runs, not measured) | Output | Success check |
|---|---|---|---|
| smoke | 10-25 min (first start compiles DeepGEMM / FlashInfer / Triton kernels) | `h200_microbench/glm52_tp8/smoke_<T>/smoke.json`, `server.log`, clock log | log line `smoke status=ok`; `completion_tokens == 16`; KV capacity >= 64 x 9,216 = 589,824 tokens; the printed backend lines show FLASHMLA_SPARSE, the Triton FP8 MoE, the FlashInfer all-reduce workspace and "Simulating MoE routing"; all eight GPUs free afterwards |
| closed loop | 5-20 min start + about 2.5 / 3 / 6 / 13 min for C = 1 / 4 / 16 / 64 | `h200_microbench/closedloop/glm52_tp8_8k1k_uniform_random_<T>/` (`c{C}.json`, `run.json`) | one `c{C}.json` per C with 10*C completed requests, 0 failed, 0 preemptions; clock flags per C |
| nsys windows | 5-15 min start + 8 x (0.5-2 min ramp + 1-6 min report flush) | `h200_microbench/nsys/glm52_tp8_windows_uniform_random_<T>/` (`window_1..8.nsys-rep`, `windows.txt` in the DSV3 `<label> start|stop <time>` format, `windows.json` with purity checks and clock flags) | 8 `.nsys-rep` files; every decode window `pure_decode: true` in `windows.json`; `summarize_nvidia_h200_nsys_decode_windows.py` reads `windows.txt` |
| total | about 1.3-2.0 h (budget 2.5 h, 12:30-15:00) | | |

If the smoke fails in FlashMLA sparse with FP8 KV (only exercised by kernel-campaign runners so far), rerun the smoke with `--kv-cache-dtype auto` (BF16 KV -> `FLASH_ATTN_MLA_SPARSE`); the orchestrator stops before measuring when KV capacity is below max(C) x 9,216, so use the capacity the smoke prints to cut the C list (expected to fit about C <= 32). nsys reports are not exported to sqlite in the slot; do that later on h200-aicoder.

## 3b. DeepSeek-R1-0528 TP8 real serving (vLLM 0.30, dummy weights, MTP off and on)

H200 ground truth comparable with the InferenceX DSR1 8K/1K TGS-Interactivity points (`dsr1-fp8-h200-sglang` and `dsr1-fp8-h200-sglang-mtp`: DeepSeek-R1-0528 FP8, one TP8/EP1 SGLang instance, C = 4, 8, 16, 32, 64; raw rows in `experiment_logs/inferencex_alignment/raw/dsr1_all_20260924.json`). Needs all eight locks for about 2 h, so it cannot overlap sections 1b/1c, 2 or 3; order it against them explicitly.

Scripts: `scripts/evaluation/serving/nvidia_h200_deepseek_r1_tp8_capture_runner.py` (DSR1 model profile of the same orchestrator as section 3: same lock-fd / empty-GPU / port / clock checks, same smoke, closed-loop and nsys window logic, plus `kill <runner pid>` stops the server before exiting) and `scripts/evaluation/serving/nvidia_h200_deepseek_r1_tp8_serve.sh`. The docstring of the runner lists every protocol decision with its source; in short:

| Item | Setting | Source |
|---|---|---|
| Model | `/mnt/public/yangxinhao/models/DeepSeek-R1-0528-config` (config, tokenizer, generation config from ModelScope `deepseek-ai/DeepSeek-R1-0528`, downloaded 2026-09-26 via h200-aicoder), `--load-format dummy`, TP8, FP8 e4m3 block [128,128] | InferenceX master config `model: deepseek-ai/DeepSeek-R1-0528`; its config.json equals `DeepSeek-V3-config/config.json` except `transformers_version`, so execution is DeepSeek-V3's; the R1-0528 tokenizer (differs from V3's) builds the prompts |
| Routing | `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random`; run fails without the "Simulating MoE routing" log line | dummy weights skew the router |
| Server | `--max-num-batched-tokens 32768 --max-num-seqs 128 --max-model-len 10240 --no-enable-prefix-caching --kv-cache-dtype auto --generation-config vllm --gpu-memory-utilization 0.95` | InferenceX SGLang recipe `--chunked-prefill-size 32768`, `--disable-radix-cache`, no KV dtype flag (BF16 KV); BF16 KV selects FLASH_ATTN_MLA on SM90 (vLLM 0.30 `platforms/cuda.py`), the path of the `dsr1_mtpoff_fa3` / `dsr1_mtpon_verify` kernel plans |
| MTP on | `--speculative-config {"method":"mtp","num_speculative_tokens":2,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.8244,0.5216]}` | InferenceX `dsr1_fp8_h200_mtp.sh`: EAGLE steps 2, draft tokens 3, top-k 1 = 2 drafts per verify; dummy weights accept ~no draft, so vLLM 0.30 synthetic rejection imposes the repo's `inferencex_sglang_h200_mean` profile (per-token 0.673 from the InferenceX SGLang server logs, run 34450728410) with the simulator's maximum-entropy count distribution P(K=0,1,2) = 0.1756/0.3027/0.5216 (mean acceptance length 2.346) |
| Client | `vllm bench serve` random, 7373 / 922 +-1/9 (6554..8192 / 820..1024), `--ignore-eos`, `--max-concurrency C`, `--num-prompts 10*C`, `--num-warmups 2*C`, `--temperature 0` | InferenceX `benchmark_lib.sh` (`--random-range-ratio 0.8`, `--num-prompts CONC*10`, `--num-warmups 2*CONC`, `--request-rate inf`) and its client's temperature 0; the MTP client's `--use-chat-template` (a few tokens per 7.4K prompt) is not reproduced |
| C | closed loop 1, 4, 8, 16, 32, 64; nsys windows (one pure-decode + one prefill per C) at 1, 4, 16, 64 | InferenceX C list plus C=1; nsys C as the DSV3-L32 captures and kernel plans |

Memory (estimated from config.json, the smoke measures it): FP8 weights about 79.5 GiB per GPU (+2 GiB MTP layer); BF16 MLA KV 70,272 B per token (71,424 B with the MTP layer), so max C x (9,216 + drafts) = 589,824 tokens (MTP off) / 589,952 tokens (MTP on) need 38.6 / 39.2 GiB per GPU. At 0.95 of 140.4 GiB this leaves about 12 GiB for activations, CUDA graphs and communication buffers; MTP on is the tight case, hence the smoke starts it first.

```bash
SRC=/mnt/public/yangxinhao/src/midlayer_h200_8gpu_prep/current; R=$SRC/microbench/nvidia_h200/kcampaign/remote
D=$SRC/scripts/evaluation/serving/nvidia_h200_deepseek_r1_tp8_capture_runner.py; C=$(cat $SRC/DEPLOYED_COMMIT)
COMMON="--source-commit $C --routing uniform_random --kv-cache-dtype auto --gpu-memory-utilization 0.95"
cd /mnt/public/yangxinhao/h200_microbench
# optional, no GPU: print the plan
/usr/bin/python3.12 $D run --tag dry $COMMON --mtp off on --concurrency 1 4 8 16 32 64 --nsys-concurrency 1 4 16 64 --phases closedloop nsys --dry-run
# smoke: server start -> one 16-token completion -> shutdown, MTP on then off (warms both compile caches)
T=$(date +%Y%m%d_%H%M); setsid nohup bash $R/with_gpu_locks.sh 0,1,2,3,4,5,6,7 900 -- /usr/bin/python3.12 $D smoke --tag $T $COMMON --mtp on off --concurrency 64 > logs/dsr1_tp8_smoke_$T.log 2>&1 < /dev/null &
# full run: closed loop MTP off, closed loop MTP on, nsys MTP off, nsys MTP on (in this order)
T=$(date +%Y%m%d_%H%M); setsid nohup bash $R/with_gpu_locks.sh 0,1,2,3,4,5,6,7 900 -- /usr/bin/python3.12 $D run --tag $T $COMMON --mtp off on --concurrency 1 4 8 16 32 64 --nsys-concurrency 1 4 16 64 --phases closedloop nsys > logs/dsr1_tp8_run_$T.log 2>&1 < /dev/null &
# follow: timeout 10 tail -n 30 logs/dsr1_tp8_run_$T.log
# stop early (the runner stops the server, waits for free GPUs, then exits; the locks drop with it):
#   ps -eo pid,args | awk '$2=="/usr/bin/python3.12" && $3 ~ /nvidia_h200_deepseek_r1_tp8_capture_runner.py$/ {print $1}'   then   kill <pid>
```

| | Wall time (estimate from InferenceX SGLang median E2E latency x 12 rounds per client, x1.0-1.4 for vLLM; not measured) | Output (`/mnt/public/yangxinhao/h200_microbench/...`) | Success check |
|---|---|---|---|
| smoke | 10-25 min per MTP setting cold (DeepGEMM / Triton / torch.compile), 5-10 min warm | `dsr1_tp8/smoke_mtp{on,off}_<T>/` (`smoke.json`, `server.log`, clock log) | log line `smoke status=ok` for both settings; `completion_tokens == 16`; `kv_tokens` >= 589,952 (on) / 589,824 (off); the printed `selection` shows `engine_kv_cache_dtype` `auto`, `attention_backends` `["FLASH_ATTN_MLA"]` (expected from source), the MLA prefill / Fp8 MoE / all-reduce backends (recorded, nothing assumed); a "Simulating MoE routing" backend line; MTP on also prints `spec_decode` with drafts > 0; all eight GPUs free afterwards |
| closed loop MTP off | 3-8 min start + about 1.7/2.1/2.6/3.8/6.0/9.5 min for C = 1/4/8/16/32/64 (29-44 min) | `closedloop/dsr1_tp8_8k1k_mtpoff_uniform_random_<T>/` (`c{C}.json`, `run.json`) | one `c{C}.json` per C with 10*C completed, 0 failed; `run.json` per C `preemptions` 0 and clock flags; `startup.selection` as in the smoke |
| closed loop MTP on | 3-8 min start + about 1.1/1.4/2.1/3.1/4.7/7.9 min (23-36 min) | `closedloop/dsr1_tp8_8k1k_mtpon_uniform_random_<T>/` | as MTP off, plus per C `spec_decode.mean_acceptance_length` close to 2.35 (synthetic target 2.346; includes the warm-ups) |
| nsys MTP off | 5-10 min start + 8 windows x 1.5-4 min (17-42 min) | `nsys/dsr1_tp8_windows_mtpoff_uniform_random_<T>/` (`window_1..8.nsys-rep`, `windows.txt`, `windows.json`, `run.json`) | 8 reports; every decode window `pure_decode: true`; `summarize_nvidia_h200_nsys_decode_windows.py` reads `windows.txt` |
| nsys MTP on | same (17-42 min) | `nsys/dsr1_tp8_windows_mtpon_uniform_random_<T>/` | as MTP off, plus per decode window `spec_decode` (verify steps with q_len 3) |
| total run | about 1.5-2.7 h, central about 2.0 h (smoke not included) | | nsys MTP on runs last; if the slot ends it is the part to stop |

If a smoke ends with `status=kv_capacity_insufficient`, its `selection.gpu_memory_utilization_for_required_kv` is the utilization at which that start's KV pool would just hold the target: up to 0.97, rerun that smoke with `--gpu-memory-utilization` set to it rounded up to 0.005 and use the same value for the run; above that, run MTP off with the full C list and MTP on as a separate run with `--concurrency 1 4 8 16 32 --nsys-concurrency 1 4 16` rather than switching to FP8 KV (`--kv-cache-dtype fp8` changes the attention backend and departs from the InferenceX BF16 KV). If the server fails to parse the config (native transformers 5.17 `DeepseekV3Config`, no remote code) the smoke log shows it before any timed phase.

## 3c. DeepSeek-R1 TP8 KV host offload (AgentX-derived multi-turn, MTP off)

Measures vLLM 0.30 CPU KV offload with both swap-out/in (preempted or GPU-evicted blocks stored to host and loaded back) and host prefix hits (an evicted session prefix found by the connector lookup and loaded instead of recomputed), against a prefix-caching-only control at the same load. Needs all eight locks for about 50 min; run it after the section 3b closed loops (it reuses their warm JIT caches) and before 3b's nsys phases if the slot is short.

Mechanism (vLLM 0.30 source, `site-packages/vllm/`): `--kv-offloading-size <GiB>` with `--kv-offloading-backend native` sets `kv_connector=OffloadingConnector` (or `SimpleCPUOffloadConnector` when `VLLM_USE_SIMPLE_KV_OFFLOAD=1`), `kv_role=kv_both`, `cpu_bytes_to_use=size*2^30` (`config/vllm.py:1038-1066`). The scheduler first takes the GPU prefix-cache hit, then asks the connector for the tokens beyond it (`v1/core/sched/scheduler.py:917-970`); `OffloadingConnectorScheduler.get_num_new_matched_tokens` looks the request's block hashes up in the host LRU and `update_state_after_alloc` queues an async host-to-GPU load (`distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1016-1187`), so a host prefix hit is a normal prefix-cache hit served from host. Stores are write-through of every full prompt chunk as it is computed plus the tail at finish (`_build_store_jobs`, same file :1464); there is no separate swap path, a preempted request is re-admitted through the same lookup. With MLA at TP8 (single node, `mp`) the host buffer is one replicated copy in `/dev/shm/vllm_offload_<engine>.mmap`, pinned with `cudaHostRegister` (`offloading/config.py:158-176`, `v1/kv_offload/cpu/spec.py:103-127`, `cpu/shared_offload_region.py:111-146`), so 256 GiB = 256*2^30/70,272 B = 3.9M tokens; `/dev/shm` on the 2026-09-27 host is 920 GiB tmpfs, RAM 1.8 TiB. `SimpleCPUOffloadConnector` splits the size per rank and keeps a full copy per rank (`kv_connector/v1/simple_cpu_offload_connector.py:74-92`). Prometheus: `vllm:prefix_cache_{queries,hits}_total` (GPU), `vllm:external_prefix_cache_{queries,hits}_total` (connector = host), `vllm:prompt_tokens_by_source_total{source=local_compute|local_cache_hit|external_kv_transfer}` (`v1/metrics/loggers.py:589-700`), `vllm:kv_offload_total_{bytes,time}_total{transfer_type=CPU_to_GPU|GPU_to_CPU}` (created at startup) and `vllm:kv_offload_{load,store}_{bytes,time}_total` (after the first transfer), `vllm:kv_offload_cpu_cache_usage_perc` (`offloading/metrics.py:20-130`, `v1/kv_offload/cpu/common.py`); per request `usage.prompt_tokens_details.cached_tokens` (local + external) with `--enable-prompt-tokens-details`.

Workload: InferenceX has an official agentic benchmark (AgentX, `agentic-coding` scenario in `configs/nvidia-master.yaml`, client `benchmarks/srt_agentic.sh` -> SemiAnalysis AIPerf `--scenario inferencex-agentx-mvp`; corpus HF `semianalysisai/cc-traces-weka-062126`, 393 Claude Code sessions, 56,798 main turns, median ISL 142K / OSL 444; closed loop of C session lanes, each starting at a random 25-75% point, recorded idle gaps capped at 300 s, `max_tokens` = recorded output with `ignore_eos`, 3600 s per C; offload arms `KV_OFFLOADING=dram` with SGLang HiCache or vLLM `SimpleCPUOffloadConnector`). It has no DeepSeek-R1 config (DSR1 is 8K/1K only). `scripts/evaluation/serving/agentx_session_replay.py` replays the corpus main-agent stream with exact 64-token block prefix structure (synthetic tokens per hash id, per session instance, no cross-session sharing), drops requests with in+out > 163,840 (DSR1 context; the corpus' own `_256k` variant does the same at 256K), no subagents, idle gaps capped at 60 s, 150 s warm-up + 300 s measured + up to 60 s grace per C. Workload file (built once on h200-aicoder from the corpus, sha256 `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`): `/mnt/public/yangxinhao/datasets/cc-traces-weka-062126/agentx_dsr1_160k_seed42_l16.json`; its lane starts hold 468K (C=4) and 1.01M (C=8) tokens of context, against about 0.7M tokens of GPU KV at 0.95 (3b smoke prints the real value), so C=8 overcommits GPU KV and C=4 mostly fits.

| Item | Setting |
|---|---|
| Server | `scripts/evaluation/serving/nvidia_h200_deepseek_r1_tp8_offload_serve.sh`: 3b flags except `--enable-prefix-caching --max-model-len 163840 --enable-prompt-tokens-details --enable-per-request-metrics`; `--gpu-memory-utilization` as the 3b run used |
| Arms (run order = `--offload-arms`) | `native`: `--kv-offloading-size 256 --kv-offloading-backend native` (OffloadingConnector); `gpu`: prefix caching only (control); optional `simple`: `--kv-offloading-size 768` with `VLLM_USE_SIMPLE_KV_OFFLOAD=1` (SimpleCPUOffloadConnector, eager, 96 GiB = 1.47M tokens per rank; the backend InferenceX uses for vLLM) |
| MTP | off: no DSR1 AgentX config to align MTP with, the 3b synthetic acceptance comes from 8K/1K, and it keeps the offload effect separate from speculative decoding (the MTP draft group is also special-cased by the connector) |
| C | 4 and 8 session lanes (fits GPU KV / overcommits it) |

```bash
SRC=/mnt/public/yangxinhao/src/midlayer_h200_8gpu_prep/current; R=$SRC/microbench/nvidia_h200/kcampaign/remote
D=$SRC/scripts/evaluation/serving/nvidia_h200_deepseek_r1_tp8_capture_runner.py; C=$(cat $SRC/DEPLOYED_COMMIT)
W=/mnt/public/yangxinhao/datasets/cc-traces-weka-062126/agentx_dsr1_160k_seed42_l16.json
UTIL=0.95   # the value the 3b smoke/run used
COMMON="--source-commit $C --routing uniform_random --kv-cache-dtype auto --gpu-memory-utilization $UTIL"
cd /mnt/public/yangxinhao/h200_microbench
# optional, no GPU: print the plan (also prints each lane's start context)
/usr/bin/python3.12 $D run --tag dry $COMMON --mtp off --concurrency 4 8 --phases offload --offload-arms native gpu --offload-workload $W --dry-run
# run: native arm (C=4, C=8), then the gpu control arm (C=4, C=8)
T=$(date +%Y%m%d_%H%M); setsid nohup bash $R/with_gpu_locks.sh 0,1,2,3,4,5,6,7 900 -- /usr/bin/python3.12 $D run --tag ${T}o $COMMON --mtp off --concurrency 4 8 --phases offload --offload-arms native gpu --offload-workload $W > logs/dsr1_tp8_offload_$T.log 2>&1 < /dev/null &
# optional if 20 min remain: InferenceX's vLLM connector at the overcommitted point
# ... run --tag ${T}s $COMMON --mtp off --concurrency 8 --phases offload --offload-arms simple --offload-workload $W
# stop early: kill <runner pid> as in 3b (the runner stops the server first)
```

| | Wall time (estimate, not measured) | Output (`/mnt/public/yangxinhao/h200_microbench/offload/`) | Success check |
|---|---|---|---|
| arm `native` | 5-10 min start (new compile hash: `kv_offloading_size` is a cache-config hash factor; 256 GiB /dev/shm populate + pin about 1-2 min) + 2 x 8.5-9 min | `dsr1_tp8_agentx_native_uniform_random_<T>o/` (`run.json`, `server.log`, `metrics_c{C}_{before,after}.prom`, `c{C}/requests.jsonl`, `c{C}/summary.json`, `c{C}/metrics_{start,measure_start,end}.prom`, clock log) | log `offload arm native: ... connectors=['OffloadingConnector']`; runner did not raise (it raises when the engine lacks `enable_prefix_caching=True`, the connector differs, the `CPUOffloadingSpec` / `Created mmap file /dev/shm/vllm_offload_...` / `'kv_offloading_size': 256` lines are missing, `cudaHostRegister failed` appears, any request fails, or no host prefix hit happens in any C); per C `checks` all true at C=8 (`gpu_prefix_hits_positive`, `host_prefix_hits_positive`, `gpu_to_host_bytes_positive`, `host_to_gpu_bytes_positive`; C=4 may show no host hit); `metrics_measure_window.tokens_external_kv_transfer` > 0 at C=8 |
| arm `gpu` | 5-10 min start + 2 x 8.5-9 min | `dsr1_tp8_agentx_gpu_uniform_random_<T>o/` | `connectors=[]`, `checks.no_host_queries` true, `gpu_prefix_hits_positive` true |
| total | about 45-58 min (`simple` at C=8 adds 15-20 min) | | compare per C: TTFT p50/p90, `tokens_local_compute` (recompute avoided by host hits), `preemptions`, output tokens/s |

The per-C numbers to compare are in `run.json` `concurrency.<C>`: `metrics_measure_window` (counter deltas between the replay's measure-start and end scrapes: GPU vs host prefix queries/hits and hit rates, prompt tokens by source, preemptions, host<->GPU bytes and time for `native`), `ttft_s` / `tpot_s` / `e2e_s` over measured completed requests, `requests` (measured, completed, cancelled at the end of the grace period), and `clock` flags over the measured window. Requests still running 60 s after the window are cancelled (recorded with their TTFT); long recorded outputs (p99 6.5K tokens) make those the slowest ones, so percentiles are over completed requests. No nsys window in this group (budget). If the native arm fails at startup the runner stops before the control arm; the most likely causes are `/dev/shm` space (the runner checks free space first) or host pinning time, and `--offload-arms gpu` alone is not worth the slot time.

## 4. Stopping single-GPU chains before a whole-machine job

`chain.sh <gpu> <plans...>` runs `launch.sh` per plan and `launch.sh` holds `gpu<N>.proc.lock` for the whole plan. Measured windows are kept and a later restart of the same plan skips them, so stopping mid-plan loses at most the window in flight.

```bash
# list chains (exact-prefix match, never pkill -f)
ps -eo pid,args | awk '$2=="/bin/bash" && $3=="./chain.sh" {print}'
kill <chain_pid> ...                       # no further plans start
ps -eo pid,ppid,args | grep -E 'flock -x .*gpu[0-9]+\.proc\.lock' | grep -v grep   # the running plan per GPU
kill <flock_pid> <its python child pid>    # SIGTERM the running plan
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv   # must be empty before sections 1b/2/3
```

Restart after the slot with the same `chain.sh` command lines (they are in the chain logs under `kcampaign/logs/`).

## 5. Open risks

- GPU 0 is the DSR1 instance's all day; every 8-GPU step waits on `gpu0.proc.lock` (the DSR1 instance must also take it for its own GPU use or the lock protects nothing).
- If the 8-GPU host is a different machine or GPU indices shift (`check_8gpu_machine.sh` WARN lines), the kernel chains' GPU ids and locks must be re-mapped before anything starts.
- GLM-5.2: FP8 KV + FlashMLA sparse at 8 heads per GPU has never run in a real server here; startup and nsys flush times at TP8 x 78 layers are unmeasured (run the smoke early so caches are warm). The repo's `configs/model/glm-5.2-fp8.json` says per-tensor FP8 while the official GLM-5.2-FP8 config is block [128,128]; the serving run uses the official config.
- vLLM 0.30 `bench serve` sends no temperature unless `--temperature` is given (server default, 1.0 with `--generation-config vllm`); the DSR1 runner passes `--temperature 0` like the InferenceX client, the GLM-5.2 runner keeps the server default.
- DeepSeek-R1 TP8: first 61-layer DeepSeek-V3 server on vLLM 0.30 here; startup time, the KV headroom at 0.95 and nsys flush times at TP8 are unmeasured until the smoke.
- FlashInfer mnnvl needs NVSwitch multicast (fabric manager `Completed/Success` today on the 4-GPU view); if unavailable vLLM falls back to trtllm and torch symm-mem is disabled, which changes the dispatch table (the runner records it; nothing is assumed).
