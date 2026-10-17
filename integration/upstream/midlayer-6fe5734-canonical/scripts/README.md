# scripts/ 索引

`scripts/` 只放两类入口：本地薄 CLI（解析参数、调用 `src/`、`evaluation/` 或 `dataset_pipeline/` 的实现）与在目标硬件上执行的 profiler workload `*_runner.py`。活动代码不得 import `scripts`；脚本被仓内其他文件引用与否不作为存活依据——多数 runner 与采集 CLI 由操作员手动或在远端硬件上运行。下表按目录分组列出全部入口及各自 docstring/usage 首行；主门禁入口是 `scripts/run_tests.py`（唯一测试入口）与 `scripts/evaluation/run_standard_evaluation_report.py`（标准 Kernel+Serving 报告）。

### `scripts/`

| 脚本 | 用途 |
|---|---|
| `measure_external_vllm_scheduler_service.py` | Measure a container's installed vLLM Scheduler without model execution. |
| `measure_vllm_scheduler_service.py` | Capture an independently reproducible native-vLLM Scheduler CPU profile. |
| `run_serving_simulation.py` | CLI for the production no-GPU serving simulation. |
| `run_tests.py` | Run repository tests with the canonical local Python environment. |
| `view_branch.sh` | Show local branches with their worktree, ahead/behind and dirty state. |

### `scripts/baselines/pipeweave/`

| 脚本 | 用途 |
|---|---|
| `benchmark_converter.py` | Convert benchmark-runner JSON into PipeWeave training CSVs. |
| `benchmark_runner.py` | Run optional PipeWeave-specific operator benchmarks on target hardware. |
| `kernel_dataset_converter.py` | Convert canonical Midlayer kernel datasets into PipeWeave training CSVs. |
| `model_trainer.py` | Train hardware-specific PipeWeave MLPs from generated CSV datasets. |

### `scripts/collective_selection/`

| 脚本 | 用途 |
|---|---|
| `build_hccl_observations.py` | Validate one HCCL INFO run and write normalized selection facts. |
| `build_mccl_observations.py` | Validate one MCCL mcTracer run and write normalized selection facts. |
| `build_nccl_observations.py` | Validate one NCCL profiler run and write normalized selection facts. |
| `validate_collective_rules.py` | Validate normalized collective observations against selection rules. |

### `scripts/evaluation/`

| 脚本 | 用途 |
|---|---|
| `run_full_evaluation.py` | CLI for the cross-hardware full evaluation. |
| `run_nvidia_h200_pd_glm_max_requests_sweep.sh` | Sweep H200 PD max-requests for the GLM models (thin wrapper of run_nvidia_h200_pd_sweep.py). |
| `run_nvidia_h200_pd_kimi_max_requests_sweep.sh` | Sweep H200 PD max-requests for the Kimi models (thin wrapper of run_nvidia_h200_pd_sweep.py). |
| `run_standard_evaluation_report.py` | Update the canonical Kernel/full-Serving report from existing evaluators. |

### `scripts/evaluation/kernel/`

| 脚本 | 用途 |
|---|---|
| `ascend_910b_dram_derivation_audit.py` | CLI for the Ascend 910B DRAM-derivation audit. |
| `check_prediction_order.py` | Thin CLI for deterministic three-hardware kernel order checks. |
| `gemm_launch_comparison_report.py` | Write the observed-versus-injected GEMM launch comparison report. |
| `kernel_golden_report.py` | CLI for deterministic kernel-golden sampling and Event comparison. |
| `kernel_latency_scatter_report.py` | CLI for the cross-hardware kernel latency scatter report. |
| `kernel_mechanism_validation_report.py` | CLI for the cross-hardware kernel mechanism-validation report. |
| `kernel_memory_traffic_report.py` | CLI for the cross-hardware kernel memory-traffic scatter report. |
| `rank_gemm_mechanism_clusters.py` | Rank current GEMM Forward error by static launch-program mechanism. |
| `run_kernel_ablation.py` | CLI for the kernel predictor ablation study. |
| `run_kernel_evaluation.py` | CLI for single-kernel evaluation. |

### `scripts/evaluation/serving/`

| 脚本 | 用途 |
|---|---|
| `agentx_session_replay.py` | AgentX-derived multi-turn session replay (KV host-offload group, DSR1 TP8). |
| `aggregate_nvidia_h200_nsys_window_kernels.py` | Read-only aggregation of one nsys window (sqlite export), device 0, for the family tables. |
| `analyze_inferencex_deepseek_v3_req10_protocol.py` | Old (3 req/client) vs InferenceX-protocol (10 req/client) DSV3 8K/1K errors. |
| `ascend_910b_qwen3_4b_tp1_qps16_reduction.py` | Run the audited 910B Qwen3-4B QPS16 timing reduction. |
| `assemble_nvidia_h200_host_profiles.py` | Assemble the four H200 Host timing profiles for one vLLM version. |
| `build_ground_truth_metrics_candidates.py` | Build candidate ShareGPT metrics from production ground-truth results. |
| `build_metax_c500_mctracer_sync_boundaries.py` | Create one C500 native mcTracer structural-boundary sidecar for a case. |
| `build_nvidia_h200_sharegpt_candidate.py` | Convert one H200 ShareGPT capture into a compact Serving case candidate. |
| `build_nvidia_h200_sharegpt_candidate_root.py` | Build one H200 ShareGPT candidate dataset root for one vLLM version. |
| `build_pd_compact_candidate.py` | Build one immutable 1P1D Serving compact candidate from measured raw data. |
| `build_production_candidates.py` | Build strict local Production request or target-alignment candidates. |
| `build_sharegpt_candidates.py` | CLI for building compact ShareGPT Serving candidates. |
| `build_slow_trace_candidate.py` | Build a provenance-bearing, uniformly slowed production-trace window. |
| `capture_metax_c500_sharegpt_runner.py` | Capture one C500 ShareGPT cell with a frozen request stream and optional hooks. |
| `capture_nsys.py` | CLI for capturing an Nsight Systems serving trace. |
| `compare_nvidia_h200_decode_kernel_families.py` | Compare one simulated H200 DSV3-L32 TP4 engine step with real nsys families. |
| `compare_nvidia_h200_deepseek_v3_l32_real_vs_sim.py` | Compare real 4x H200 DeepSeek-V3-L32 TP4 closed-loop runs with the simulator. |
| `compare_nvidia_h200_fa3_geometries.py` | Per-geometry FA3 MLA predicted vs real per-call durations on H200 (wp10). |
| `compare_nvidia_h200_fp8_gemm_calls.py` | Per-call comparison of H200 block-FP8 dense GEMMs: Midlayer vs real nsys. |
| `compare_nvidia_h200_fused_moe_fp8_calls.py` | Per-call H200 FP8 Triton fused-MoE prediction versus real trace durations. |
| `compare_trace_with_nsys.py` | CLI for comparing a simulated trace with Nsight Systems. |
| `derive_nvidia_h200_runtime_memory_policy.py` | Derive the H200 runtime memory policies from real vLLM startup logs. |
| `extract_nvidia_h200_attention_kernel_calls.py` | Extract real rank-0 attention-family kernel calls from H200 nsys traces. |
| `extract_nvidia_h200_fp8_gemm_real_calls.py` | Extract real H200 FP8 block-scaled dense GEMM calls from nsys GPU traces. |
| `extract_nvidia_h200_fused_moe_trace_calls.py` | Extract H200 Triton fused-MoE launch geometry and per-call durations. |
| `mctracer_to_nsys_converter.py` | CLI for converting mcTracer observations to the Nsight schema. |
| `metax_c500_qwen3_4b_tp1_qps16_reduction.py` | Run the audited C500 Qwen3-4B QPS16 timing reduction. |
| `msprof_summary_converter.py` | CLI for converting msprof operator summaries. |
| `msprof_timeline_converter.py` | CLI for converting msprof timelines. |
| `nvidia_h200_deepseek_r1_nsys_decode_capture.sh` | Capture pure-decode nsys windows of the real vLLM DeepSeek-R1 server on H200 |
| `nvidia_h200_deepseek_r1_nsys_prefill_capture.sh` | Capture one prefill-only nsys window of the real vLLM DeepSeek-R1 server on |
| `nvidia_h200_deepseek_r1_rank_proxy_config.py` | Write a single-GPU DeepSeek-R1 model directory with the per-rank shapes of TP=N. |
| `nvidia_h200_deepseek_r1_tp8_capture_runner.py` | DeepSeek-R1 TP8 real-serving capture on 8x H200 with vLLM 0.30 (dummy weights). |
| `nvidia_h200_deepseek_r1_tp8_day2.sh` | DeepSeek-R1 on 8x H200, 2026-09-27 8-GPU window (owner: wt-dsr1-mtpoff). |
| `nvidia_h200_deepseek_r1_tp8_offload_serve.sh` | vLLM 0.30 DeepSeek-R1-0528 TP8 server for the KV host-offload group (RUNBOOK_8GPU.md 3c). |
| `nvidia_h200_deepseek_r1_tp8_serve.sh` | Real vLLM 0.30 DeepSeek-R1-0528 server on 8x H200, TP8, dummy weights. |
| `nvidia_h200_export_nsys_windows.sh` | sqlite-export every window_N.nsys-rep of one capture directory and write the |
| `nvidia_h200_glm_5_2_tp8_capture_runner.py` | GLM-5.2 TP8 real-serving capture on 8x H200 with vLLM 0.30 (dummy weights). |
| `nvidia_h200_glm_5_2_tp8_prebuild_flashinfer.py` | CPU-only prebuild of the FlashInfer 0.6.18 JIT module GLM-5.2 needs on SM90. |
| `nvidia_h200_glm_5_2_tp8_serve.sh` | Real vLLM 0.30 GLM-5.2 server on 8x H200, TP8, dummy weights. |
| `nvidia_h200_memory_ledger.py` | Emit a reproducible projected H200 PD per-rank memory ledger. |
| `nvidia_h200_nsys_prefill_capture.sh` | Capture prefill-only windows of the real vLLM DSV3-L32 TP4 server on 4x H200. |
| `nvidia_h200_nsys_pure_decode_capture.sh` | Capture pure-decode windows of the real vLLM DSV3-L32 TP4 server on 4x H200. |
| `nvidia_h200_serve_deepseek_r1.sh` | Real vLLM 0.24 DeepSeek-R1 architecture (dummy FP8 block-128 weights) on H200. |
| `nvidia_h200_tp8_serving_capture.py` | Real-serving TP8 capture on 8x H200 with vLLM 0.30 (dummy weights). |
| `plot_inferencex_deepseek_v3_raw_alignment.py` | Plot DSV3 H200 8K/1K simulation against InferenceX per-config raw points. |
| `production_trace_report.py` | Render a projected production-Trace arrival-rate sweep as Markdown. |
| `run_inferencex_deepseek_v3_1k128_closedloop_comparison.py` | Closed-loop colocated vs PD DeepSeek-V3 1K/128 sweeps with MTP off/on. |
| `run_inferencex_deepseek_v3_8k1024_closedloop_comparison.py` | Closed-loop colocated vs PD DeepSeek-V3 8K/1K sweeps with MTP off/on. |
| `run_inferencex_glm_1k128_mtp5_closedloop_comparison.py` | Closed-loop colocated vs PD GLM-5.2 1K/128 sweeps with MTP off/on. |
| `run_inferencex_glm_1k128_mtp5_pd_comparison.py` | Compare colocated and PD GLM-5.2 1K/128 sweeps with MTP off/on. |
| `run_inferencex_glm_1k128_mtp5_pd_scaleout_comparison.py` | Compare colocated and scale-out PD GLM-5.2 1K/128 sweeps with MTP off/on. |
| `run_inferencex_glm_8k1024_mtp5_closedloop_comparison.py` | Closed-loop colocated vs PD GLM-5.2 8K/1K sweeps with MTP off/on. |
| `run_inferencex_kimi_1k128_closedloop_comparison.py` | Closed-loop colocated vs PD Kimi-K2.6 1K/128 sweeps, MTP off only. |
| `run_inferencex_kimi_8k1024_closedloop_comparison.py` | Closed-loop colocated vs PD Kimi-K2.6 8K/1K sweeps, MTP off only. |
| `run_inferencex_style_glm_1k128_concurrency_sweep.py` | Sweep first-wave client concurrency for the 1K/128 GLM-5.2 H200 smoke setup. |
| `run_inferencex_style_glm_1k128_mtp_sweep.py` | Compare GLM-5.2 1K/128 concurrency with MTP disabled/enabled. |
| `run_inferencex_style_glm_1k128_smoke.sh` | One InferenceX-style GLM-5.2 1K/128 H200 simulation smoke run (concurrency 4). |
| `run_nvidia_h200_deepseek_v3_l32_tp4_closedloop_sim.py` | Simulate the real 4x H200 DeepSeek-V3-L32 TP4 closed-loop runs. |
| `run_nvidia_h200_pd_sweep.py` | Run a bounded H200 proxy sweep and render its metric-contract plots. |
| `run_pd_disaggregation_study.py` | Run fair colocated-versus-PD ShareGPT simulation experiments. |
| `run_production_ground_truth_comparison.py` | Compare Serving simulation aggregates with production ground truth. |
| `run_serving_metric_evaluation.py` | CLI for serving-metric evaluation and aggregation. |
| `run_sharegpt_compact_evaluation.py` | Run aggregate-only ShareGPT compact-suite evaluation. |
| `simulation_runner_entrypoint.py` | Spawn-safe command-line entry point for the Serving simulator. |
| `summarize_nvidia_h200_nsys_decode_windows.py` | Summarise nsys decode windows of the real vLLM DSV3-L32 TP4 server. |
| `validate_serving_trace.py` | CLI for validating a serving trace. |
| `validate_sharegpt_compact_alignment.py` | Validate all compact ShareGPT cases and run structural alignment. |
| `validate_target_model_blocks.py` | Run source-pinned tiny GLM-5.2 and Kimi-K2.6 CUDA block validation. |
| `validate_trace_alignment.py` | CLI for validating serving-trace alignment. |
| `write_nvidia_h200_sharegpt_suites.py` | Write the three H200 ShareGPT suite configs of one vLLM version. |

### `scripts/evaluation/serving/diagnostics/`

| 脚本 | 用途 |
|---|---|
| `analyze_ascend_910b_selected_step_msprof.py` | Analyze a same-run 910B selected-step msprof capture. |
| `analyze_metax_c500_selected_step_mctracer.py` | Correlate a validated C500 selected-step capture with mcTracer tasks. |
| `analyze_nvidia_a800_selected_step_nsys.py` | Extract correlated A800 selected-step CUDA Graph intervals from NSYS. |
| `audit_scheduler_depth_metadata.py` | Audit the complete Scheduler-depth metadata-only migration candidate. |
| `build_metax_c500_moe_duration_decomposition.py` | Build the C500 TP2/EP2 decode MoE same-launch duration decomposition. |
| `compare_dataset_serving_inputs.py` | Compare exact Dataset and Serving predictor inputs for three hardware. |
| `inspect_runtime_prepare_occurrences.py` | Inspect source-authored runtime-prepare occurrence signatures for one cell. |
| `materialize_selected_step.py` | Write complete C_step rankings and one selected-step injection input. |
| `migrate_scheduler_depth_metadata.py` | Create a complete Scheduler-depth metadata candidate for one hardware. |
| `replay_selected_step_injection.py` | Derive a full prediction fingerprint for one injected Scheduler step. |
| `report_ascend_910b_host_residual.py` | Build a reproducible, non-fitting 910B eager-fallback attribution report. |
| `report_ascend_910b_sharegpt_step_decomposition.py` | Summarize 910B ShareGPT step/device/Host decomposition. |
| `run_admission_sensitivity_sweep.py` | Sweep a controlled admission shift across selected Serving cells. |
| `run_graph_padding_paired_steps.py` | Price one case's observed steps under exact and real graph capture sets. |
| `run_kernel_coverage.py` | Build a Serving-to-Kernel coverage census for one hardware. |
| `run_oracle_ttft_reconstruction.py` | Reconstruct client TTFT from an absolute-clock Scheduler capture. |
| `run_scheduler_depth_breakdown.py` | Run the exact Scheduler-depth TTFT/TPOT signed error breakdown. |
| `run_scheduler_error_propagation.py` | Measure how dynamic Scheduling propagates synthetic step-duration errors. |
| `run_scheduler_pipeline_depth_study.py` | Run the 45-case native-vs-depth-one Scheduler capacity study. |
| `select_extrapolation_kernels.py` | Select high-contribution Serving kernels for extrapolation checks. |
| `selected_step_runner.py` | Run a deterministic pure-decode selected-step probe on vLLM hardware. |
| `summarize_extrapolation_benchmarks.py` | Compare selected Serving predictions with independent Event results. |
| `validate_scheduler_alignment.py` | Run aggregate-only Scheduler alignment for an explicit suite subset. |
| `validate_selected_step_injection.py` | Validate a selected-step hardware injection against its canonical state. |

### `scripts/evaluation/serving/scheduler/`

| 脚本 | 用途 |
|---|---|
| `ascend_910b_canonical_trajectory_counterfactual.py` | Run the formal 910B canonical-trajectory counterfactual. |
| `ascend_910b_same_trajectory_reduction.py` | Run the audited 910B matched-trajectory timing reduction. |
| `build_scheduler_capture_candidate.py` | Build one schema-v3 Serving case from an accepted Scheduler capture. |
| `compare_scheduler_trace_ab.py` | Compare trace-off and trace-on Serving ground-truth runs. |
| `publish_scheduler_capture_candidates.py` | Publish fully validated Scheduler capture candidates to one dataset root. |
| `reconcile_scheduler_capture_suite.py` | Validate and summarize a complete Scheduler/Graph Serving capture suite. |
| `report_scheduling_trajectory_replay.py` | Plot Scheduling Trajectory Replay and optional A/B/C/D attribution results. |
| `run_scheduling_trajectory_replay.py` | Run Scheduling Trajectory Replay for one canonical observed Scheduler trace. |

### `scripts/experiments/`

| 脚本 | 用途 |
|---|---|
| `ascend_910b_mooncake_device_transfer_runner.py` | Verify one Ascend Mooncake NPU-to-NPU KV-style transfer without host staging. |
| `ascend_910b_serving_alignment_runner.py` | Capture one real Ascend 910B vLLM eager-prefill window for kernel alignment. |
| `compare_pd_observer_effect.py` | Write strict paired trace-off/trace-on PD observer-effect evidence. |
| `deploy_nvidia_h200_serving_capture.sh` | Usage: deploy_nvidia_h200_serving_capture.sh  (from a clean checkout; prints the snapshot dir) |
| `execute_pd_capture_plan.py` | Execute one materialized real 1P1D PD capture plan. |
| `freeze_nvidia_a800_pd_source_suite.py` | Freeze the current 15-case A800 ShareGPT source contract for PD capture. |
| `materialize_pd_capture_plan.py` | CLI for creating one frozen ShareGPT 1P1D PD capture plan. |
| `metax_c500_mccl_device_transfer_runner.py` | Validate one direct MCCL transfer between two MetaX C500 devices. |
| `metax_c500_mooncake_device_transfer_runner.py` | Run one isolated same-host Mooncake transfer on two MetaX C500 devices. |
| `nvidia_a800_serving_alignment_runner.py` | Capture one real A800 vLLM eager-prefill window for kernel alignment. |
| `nvidia_h200_serving_alignment_runner.py` | Capture one H200 ShareGPT alignment raw case (operators + Scheduler + Nsight). |
| `request_router.py` | CLI for the OpenAI-compatible evaluation request router. |
| `run_nvidia_h200_sharegpt_capture.py` | Run the full H200 ShareGPT capture for one lane of cases (stdlib only). |
| `run_serving_experiments.py` | CLI for manifest-driven serving experiments. |
| `run_sharegpt_ground_truth.py` | CLI for canonical ShareGPT production Serving ground truth. |
| `run_sharegpt_pd_ground_truth.py` | CLI for one frozen ShareGPT 1P1D Serving ground-truth benchmark. |

### `scripts/hardware/`

| 脚本 | 用途 |
|---|---|
| `nvidia_h200_yaml_edit.py` | Comment-preserving edits of the H200 hardware yaml used by the H200 port. |
| `set_yaml_block_values.py` | Set scalar keys inside one top-level-indented block of a hardware YAML. |

### `scripts/kernel_dataset/`

| 脚本 | 用途 |
|---|---|
| `fa_serving_capture_ascend_910b_full_runner.py` | Capture frozen Ascend 910B FA cases as Timeline and op-mode evidence. |
| `fa_serving_capture_audit_runner.py` | Create fail-closed candidate audit artifacts for FA serving capture. |
| `fa_serving_capture_metax_c500_full_runner.py` | Capture frozen FA serving candidates on MetaX C500 one case at a time. |
| `fa_serving_capture_metax_c500_timing_audit.py` | Emit C500 FA Event/trace timing evidence when counters are unavailable. |
| `fa_serving_capture_nvidia_a800_full_runner.py` | Capture frozen A800 FA cases as NSYS windows plus NCU raw-page evidence. |
| `fa_serving_capture_report.py` | Write final candidate-only FA serving publication-audit artifacts. |
| `fa_serving_capture_runner.py` | Validate and materialize frozen cross-hardware FA serving-capture inputs. |
| `fa_serving_forward_report.py` | Run Forward accuracy on the published FA serving capture datasets. |
| `fa_serving_release.py` | Publish one hardware's FA serving capture evidence dataset. |
| `merge_fa_serving_release.py` | Merge one published FA serving campaign into the canonical FA dataset. |

### `scripts/kernel_dataset/ascend_910b/`

| 脚本 | 用途 |
|---|---|
| `attach_gemm_launch_observations_runner.py` | Attach duration-free CANN launch observations to a GEMM Dataset. |
| `build_dataset.py` | CLI for building the Ascend 910B kernel dataset. |
| `event_timer_sanity_runner.py` | Validate Ascend 910B ACL-backed NPU Event timing without a profiler. |
| `flash_attn_runner.py` | FlashAttention runner for 910B dataset profiling. |
| `fused_moe_runner.py` | Standalone vLLM-Ascend fused-MoE runner for 910B dataset profiling. |
| `gemm_runner.py` | GEMM runner for 910B dataset profiling (torch.matmul on NPU). |
| `kernel_golden_runner.py` | Run the fixed no-profiler Ascend 910B kernel-golden sample serially. |
| `profile_parallel.sh` | Parallel msprof-op profiling across the 8 Ascend 910B NPUs. |
| `vector_runner.py` | Run one Ascend 910B Vector workload or a persistent Event batch. |

### `scripts/kernel_dataset/metax_c500/`

| 脚本 | 用途 |
|---|---|
| `audit_full_event_capture_runner.py` | CLI for the MetaX C500 full OP Event campaign audit. |
| `build_dataset.py` | CLI for building the MetaX C500 kernel dataset. |
| `build_full_event_campaign_manifest_runner.py` | CLI for assembling one immutable MetaX C500 Event campaign manifest. |
| `build_full_event_candidate_runner.py` | CLI for the candidate-only MetaX C500 987-case Event dataset builder. |
| `build_full_event_selection_runner.py` | CLI for building the audited MetaX C500 Device Event selection. |
| `collect_process_isolation_snapshot_runner.py` | CLI for collecting one complete MetaX C500 host process snapshot. |
| `event_timer_sanity_runner.py` | Validate MetaX C500 current-stream Event timing without a profiler. |
| `flash_attn_runner.py` | Serving-aligned MetaX C500 FlashAttention profiler runner. |
| `fused_moe_runner.py` | Standalone vLLM-equivalent MetaX fused-MoE runner without importing vLLM. |
| `fused_moe_vllm024_runner.py` | Capture vLLM-metax 0.24 FusedMoE launch geometry on C500. |
| `gemm_runner.py` | Run the canonical MetaX Linear/GEMM workload with device Events. |
| `kernel_golden_graph_runner.py` | Measure canonical C500 cases with graph-amortized device Events. |
| `kernel_golden_runner.py` | Run the fixed no-profiler C500 kernel-golden sample serially. |
| `profile_all.sh` | Run full mcProfiler + mcTracer profiling for all kernel types. |
| `profile_fused_moe_parallel.sh` | Compatibility entry point for segmented fused-MoE profiling. |
| `run_event_campaign_attempt_runner.py` | CLI for one auditable MetaX C500 Device Event attempt. |
| `vector_runner.py` | MetaX C500 Serving-aligned vector operator runner. |

### `scripts/kernel_dataset/nvidia_a800/`

| 脚本 | 用途 |
|---|---|
| `build_dataset.py` | CLI for building the NVIDIA A800 kernel dataset. |
| `event_timer_sanity_runner.py` | Validate NVIDIA A800 CUDA Event timing without a profiler. |
| `flash_attn_runner.py` | FlashAttention runner for the A800 single-op NCU dataset. |
| `fused_moe_runner.py` | Fused-MoE runner for the A800 single-op NCU dataset (vLLM fused_experts). |
| `gemm_runner.py` | GEMM runner for the A800 single-op NCU dataset (torch.matmul on CUDA). |
| `kernel_golden_runner.py` | Run the fixed no-profiler NVIDIA A800 kernel-golden sample serially. |
| `vector_runner.py` | Serving-aligned A800 Vector runner for NCU and no-profiler CUDA Events. |

### `scripts/kernel_dataset/nvidia_h200/`

| 脚本 | 用途 |
|---|---|
| `build_dataset.py` | CLI for NVIDIA H200 kernel-dataset candidates (plans / assemble / check / publish). |
| `flash_attn_runner.py` | NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/flash_attn_runner.py``. |
| `fused_moe_runner.py` | NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/fused_moe_runner.py``. |
| `gemm_runner.py` | NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/gemm_runner.py``. |
| `kc_deepseek_r1_decode_runner.py` | H200 kcampaign runner for the DeepSeek-R1/V3 decode-path operators vLLM 0.24 launches. |
| `kc_dsa_cache_write_runner.py` | NVIDIA H200 DeepSeek Sparse Attention (DSA) KV-cache write runner. |
| `kc_dsa_indexer_runner.py` | NVIDIA H200 DeepSeek Sparse Attention (DSA) indexer runner. |
| `kc_dsa_sparse_mla_runner.py` | NVIDIA H200 DeepSeek Sparse Attention (DSA) sparse-MLA attention runner. |
| `kc_k3_mla_runner.py` | NVIDIA H200 Kimi-K3 gated-MLA decode runner (vLLM 0.30.0 only). |
| `kc_kda_runner.py` | NVIDIA H200 Kimi Delta Attention (KDA) kernel runner. |
| `kc_v4attn_runner.py` | NVIDIA H200 DeepSeek-V4 attention-component runner (CSA / HCA / SWA, |
| `kc_wq_moe_runner.py` | NVIDIA H200 kernel-dataset runner for weight-only-quantized (W4A16) MoE. |
| `nsys_campaign_runner.py` | Remote NSYS campaign for the H200 kernel dataset (A800 canonical protocol). |
| `vector_runner.py` | NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/vector_runner.py``. |

### `scripts/kernel_dataset/samsung_hbm_pim/`

| 脚本 | 用途 |
|---|---|
| `build_dataset.py` | CLI for building the Samsung HBM-PIM kernel dataset. |

### `scripts/launch_data/`

| 脚本 | 用途 |
|---|---|
| `ascend_910b_launch_generator.py` | CLI for validating or generating Ascend 910B launch data. |
| `build_gemm_workload_candidates.py` | Write the aligned ShareGPT GEMM workload universe as candidate data. |
| `metax_c500_launch_generator.py` | CLI for validating or generating MetaX C500 launch-selection CSV. |
| `nvidia_a800_launch_generator.py` | CLI for validating or generating NVIDIA A800 launch-selection CSV. |
| `nvidia_h200_launch_generator.py` | CLI for validating or generating NVIDIA H200 launch-selection CSV. |

### `scripts/paper_plot/cal26/`

| 脚本 | 用途 |
|---|---|
| `collect_data.py` | Collect every data input behind the CAL'26 paper figures and tables. |
| `plot_figures.py` | Draw the CAL'26 paper data figures from one collect_data.py run. |
