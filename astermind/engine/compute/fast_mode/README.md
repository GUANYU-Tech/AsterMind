# Fast Mode

Select `"compute_mode": "fast_mode"` in the usual scenario and launch with the usual OrbitServe CLI. The editor and reports use Fast Mode. Existing experiment JSON with the provisional mode ID is normalized on input. The scenario's existing `midlayer` and `roofline` choices retain their KernelPlan-based paths. Internal execution_mode continues to select how the original kernel path runs.

The module starts from the existing Cell's scheduled step and actual prompt lengths. The scheduled step supplies the per-request query count, computed context and DP rank. Cell startup resolves the model file, layer partition, parallel groups and rank-to-GPU placement once. The batch executor validates the aggregate and per-DP token counts, applies the model-family Roofline formula and quotes the corresponding collective and P2P work using the current Network Frame. It returns completion on the existing timeline and through the existing Scheduler/KV callbacks.

## Implementation boundaries

| Code | Responsibility |
| --- | --- |
| `executor.py` | Native Cell adapter. Binds the deployment, consumes a Scheduler step, and publishes batch completion on the existing timeline. |
| `predictor.py`, `mla_gqa.py`, `communication.py` | Model and parallel layout. Determine tensor shapes, per-layer FLOPs and HBM bytes, and the rank operations, payloads and phase order. |
| `hardware.py` | Device-side Roofline rates. Converts the selected hardware specification and timing policy to effective tensor throughput by precision and a DRAM read rate used for grouped memory traffic. |
| `admission.py`, `source_memory.py`, `v3_residency.py` | Validate the selected model/layout/precision and fit per-rank weights, workspace and native KV allocation within the target HBM capacity. |
| `comm_pricing.py`, `quote.py`, `comm_reuse.py`, `comm_direct.py` | Bind rank tasks to physical GPU paths and reuse eligible quotes. The existing COMM_Backend profiles own the interconnect rules. |

The executor prices the ordered model phases, with rank groups, byte semantics, and occurrence counts retained in the timing program. It records one batch completion and reports compute and communication separately. Within a Frame, the static quote treats transfers as isolated. When a batch spans Frames, the executor keeps the quote if it can verify that the paths and timing conditions remain valid. Otherwise it releases the original rank operations in phase order to the existing communication runtime, where active flows may share capacity. Transfers outside the Cell continue to use the existing network engine.

`frame_forecast.py` reuses V1's Frame projection on a fork of the network state. The live frame provider, timeline, and COMM_Backend implementation are shared with the detailed modes. In a mixed non-H200 wave, Fast Mode submits the planner's local GPU messages as individual SendRecv operations so V1 uses the selected device profile. Cross-satellite messages still use the same runtime and shared resources. Every wave waits for its messages before releasing the next one.

At Cell startup the grouped predictor takes its HBM bandwidth from the selected
hardware profile's calibrated DRAM read capacity. It uses that rate as a single
proxy for grouped memory traffic. The H200 profile gives about 4.62 TB/s, and the
CLI applies that rate directly. The 70% factor applies to BF16 and FP8 tensor
compute throughput. COMM_Backend owns the separate link timing and contention policy.

The fast path supports colocated H200 and selected A800, Ascend 910B and MetaX C500 configurations with SP1 and no KV migration. DP greater than one requires explicit WideEP, Direct All-to-All, and the existing source and Scheduler admission. Model shape and per-rank HBM checks apply. The H200 reuse path can bind supported TP4/TP8 paths and Direct All-to-All paths on a fixed topology. Frame providers can also opt in if their paths depend only on the current immutable Frame. Each batch supplies its actual payload sizes. The record field `communication_pricing` distinguishes `reused`, `partial_reuse`, `isolated`, and `runtime`. Portable profiles currently use the isolated quote path with the selected hardware's communication model.

Model assumptions include grouped Roofline, synthetic length-only feedback from the existing vLLM Scheduler, and uniform expert routing where the formula uses that approximation. The model uses one execution duration per batch and omits PP microbatch overlap and per-batch KernelPlan construction. The native CPU vLLM adapter determines support for chunked prefill and cross-request prefix caching.

Build the native helpers with `python -m astermind.common.build_native`. The vLLM 0.30 adapter requires the pinned vLLM source and Python environment via `ORBITServe_VLLM030_SOURCE` and `ORBITServe_VLLM030_PYTHON`. The H200 admission path also requires an explicit rank-memory projection for some model families. Select the applicable policy file with `ORBITServe_H200_EXPERIMENTAL_MEMORY_PROJECTION` or, for DeepSeek V4, `ORBITServe_V4_EXPERIMENTAL_MEMORY_PROJECTION`.

On A800, 910B and C500, the bundled GLM-5.2 BF16 profile can explicitly select `fast_glm_dsa_score: "bf16_timing_proxy"` in a Cell. This uses the target BF16 compute peak for DSA index-score timing and keeps the source model's packed index-cache byte count. On 910B and C500, the bundled Kimi K2 Group-32 INT4 configurations require an explicit per-Cell `memory_policy_file` tied to the model and hardware hashes, sharding, and workload limits. These forecasts use the chosen board's hardware profile and experimental workspace reserve.

On the three non-H200 boards, the pinned Kimi K3 W4A16 reference also requires a per-Cell memory policy. The pinned GLM-5.3, DeepSeek V3, V3-L32, V4 Pro and V4 Flash profiles can explicitly select the `fp8-widened-bf16-reference` memory policy. This reference retains packed weights and source KV layouts and uses target BF16 tensor rates for FP8 compute groups. It omits the cost of widening, block scaling and intermediate tile traffic. This option supports functional simulation and does not establish a vendor-native FP8 timing profile.
