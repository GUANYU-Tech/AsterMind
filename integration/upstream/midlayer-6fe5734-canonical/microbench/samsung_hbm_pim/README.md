# Samsung HBM-PIM microbenchmarks

HBM-PIM has no silicon in this lab.  Its ground-truth instrument is the UniNDP
(HPCA25) cycle-accurate simulator at `$UNINDP_ROOT`
(default `/share_data/yangxinhao/chip_modeling/UniNDP`), which stands in the
same relation to this hardware that ncu does to A800 and msprof to 910B: it is
where measurements come from, not a model to copy.  Everything here submits
shapes to that instrument and reads latencies back.

| Path | What it measures |
|---|---|
| `unindp_probe.py` | Generic sweep driver: runs one UniNDP primitive per shape in a plan file and records the reported latency and launch counters. Simulator scratch lands in `profiler_output/samsung_hbm_pim/unindp_probe_ws`. |
| `unindp_timing_sensitivity.py` | Attributes each bank-stream cost to a named DDR command timing, by perturbing one timing at a time in an isolated copy of the architecture configuration and re-measuring the differential cost of one column / one operand-buffer boundary / one bank-row boundary. The real checkout is never modified. |
| `memory/elewise_sweep.csv` | Element-wise latency against element count, from one sub-column instance to 1024 columns per unit. Resolves the plain-column, operand-buffer-boundary and bank-row-boundary step costs. |
| `operators/layernorm_softmax_sweep.csv` | The same sweep for the normalization and softmax primitives, plus the aspect-ratio scan showing both depend only on the total element count. |
| `compute/mm_m_sweep.csv` | Matrix-product latency against the row count at seven fixed (K, N) pairs. Isolates the result-delivery constant and bounds the simulator's partition-quantization residual. |
| `references/bank_protocol_timing_sensitivity.txt` | Recorded output of `unindp_timing_sensitivity.py` at the canonical configuration: the evidence table behind every protocol composite in `configs/hardware/samsung_hbm_pim.yaml`. |

Each driver pins its subprocesses to a single thread, so the worker count is
the core budget.  Invocation flags mirror the canonical dataset pipeline
(`configs/dataset/samsung_hbm_pim.yaml`): architecture `hbm-pim`, top-k 1,
predictor-aided search on.
