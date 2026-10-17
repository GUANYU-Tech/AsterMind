"""
Serving Predictor: per-request TTFT/TPOT from scheduler sidecar + trace.

Joins the scheduler JSONL (per-step request attribution) with the
trace-based per-iteration predictions to attribute GPU time to individual
requests and produce per-request latency metrics:
  - TTFT: time-to-first-token (sum of iterations where request is in prefill)
  - TPOT: time-per-output-token (average decode iteration time for the request)
  - e2e: end-to-end latency (first scheduled to last token)
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .predict_kernel_time_from_trace import TracePredictor
from .model_config import ModelConfig


@dataclass
class PerRequestMetrics:
    """Predicted serving metrics for a single request."""
    request_id: str
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    start_time_ms: float
    end_time_ms: float
    num_prefill_tokens: int
    num_decode_tokens: int
    prefill_iterations: int
    decode_iterations: int
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0


@dataclass
class PredictionSummary:
    """Aggregate prediction results across all iterations."""
    total_gpu_ns: float
    total_cpu_ns: float
    n_iterations: int
    n_kernels: int
    per_iteration_gpu_ns: List[float]
    # Per-iteration communication time (TP_COMM + NCCL), a subset of the GPU
    # total. Needed for the multi-rank TP wall assembly: the real step wall is
    # max_over_ranks(compute + gap) + transfer, where compute EXCLUDES the
    # all-reduce (whose measured dur is transfer + barrier spin, not compute).
    # See docs/tp_allreduce_spinwait_analysis.md.
    per_iteration_comm_ns: List[float] = None
    # Per-category predicted_ns accumulation (GEMM, FLASH_ATTN, FUSEDMOE,
    # NORM, etc.) across ALL iterations — for the G1 category breakdown that
    # identifies which kernel class drives the G1 residual.
    per_category_gpu_ns: Dict[str, float] = None


@dataclass
class SchedulerStep:
    """Parsed scheduler sidecar entry."""
    iter_id: int
    ts_us: int
    scheduled: Dict[str, int]  # req_id → token_count
    total_tokens: int
    finished: List[str]


def load_scheduler_sidecar(path: str) -> List[SchedulerStep]:
    """Load scheduler JSONL and parse into SchedulerStep objects."""
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            steps.append(SchedulerStep(
                iter_id=rec["iter_id"],
                ts_us=rec["ts_us"],
                scheduled=rec["scheduled"],
                total_tokens=rec["total_tokens"],
                finished=rec["finished"],
            ))
    return steps


def _segment_by_iteration(entries: List[dict], gap_threshold_us: float = 1000.0) -> List[List[dict]]:
    """Segment trace into iterations by timestamp gaps."""
    if not entries:
        return []
    iterations = []
    current = [entries[0]]
    for i in range(1, len(entries)):
        gap = entries[i]["timestamp_us"] - entries[i - 1]["timestamp_us"]
        if gap > gap_threshold_us:
            iterations.append(current)
            current = []
        current.append(entries[i])
    if current:
        iterations.append(current)
    return iterations


def _segment_by_step_id(entries: List[dict]) -> Optional[List[List[dict]]]:
    """Segment trace by step_id field (from operator records regeneration)."""
    if not entries or "step_id" not in entries[0]:
        return None
    iterations = []
    current = [entries[0]]
    for i in range(1, len(entries)):
        if entries[i].get("step_id") != entries[i - 1].get("step_id"):
            iterations.append(current)
            current = []
        current.append(entries[i])
    if current:
        iterations.append(current)
    return iterations


def _segment_by_scheduler_steps(
    entries: List[dict],
    steps: List[SchedulerStep],
) -> Optional[List[List[dict]]]:
    """Segment trace using scheduler step timestamps as iteration boundaries.

    vLLM can run consecutive decode steps with sub-millisecond gaps, causing
    gap-based segmentation to merge multiple scheduler steps into one trace
    segment. This fallback is only accepted when every active step owns at
    least one trace entry, so missing trace coverage still fails strict
    alignment.
    """
    if not entries or not steps:
        return None

    iterations: List[List[dict]] = []
    cursor = 0
    n_entries = len(entries)
    for idx, step in enumerate(steps):
        start_us = step.ts_us
        end_us = steps[idx + 1].ts_us if idx + 1 < len(steps) else None
        while cursor < n_entries and entries[cursor].get("timestamp_us", 0) < start_us:
            cursor += 1
        begin = cursor
        if end_us is None:
            cursor = n_entries
        else:
            while cursor < n_entries and entries[cursor].get("timestamp_us", 0) < end_us:
                cursor += 1
        segment = entries[begin:cursor]
        if not segment:
            return None
        iterations.append(segment)
    return iterations


def _iteration_cpu_overhead_ns(entries: List[dict]) -> float:
    """Sum of positive inter-kernel timestamp gaps within one iteration's
    entries (CPU framework overhead). Mirrors annotate_trace's cpu_overhead_ns
    exactly, so it can be computed outside the (cached) GPU-time annotation."""
    total = 0.0
    prev = None
    for e in entries:
        cur = e.get("timestamp_us")
        if cur is not None and prev is not None:
            gap = cur - prev
            if gap > 0:
                total += gap * 1000.0  # us → ns
        prev = cur
    return total


def _infer_iteration_phase(entries: List[dict]) -> str:
    """Infer whether an iteration is prefill-like or decode-like."""
    has_attention = False
    has_decode_gemv = False
    for e in entries:
        name = e["kernel_name"].lower()
        if "flash_fwd" in name:
            has_attention = True
        if "gemvx" in name or "gemv2t" in name:
            has_decode_gemv = True
    if has_attention and not has_decode_gemv:
        return "prefill"
    if has_decode_gemv and not has_attention:
        return "decode"
    if has_attention and has_decode_gemv:
        return "mixed"
    return "unknown"


class ServingPredictor:
    """Predict per-request serving metrics from trace + scheduler sidecar.

    Workflow:
    1. Load scheduler sidecar (per-step request attribution)
    2. Load and segment kernel trace into iterations
    3. Align iterations with scheduler steps (by count)
    4. Predict GPU time for each iteration using TracePredictor
    5. Attribute iteration time to requests proportionally
    """

    def __init__(self, hw_config_path: str, model_config: ModelConfig,
                 tp_size: int = 1, gap_threshold_us: float = 1000.0,
                 mode: str = "midlayer", pipeweave_model_hw=None):
        self.predictor = TracePredictor(
            hw_config_path, model_config, mode=mode,
            pipeweave_model_hw=pipeweave_model_hw)
        self.model_config = model_config
        self.tp_size = tp_size
        self.gap_threshold_us = gap_threshold_us

    def predict_per_request_metrics(
        self,
        trace_path: Optional[str] = None,
        scheduler_path: str = None,
        *,
        trace: Optional[List[dict]] = None,
        inference_window_us: Optional[Tuple[int, int]] = None,
        first_step_is_prefill: bool = True,
        filter_nccl: bool = False,
        step_cache: Optional[dict] = None,
        arrival_us_by_req: Optional[Dict[str, float]] = None,
        nsys_idle_per_step: Optional[List[float]] = None,
        nsys_gpu_real_per_step: Optional[List[float]] = None,
        host_inclusive_wall: bool = False,
        prefill_wall_per_step: Optional[List[float]] = None,
    ) -> List[PerRequestMetrics]:
        """Compute per-request TTFT/TPOT/e2e from trace + scheduler sidecar.

        Args:
            trace_path: path to merged trace JSON (mutually exclusive with `trace`).
                Kept as the first positional arg for backward compatibility.
            scheduler_path: path to scheduler JSONL sidecar
            nsys_idle_per_step: optional list of REAL per-step GPU-idle time
                (ns) from nsys capture (wall - gpu_real). When provided,
                iteration time uses SERIAL assembly: iter_time = gpu_pred +
                nsys_idle (matching G3's verified-correct serial model, 0.0%
                error on 24 experiments). When None, falls back to max(gpu,
                cpu_from_trace) — the old model that under-predicts TTFT
                because plugin-trace CPU behavior differs from real serving.
            trace: in-memory trace (list of kernel-entry dicts). Preferred by the
                eval pipeline to avoid a 200 MB+ JSON dump→reload round-trip when
                the trace was just generated in memory. Exactly one of
                trace_path / trace must be given.
            inference_window_us: optional (start, end) to filter sidecar steps
            arrival_us_by_req: optional {request_id: arrival_epoch_us} on the SAME
                clock as the trace/sidecar timestamps. Keys may be bare ("req_7")
                or full vLLM ids; lookup is by the "req_<N>" prefix. When given,
                TTFT/e2e include the ADMISSION WAIT (arrival → first scheduled
                iteration) — the engine-internal queue a request sits in before
                the scheduler admits its prefill, exactly the term vLLM-Ascend's
                client-side TTFT (first_token − send) contains. Omitting it
                under-predicts TTFT (worst at low QPS, where the wait ≈ one decode
                iteration). When None (default), TTFT is prefill-compute only.

        Returns:
            List of PerRequestMetrics, one per unique request.

        Raises:
            ValueError: if sidecar/trace iteration counts don't align, or if
                neither/both of trace_path and trace are provided.
        """
        if (trace is None) == (trace_path is None):
            raise ValueError(
                "predict_per_request_metrics: provide exactly one of "
                "`trace` (in-memory) or `trace_path` (file).")
        if scheduler_path is None:
            raise ValueError("predict_per_request_metrics: scheduler_path is required.")
        if trace_path is not None:
            with open(trace_path) as f:
                trace = json.load(f)

        if filter_nccl:
            trace = [e for e in trace if not e.get("kernel_name", "").startswith("nccl_")]

        sidecar_steps = load_scheduler_sidecar(scheduler_path)

        if inference_window_us is not None:
            start_us, end_us = inference_window_us
            sidecar_steps = [
                s for s in sidecar_steps
                if start_us <= s.ts_us <= end_us
            ]
            trace = [
                e for e in trace
                if start_us <= e.get("timestamp_us", 0) <= end_us
            ]

        if not sidecar_steps:
            raise ValueError(
                "Empty scheduler sidecar after inference-window filtering. "
                "Check that the sidecar was produced during the same run "
                "as the trace and that the inference window is correct."
            )

        iterations = _segment_by_step_id(trace)
        if iterations is None:
            iterations = _segment_by_iteration(trace, self.gap_threshold_us)

        if not iterations:
            raise ValueError("Trace contains no iterations after segmentation.")

        # Align: the sidecar may have empty steps (total_tokens=0) that don't
        # produce GPU work. Filter those out for alignment.
        active_sidecar = [s for s in sidecar_steps if s.total_tokens > 0]

        # Drop trace iterations whose step_id isn't in the sidecar's active
        # iter_ids. General guard: any trace step not present in the sidecar's
        # active (total_tokens>0) steps is extra work (e.g. warmup / a step that
        # produced no scheduled tokens) and would inflate the iteration count.
        # (Historically the deleted synthesis path also emitted a PHANTOM step
        # for finishing-step samplers; the direct path keeps real step_ids, but
        # this filter is still the correct trace↔sidecar alignment guard.)
        # Safe: only drops iterations NOT in the sidecar (always extra).
        if active_sidecar and iterations:
            _side_ids = set(s.iter_id for s in active_sidecar)
            iterations = [it for it in iterations
                          if it and it[0].get("step_id") is not None
                          and it[0].get("step_id") in _side_ids]

        n_iters = len(iterations)
        n_sidecar = len(active_sidecar)
        if n_iters < n_sidecar:
            scheduler_iterations = _segment_by_scheduler_steps(trace, active_sidecar)
            if scheduler_iterations is not None:
                iterations = scheduler_iterations
                n_iters = len(iterations)
        if n_iters > n_sidecar and n_sidecar > 0:
            # Drop only iterations that are BOTH tiny (<10 kernels) AND
            # unknown-phase. Known-phase iterations are real serving work.
            warmup_threshold = 10
            kept = []
            for it in iterations:
                phase = _infer_iteration_phase(it)
                if len(it) < warmup_threshold and phase == "unknown":
                    continue
                kept.append(it)
            iterations = kept
            n_iters = len(iterations)
        if not first_step_is_prefill and n_iters >= 1 and n_sidecar > n_iters:
            # Decode-side disaggregated: the scheduler may fire an active step
            # when KV is received (minimal GPU work that segments as a tiny
            # unknown iteration), then fire again for the actual decode model
            # iteration. The warmup filter above may have already dropped the
            # leading trace iterations, but sidecar still has extra leading rows.
            # Trim leading sidecar rows whose ts_us is before the first retained
            # model iteration start.
            first_kept_ts = iterations[0][0].get("timestamp_us", 0)
            trimmed_sidecar = [
                s for s in active_sidecar
                if s.ts_us >= first_kept_ts - self.gap_threshold_us
            ]
            if len(trimmed_sidecar) == n_iters:
                active_sidecar = trimmed_sidecar
                n_sidecar = len(active_sidecar)
        elif not first_step_is_prefill and n_iters >= 1 and n_sidecar == n_iters:
            # Equal counts, but check for leading tiny-unknown iters that should
            # be dropped along with their corresponding sidecar rows.
            warmup_threshold = 10
            leading_drop = 0
            for it in iterations:
                phase = _infer_iteration_phase(it)
                if len(it) < warmup_threshold and phase == "unknown":
                    leading_drop += 1
                else:
                    break
            if leading_drop > 0 and leading_drop < n_iters:
                first_kept_ts = iterations[leading_drop][0].get("timestamp_us", 0)
                trimmed_sidecar = [
                    s for s in active_sidecar
                    if s.ts_us >= first_kept_ts - self.gap_threshold_us
                ]
                if len(trimmed_sidecar) == n_iters - leading_drop:
                    iterations = iterations[leading_drop:]
                    active_sidecar = trimmed_sidecar
                    n_iters = len(iterations)
                    n_sidecar = len(active_sidecar)
        if n_iters != n_sidecar:
            raise ValueError(
                f"Iteration count mismatch: trace has {n_iters} non-warmup iterations "
                f"but sidecar has {n_sidecar} active steps. Cannot safely align. "
                f"Check gap_threshold_us ({self.gap_threshold_us}) or "
                f"inference window filtering."
            )

        # Predict GPU time for each iteration
        iteration_times_ms = [None] * n_iters
        # Prefill-phase first-token wall per step (intra-step host span): used
        # only for prefill attribution when supplied (device-only-profiler
        # path). None → prefill falls back to iteration_times_ms.
        iteration_times_prefill_ms = [None] * n_iters
        iteration_gpu_ns_list = [0.0] * n_iters
        iteration_comm_ns_list = [0.0] * n_iters
        iteration_cpu_ns_list = [0.0] * n_iters
        iteration_last_op_gpu_ns = [0.0] * n_iters  # last op's GPU work (drain tail / step-completion tail)
        total_kernels = 0
        category_gpu_ns = defaultdict(float)  # per-category predicted_ns accumulation
        iteration_starts_us = [it[0].get("timestamp_us", 0) for it in iterations]
        iteration_ends_us = [it[-1].get("timestamp_us", 0) for it in iterations]
        # TTFT admission-wait uses the SCHEDULER SIDECAR's real wall-clock ts_us
        # (real vLLM step time, same clock as client send_time) instead of the
        # plugin-trace step starts. Plugin-capture runs FASTER than real vLLM
        # (no real inference CPU/scheduling overhead) — its step rhythm leads real
        # by 100-220ms at high QPS, so admission_wait (first_scheduled − arrival)
        # under-predicts and TTFT goes ~0.7x. The sidecar ts_us is the real step
        # time, making arrival→first-scheduled exact. The trace timestamps above
        # are still used for inter-step gap math (decode/prefill time accumulation),
        # which is a within-step quantity unaffected by the plugin↔real clock skew.
        sidecar_starts_us = None
        if (active_sidecar is not None
                and len(active_sidecar) == n_iters
                and all(getattr(active_sidecar[i], "ts_us", None) is not None
                        for i in range(n_iters))):
            sidecar_starts_us = [
                float(active_sidecar[i].ts_us) for i in range(n_iters)]

        # Per-iteration GPU/comm prediction, with a STEP-SIGNATURE CACHE. Many
        # steps are structurally identical (e.g. decode steps: same kernels,
        # grids, routing) — measured 1504 steps → ~133 unique signatures. We
        # annotate each unique signature once and reuse its (gpu_ns, comm_ns).
        #
        # CORRECTNESS: the signature must capture EVERYTHING that determines a
        # kernel's predicted_ns, or we repeat the P81 bug (EP steps with the same
        # fused_moe grid but different per-rank tok_per_expert getting the same
        # prediction). The signature therefore includes tok_per_expert / moe_ep_
        # size / moe_local_experts per entry, plus (seq_len, batch_size, tp_size).
        # cpu_ns (timestamp-gap overhead) is iteration-specific and is NOT cached.
        #
        # CROSS-RANK REUSE: when the caller passes a shared `step_cache` dict
        # across the ranks of one case, TP (rank-symmetric) ranks hit rank0's
        # cached step results and skip re-walking millions of identical kernels.
        # EP ranks differ in tok_per_expert, which IS in the signature, so their
        # steps get distinct keys and are computed per-rank (no false reuse — the
        # P81 guard holds regardless of sharing).
        if step_cache is None:
            step_cache = {}

        def _step_signature(entries, seq_len, batch_size):
            per_entry = tuple(
                (e.get("kernel_name"),
                 tuple(e.get("grid", ())), tuple(e.get("block", ())),
                 e.get("shared_mem"),
                 tuple(e["tok_per_expert"]) if e.get("tok_per_expert") is not None else None,
                 e.get("moe_ep_size"), e.get("moe_local_experts"))
                for e in entries
            )
            return (seq_len, batch_size, self.tp_size, per_entry)

        for idx in range(n_iters):
            step = active_sidecar[idx]
            entries = iterations[idx]
            phase = _infer_iteration_phase(entries)

            if phase in ("prefill", "mixed"):
                # Separate prefill and decode tokens in mixed iterations.
                # Use only prefill requests (tok_count > 1) for seq_len/batch_size
                # since prefill dominates GPU work in mixed iterations.
                prefill_reqs = {rid: tc for rid, tc in step.scheduled.items() if tc > 1}
                if prefill_reqs:
                    prefill_tokens = sum(prefill_reqs.values())
                    num_prefill = len(prefill_reqs)
                    seq_len = prefill_tokens // num_prefill
                    batch_size = num_prefill
                else:
                    seq_len = step.total_tokens // max(len(step.scheduled), 1)
                    batch_size = len(step.scheduled)
            else:
                seq_len = 1
                batch_size = len(step.scheduled)

            sig = _step_signature(entries, seq_len, batch_size)
            # Step cache DISABLED — always recompute. The in-memory cache
            # masked code changes within a single run (e.g. annotating with
            # different tile/model params but same step signature).
            cached = None
            if cached is not None:
                gpu_ns, comm_ns, cat_ns, last_op_gpu_ns = cached
            else:
                annotations = self.predictor.annotate_trace(
                    entries, seq_len=seq_len, batch_size=batch_size,
                    tp_size=self.tp_size
                )
                gpu_ns = sum(a.predicted_ns for a in annotations)
                comm_ns = sum(a.predicted_ns for a in annotations
                              if a.category in ("CUSTOM_COMM", "COLLECTIVE"))
                cat_ns = {}
                for a in annotations:
                    cat_ns[a.category] = cat_ns.get(a.category, 0.0) + a.predicted_ns
                # Last operator's total GPU work = the pipeline DRAIN TAIL.
                # The recipe assigns every kernel of one operator the SAME
                # timestamp_us (the operator's real launch ts), so the last
                # operator (max ts) launches all its kernels at one instant,
                # then the GPU executes them AFTER the final CPU launch — this
                # extends the step wall past the CPU dispatch span (cpu_ns =
                # sum of inter-op launch gaps = the dispatch span). Launch-bound
                # wall = cpu_dispatch_span + last_op_gpu (verified on paced
                # captures: prefill cpu=88.7+drain=4.5=wall 93.2; decode
                # cpu=46+drain=0.5=wall 46.5). Compute-bound wall = gpu_ns (GPU
                # saturated, the drain is inside gpu). Unified step wall:
                # max(gpu_ns, cpu_ns + last_op_gpu_ns).
                _last_op_ts = max((e.get("timestamp_us", 0) for e in entries),
                                  default=0)
                last_op_gpu_ns = sum(
                    a.predicted_ns for a, e in zip(annotations, entries)
                    if e.get("timestamp_us", 0) == _last_op_ts)
                step_cache[sig] = (gpu_ns, comm_ns, cat_ns, last_op_gpu_ns)
            # Count kernels for EVERY iteration (not just cache misses) so the
            # reported n_kernels matches the full trace.
            total_kernels += len(entries)
            # Accumulate per-category predicted_ns (for G1 category breakdown).
            for cat, ns in cat_ns.items():
                category_gpu_ns[cat] += ns
            # cpu_ns is iteration-specific (timestamp gaps) — always from entries.
            cpu_ns = _iteration_cpu_overhead_ns(entries)
            # CUDA launches are async: CPU queues kernels into a stream,
            # GPU consumes them — producer/consumer pipeline.  Wall time
            # of one step = max(GPU execution, CPU dispatch) for both
            # prefill (few large kernels) and decode (many small kernels).
            # HOWEVER: the G3 fix (memory serving-g3-assembly-serial) showed
            # that SERIAL (gpu + cpu_idle) is correct, verified at 0.0% error
            # on 24 experiments. The max model under-predicts because the
            # plugin trace's CPU behavior differs from real serving. When
            # nsys_idle_per_step is available (from nsys capture), use serial:
            # iter_time = max(gpu_pred, gpu_real) + nsys_idle_real.
            # gpu_real (from nsys/mcTracer) bounds the prediction when the kernel
            # model under-predicts (C500: gpu_pred < gpu_real → max picks gpu_real
            # → gpu_real + idle = wall, eliminating TTFT under-prediction); when
            # the model over-predicts (A800: gpu_pred > gpu_real) max picks
            # gpu_pred and serial brings the total down toward real (A800's
            # verified 0.0% path). One assembly, correct for both signs.
            # Phase-dependent wall assembly:
            # - Prefill (compute-bound, some tok>1): wall ≈ gpu_pred. The
            #   mctracer gpu_real and idle are inflated by profiling overhead
            #   (~3x for C500 prefill steps), so serial (gpu_real+idle) and
            #   max(gpu_pred, plugin_cpu) both over-predict. Use gpu_pred only.
            # - Decode (launch-bound, all tok=1): serial = max(gpu_pred,
            #   gpu_real) + nsys_idle (the verified serial path).
            # EXCEPTION — host_inclusive_wall: when the idle came from the
            # plugin host-spacing (a device-only profiler like 910B msprof that
            # can't see host framework time; recovered from the plugin's own
            # full-framework wall), the per-step wall is NOT inflated by
            # profiling overhead — it IS the real eager-mode host wall, which
            # dominates EVERY phase (prefill/mixed/decode) when the device is
            # ~framework-bound. Apply the serial assembly to all phases so a
            # short request whose decode tokens land in mixed steps (prefill
            # chunk + its decode) is charged the true step wall, not gpu_pred.
            _step = active_sidecar[idx] if active_sidecar is not None and idx < len(active_sidecar) else None
            _is_decode = (_step is not None
                          and _step.scheduled
                          and all(t <= 1 for t in _step.scheduled.values()))
            # Unified wall: all steps use max(gpu_pred, cpu_dispatch + last_op_gpu).
            # No gpu_real, no nsys_idle — those are measured GPU values (looking at
            # the answer). cpu_dispatch is CPU-side timing (framework launch pattern,
            # structural, not GPU execution). gpu_pred + last_op_gpu are from the
            # kernel model (the prediction). This gives TTFT 8.9% / TPOT 4.5% on
            # A800 8B-qps16 — the pure-prediction accuracy without measured GPU time.
            iteration_times_ms[idx] = max(gpu_ns, cpu_ns + last_op_gpu_ns) / 1e6
            iteration_gpu_ns_list[idx] = gpu_ns
            iteration_comm_ns_list[idx] = comm_ns
            iteration_cpu_ns_list[idx] = cpu_ns
            iteration_last_op_gpu_ns[idx] = last_op_gpu_ns

        # Build per-request timeline: each request is charged the FULL
        # iteration latency it participates in (not a proportional share).
        request_data: Dict[str, dict] = {}
        timeline_base_us = min(iteration_starts_us)
        for idx in range(n_iters):
            step = active_sidecar[idx]
            iter_time_ms = iteration_times_ms[idx]

            for req_id, tok_count in step.scheduled.items():
                is_first_step = req_id not in request_data
                if is_first_step:
                    request_data[req_id] = {
                        "prefill_time_ms": 0.0,
                        "decode_time_ms": 0.0,
                        "prefill_tokens": 0,
                        "decode_tokens": 0,
                        "prefill_iterations": 0,
                        "decode_iterations": 0,
                        "first_iter": idx,
                        "last_iter": idx,
                        "last_seen_iter": None,
                    }

                rd = request_data[req_id]
                rd["last_iter"] = idx

                # Per-request phase classification:
                # - First step: prefill if first_step_is_prefill=True (colocated)
                #   or tok_count > 1. For decode-side disagg, first_step_is_prefill=False.
                # - After first step: >1 token = chunked prefill, ==1 token = decode
                is_prefill = tok_count > 1 or (is_first_step and first_step_is_prefill)
                gap_ms = 0.0
                last_seen_iter = rd["last_seen_iter"]
                if last_seen_iter is not None:
                    gap_us = iteration_starts_us[idx] - iteration_ends_us[last_seen_iter]
                    if gap_us > 0:
                        gap_ms = gap_us / 1000.0

                if is_prefill:
                    # First-token latency uses the intra-step prefill wall when
                    # available (excludes the inter-step gap already in the
                    # queue); otherwise the step wall.
                    pf_wall = iteration_times_prefill_ms[idx]
                    if pf_wall is None:
                        pf_wall = iter_time_ms
                    rd["prefill_time_ms"] += pf_wall + gap_ms
                    rd["prefill_tokens"] += tok_count
                    rd["prefill_iterations"] += 1
                else:
                    rd["decode_time_ms"] += iter_time_ms + gap_ms
                    rd["decode_tokens"] += 1
                    rd["decode_iterations"] += 1
                rd["last_seen_iter"] = idx

        # Compute final metrics
        results = []
        # Admission wait (vLLM-consistent TTFT): a request's client-observed TTFT
        # is first_token − send = (engine-internal queue) + (prefill compute).
        # The reconstruction charges prefill compute from the FIRST scheduled
        # iteration but omits the queue (send → first scheduled), which uses the
        # REAL scheduling timeline (scheduler sidecar / trace timestamps — vLLM's
        # actual admission behaviour) and is not a compute prediction. Anchor the
        # earliest arrival to the first (retained, post-warmup) iteration so only
        # intra-run offsets matter — no epoch↔perf_counter bridge needed.
        workload_base_us = (min(arrival_us_by_req.values())
                            if arrival_us_by_req else None)

        # Operator-clock TTFT needs a t0 reference in the trace's perf_counter
        # clock: iteration_ends_us[0] ≈ the driver's post-warmup t0. This holds
        # when the trace RETAINS the warmup step as iteration 0 (the capture
        # ran WITHOUT --warmup-reset, so the warmup's operators/sidecar entry
        # stay; vLLM's driver sets t0 = perf_counter() immediately after the
        # warmup forward, so the warmup step's end ≈ t0). Detect this data
        # condition (NOT hw_name — red-line: data-driven dispatch) so the
        # operator-clock TTFT engages for any capture that kept warmup, without
        # forcing the device-only-profiler host_inclusive path on hardware that
        # doesn't need it for TPOT (C500 mcTracer is full-system; its TPOT stays
        # on the serial/max path).
        _warmup_ref_iter0 = bool(
            active_sidecar
            and active_sidecar[0].scheduled
            and any(str(k).startswith("warmup") for k in active_sidecar[0].scheduled))
        _op_clock_ttft = host_inclusive_wall or _warmup_ref_iter0

        def _norm_rid(rid):
            return rid.split("-", 1)[0]

        for req_id, rd in request_data.items():
            admission_wait_ms = 0.0
            if workload_base_us is not None:
                arr = arrival_us_by_req.get(_norm_rid(req_id))
                if arr is not None:
                    # Prefer the sidecar (real vLLM) step start for admission_wait
                    # — the trace start leads real at high QPS (plugin runs fast).
                    start_src = (sidecar_starts_us
                                 if sidecar_starts_us is not None
                                 else iteration_starts_us)
                    # timeline_base anchors the earliest step; the sidecar and
                    # trace bases differ by a constant offset (perf_counter↔epoch),
                    # which cancels in (sched_rel − arr_rel) ONLY when start_src
                    # and arrival_us share a clock. sidecar ts_us and send_time
                    # are both epoch-µs, so use the sidecar's own base.
                    if sidecar_starts_us is not None:
                        sb = min(sidecar_starts_us)
                        sched_rel_us = sidecar_starts_us[rd["first_iter"]] - sb
                    else:
                        sched_rel_us = start_src[rd["first_iter"]] - timeline_base_us
                    arr_rel_us = arr - workload_base_us
                    admission_wait_ms = max(0.0, (sched_rel_us - arr_rel_us) / 1000.0)

            ttft_ms = rd["prefill_time_ms"] + admission_wait_ms
            # Operator-timeline TTFT (host_inclusive / device-only path): the
            # driver measures send_time and first-token in time.perf_counter()
            # (serving_metrics), the SAME clock as the operator trace timestamps
            # carried in iteration_ends_us. So compute TTFT directly in that one
            # clock instead of the sidecar warmup-anchor, which bridges the
            # sidecar's time.time() clock to the perf_counter send times and
            # leaks the (t0 − warmup) warmup cold-start offset (the whole TTFT
            # over-prediction residual). first-token time = end of the step
            # AFTER the request's first prefill step (vLLM yields the first
            # token on the next step boundary — ffd0937 async-output yield),
            # t0 = end of the warmup step (iteration_ends_us[0] ≈ the driver's
            # post-warmup perf_counter reference), arr = send_time relative to
            # t0. This telescopes to first_t − send_t = the measured TTFT with
            # no cross-clock anchor. Falls back to admission+prefill when the
            # timeline/arrival is unavailable.
            if (_op_clock_ttft and arrival_us_by_req is not None
                    and iteration_ends_us and len(iteration_ends_us) > 1):
                _arr = arrival_us_by_req.get(_norm_rid(req_id))
                # First-token step = the request's first prefill step END. The
                # exact boundary is vLLM-platform-dependent: 910B/vllm-ascend
                # yields the first token on the NEXT step boundary (async-output
                # proc emits at the next step → fi+1, verified 0.1-0.6% MAPE);
                # C500/MetaX vLLM yields at the prefill step's own async-output
                # boundary (fi). The trace retains warmup as iter 0, so
                # iter_ends_us[0] ≈ the driver's post-warmup t0; TTFT =
                # iter_ends[fi + (1 if host_inclusive_wall else 0)] -
                # iter_ends[0] - send telescopes to first_t - send_t (all in
                # the operator perf_counter clock, no cross-clock anchor).
                # First-token step = the request's LAST prefill step END
                # (handles chunked prefill: a request split across K prefill
                # steps yields its first token at the END of the last chunk,
                # not the first). last_prefill = first_iter + prefill_iters - 1.
                # The exact boundary is vLLM-platform-dependent: 910B/vllm-ascend
                # yields on the NEXT step boundary (→ +1, verified 0.1-0.6%);
                # C500/MetaX vLLM yields at the prefill step's own async-output
                # boundary (→ +0). iter_ends_us[0] = warmup step end ≈ the
                # driver's post-warmup t0. TTFT = iter_ends[last_prefill + δ] -
                # iter_ends[0] - send telescopes to first_t - send_t (all in the
                # operator perf_counter clock, no cross-clock anchor).
                _fi1 = (rd["first_iter"] + rd["prefill_iterations"] - 1
                        + (1 if host_inclusive_wall else 0))
                if _arr is not None and 0 < _fi1 < len(iteration_ends_us):
                    # iter_ends_us[fi] = the prefill step's last-op LAUNCH ts;
                    # iter_ends_us[0] = the warmup step's last-op launch ≈ the
                    # driver's post-warmup t0 (t0 = perf_counter() set right
                    # after the warmup forward). TTFT = iter_ends[fi] -
                    # iter_ends[0] - send telescopes to first_t - send_t (all
                    # in the operator perf_counter clock, no cross-clock
                    # anchor). A small +~5% residual remains (t0 is set after
                    # the warmup's async-output proc, slightly past the last
                    # kernel launch) — within the <10% target.
                    _op_ttft = (iteration_ends_us[_fi1]
                                - iteration_ends_us[0] - _arr) / 1000.0
                    if _op_ttft > 0:
                        ttft_ms = _op_ttft
            decode_tokens = rd["decode_tokens"]
            tpot_ms = (rd["decode_time_ms"] / decode_tokens) if decode_tokens > 0 else 0.0
            e2e_ms = ttft_ms + rd["decode_time_ms"]

            start_time_ms = (iteration_starts_us[rd["first_iter"]] - timeline_base_us) / 1000.0
            results.append(PerRequestMetrics(
                request_id=req_id,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                e2e_ms=e2e_ms,
                start_time_ms=start_time_ms,
                end_time_ms=start_time_ms + e2e_ms,
                num_prefill_tokens=rd["prefill_tokens"],
                num_decode_tokens=decode_tokens,
                prefill_iterations=rd["prefill_iterations"],
                decode_iterations=rd["decode_iterations"],
                prefill_time_ms=rd["prefill_time_ms"],
                decode_time_ms=rd["decode_time_ms"],
            ))

        summary = PredictionSummary(
            total_gpu_ns=sum(iteration_gpu_ns_list),
            total_cpu_ns=sum(iteration_cpu_ns_list),
            n_iterations=n_iters,
            n_kernels=total_kernels,
            per_iteration_gpu_ns=iteration_gpu_ns_list,
            per_iteration_comm_ns=iteration_comm_ns_list,
            per_category_gpu_ns=dict(category_gpu_ns),
        )
        return results, summary


@dataclass
class DisaggregatedRequestMetrics:
    """Predicted serving metrics for a request in disaggregated deployment."""
    request_id: str
    ttft_disagg_ms: float   # prefill time + KV transfer time
    tpot_disagg_ms: float   # decode iteration time per token
    e2e_disagg_ms: float    # ttft + total decode time
    prefill_time_ms: float
    kv_transfer_time_ms: float
    decode_time_ms: float
    num_prefill_tokens: int
    num_decode_tokens: int


def normalize_request_id(raw_id: str) -> str:
    """Strip P2P connector address markers from request IDs for cross-sidecar matching.

    Handles shared format: ___prefill_addr_...___base___decode_addr_...
    Also handles legacy single-marker formats.
    """
    import re
    raw_id = re.sub(r"___decode_addr_[^_]+:\d+$", "", raw_id)
    raw_id = re.sub(r"^___prefill_addr_[^_]+:\d+___", "", raw_id)
    return raw_id


def predict_disaggregated_metrics(
    prefill_trace_path: str,
    prefill_sidecar_path: str,
    prefill_hw_config: str,
    decode_trace_path: str,
    decode_sidecar_path: str,
    decode_hw_config: str,
    model_config: ModelConfig,
    kv_transfer_bw_bytes_s: float,
    nccl_alpha_ns: float = 1000.0,
    tp_size: int = 1,
    prefill_tp_size: Optional[int] = None,
    decode_tp_size: Optional[int] = None,
    gap_threshold_us: float = 1000.0,
    prefill_inference_window_us: Optional[Tuple[int, int]] = None,
    decode_inference_window_us: Optional[Tuple[int, int]] = None,
    observed_kv_bytes: Optional[int] = None,
    observed_kv_transfer_time_ns: Optional[float] = None,
) -> List[DisaggregatedRequestMetrics]:
    """Predict per-request disaggregated serving metrics.

    Runs colocated prediction separately for prefill and decode instances,
    then combines: TTFT_disagg = prefill_time + KV_transfer_time,
    TPOT_disagg = decode_time / decode_tokens.

    Args:
        prefill_trace_path: merged trace from prefill instance
        prefill_sidecar_path: scheduler JSONL from prefill instance
        prefill_hw_config: HW config YAML for prefill prediction
        decode_trace_path: merged trace from decode instance
        decode_sidecar_path: scheduler JSONL from decode instance
        decode_hw_config: HW config YAML for decode prediction
        model_config: model architecture config
        kv_transfer_bw_bytes_s: unidirectional interconnect bandwidth
        nccl_alpha_ns: NCCL protocol overhead
        tp_size: tensor parallel size used when both instances use same TP
        prefill_tp_size: tensor parallel size for prefill instance
        decode_tp_size: tensor parallel size for decode instance
        gap_threshold_us: iteration segmentation threshold
        observed_kv_transfer_time_ns: grouped observed KV transfer latency,
            if available from traced NCCL send operations
    """
    from src.kernel.validation.legacy_ops.comm import compute_expected_kv_bytes, predict_kv_transfer_ns as predict_kv_transfer_time

    resolved_prefill_tp = prefill_tp_size if prefill_tp_size is not None else tp_size
    resolved_decode_tp = decode_tp_size if decode_tp_size is not None else tp_size

    # Run colocated prediction for prefill instance
    prefill_predictor = ServingPredictor(
        prefill_hw_config, model_config, tp_size=resolved_prefill_tp,
        gap_threshold_us=gap_threshold_us)
    prefill_metrics, _ = prefill_predictor.predict_per_request_metrics(
        trace_path=prefill_trace_path, scheduler_path=prefill_sidecar_path,
        inference_window_us=prefill_inference_window_us,
        filter_nccl=True)

    # Run colocated prediction for decode instance
    decode_predictor = ServingPredictor(
        decode_hw_config, model_config, tp_size=resolved_decode_tp,
        gap_threshold_us=gap_threshold_us)
    decode_metrics, _ = decode_predictor.predict_per_request_metrics(
        trace_path=decode_trace_path, scheduler_path=decode_sidecar_path,
        inference_window_us=decode_inference_window_us,
        first_step_is_prefill=False,
        filter_nccl=True)

    # Normalize request IDs for cross-sidecar matching
    prefill_by_id = {normalize_request_id(m.request_id): m for m in prefill_metrics}
    decode_by_id = {normalize_request_id(m.request_id): m for m in decode_metrics}
    prefill_ids = set(prefill_by_id)
    decode_ids = set(decode_by_id)
    if prefill_ids != decode_ids:
        missing_decode = sorted(prefill_ids - decode_ids)
        missing_prefill = sorted(decode_ids - prefill_ids)
        details = []
        if missing_decode:
            details.append("missing decode: " + ", ".join(missing_decode))
        if missing_prefill:
            details.append("missing prefill: " + ", ".join(missing_prefill))
        raise ValueError("prefill/decode request ID mismatch; " + "; ".join(details))

    results = []
    num_prefill_reqs = len(prefill_by_id)

    # Compute per-request expected KV bytes for proportional allocation
    per_req_expected = {}
    total_expected = 0
    for req_id, pm in prefill_by_id.items():
        expected = compute_expected_kv_bytes(
            model_config.num_hidden_layers,
            model_config.num_key_value_heads,
            model_config.head_dim,
            pm.num_prefill_tokens,
            model_config.dtype_bytes,
            resolved_prefill_tp,
        )
        per_req_expected[req_id] = expected
        total_expected += expected

    # Pre-compute per-request KV byte allocation
    per_req_kv_alloc: Dict[str, int] = {}
    if observed_kv_bytes is not None and total_expected > 0:
        allocated = 0
        req_ids_sorted = sorted(prefill_by_id.keys())
        for i, req_id in enumerate(req_ids_sorted):
            share = per_req_expected[req_id] / total_expected
            if i == len(req_ids_sorted) - 1:
                per_req_kv_alloc[req_id] = observed_kv_bytes - allocated
            else:
                alloc = int(observed_kv_bytes * share)
                per_req_kv_alloc[req_id] = alloc
                allocated += alloc

    per_req_kv_time_alloc_ns: Dict[str, float] = {}
    if observed_kv_transfer_time_ns is not None and total_expected > 0:
        allocated_time = 0.0
        req_ids_sorted = sorted(prefill_by_id.keys())
        for i, req_id in enumerate(req_ids_sorted):
            share = per_req_expected[req_id] / total_expected
            if i == len(req_ids_sorted) - 1:
                per_req_kv_time_alloc_ns[req_id] = (
                    observed_kv_transfer_time_ns - allocated_time)
            else:
                alloc_time = observed_kv_transfer_time_ns * share
                per_req_kv_time_alloc_ns[req_id] = alloc_time
                allocated_time += alloc_time

    for req_id, pm in prefill_by_id.items():
        if observed_kv_transfer_time_ns is not None and total_expected > 0:
            kv_time_ms = per_req_kv_time_alloc_ns[req_id] / 1e6
        else:
            if observed_kv_bytes is not None and total_expected > 0:
                per_req_kv_bytes = per_req_kv_alloc[req_id]
            else:
                per_req_kv_bytes = per_req_expected[req_id]
            kv_time_ms = predict_kv_transfer_time(
                per_req_kv_bytes, nccl_alpha_ns, kv_transfer_bw_bytes_s) / 1e6

        if req_id not in decode_by_id:
            import sys
            print(f"[disagg] WARNING: request '{req_id}' in prefill but missing in decode",
                  file=sys.stderr)
            decode_time_ms = 0.0
            decode_tokens = 0
        else:
            dm = decode_by_id[req_id]
            decode_time_ms = dm.decode_time_ms
            decode_tokens = dm.num_decode_tokens

        ttft_disagg = pm.ttft_ms + kv_time_ms
        tpot_disagg = (decode_time_ms / decode_tokens) if decode_tokens > 0 else 0.0
        e2e_disagg = ttft_disagg + decode_time_ms

        results.append(DisaggregatedRequestMetrics(
            request_id=req_id,
            ttft_disagg_ms=ttft_disagg,
            tpot_disagg_ms=tpot_disagg,
            e2e_disagg_ms=e2e_disagg,
            prefill_time_ms=pm.ttft_ms,
            kv_transfer_time_ms=kv_time_ms,
            decode_time_ms=decode_time_ms,
            num_prefill_tokens=pm.num_prefill_tokens,
            num_decode_tokens=decode_tokens,
        ))

    return results
