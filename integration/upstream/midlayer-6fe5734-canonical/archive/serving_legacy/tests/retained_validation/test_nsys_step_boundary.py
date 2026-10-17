"""Tests for nsys per-step boundary alignment (timestamp-based, replacing gap
detection). Locks in the serving-ttft-nsys-step-boundary-bug fix: a mixed
chunked-prefill step with intra-step gaps must NOT be fragmented."""
import sys, sqlite3, json, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.serve_predict.build_eval_dataset import (
    _partition_by_ts_boundaries, _ts_boundaries_from_sidecar,
    _extract_nsys_per_step, _detect_step_boundaries)


def _k(start, dur):
    """Build a kernel dict (start ns, end ns, dur_ns ns)."""
    return {"start": start, "end": start + dur, "dur_ns": dur}


def test_partition_by_ts_boundaries_basic():
    # kernels: step0 [10..30, 40..50], step1 [110..140], step2 [210..220, 230..240]
    kernels = [
        _k(10, 20), _k(40, 10),    # step 0: gpu=30, wall=40, intra=10
        _k(110, 30),               # step 1: gpu=30, wall=30, intra=0; inter=110-50=60
        _k(210, 10), _k(230, 10),  # step 2: gpu=20, wall=30, intra=10; inter=210-140=70
    ]
    bounds = [0, 100, 200, 300]
    steps = _partition_by_ts_boundaries(kernels, bounds)
    assert len(steps) == 3
    assert steps[0]["gpu_ns"] == 30 and steps[0]["cpu_ns"] == 10          # intra only (first step)
    assert steps[1]["gpu_ns"] == 30 and steps[1]["cpu_ns"] == 60         # inter=110-50=60
    assert steps[2]["gpu_ns"] == 20 and steps[2]["cpu_ns"] == 80         # intra=10 + inter=210-140=70


def test_ts_boundaries_from_sidecar():
    # sidecar ts_us epoch; nsys0=first kernel start (ns); tail = last kernel end.
    bounds = _ts_boundaries_from_sidecar(
        sidecar_ts_us=[1000, 2000, 3500], nsys0_ns=500, last_kernel_end_ns=4000)
    # B[i] = 500 + (ts[i]-1000)*1000
    assert bounds[0] == 500
    assert bounds[1] == 500 + 1000 * 1000      # 1_000_500
    assert bounds[2] == 500 + 2500 * 1000      # 2_500_500
    assert bounds[3] >= 4000                     # tail past last kernel


def test_big_step_with_internal_gaps_not_fragmented():
    """The regression: a mixed chunked-prefill step has intra-step gaps larger
    than inter-decode gaps. Gap detection SPLITS it (wrong); timestamp alignment
    keeps it whole (correct)."""
    # step 0: big prefill, 3 kernel groups with 5us internal gaps (the trap)
    # step 1: tiny decode
    # step 2: tiny decode  (decode-decode gap is 1us, smaller than prefill's 5us internal)
    kernels = [
        _k(0, 10000), _k(15000, 10000), _k(30000, 10000),   # step0: 3 chunks, 5us+5us gaps inside
        _k(50000, 500),                                      # step1 decode (gap 20us before = boundary)
        _k(51000, 500),                                      # step2 decode (gap 1us — sub-ms, the trap)
    ]
    n_steps = 3
    # Gap detection: the 3 largest gaps are 20000 (30k->50k), 15000 (15k->30k), 5000 (0->... no).
    # The 3 largest gaps sorted: 20000, 15000, 5000(two of them). Picking 2 boundaries (n_steps-1=2):
    # largest two gaps = 20000, 15000 → cuts after the 2nd and 3rd kernel of step0 → FRAGMENTS step0.
    gap_bounds = _detect_step_boundaries(kernels, n_steps)
    gap_gpu = [sum(k["dur_ns"] for k in kernels[s:e]) for s, e in gap_bounds]
    # Gap detection fragments step 0: its gpu is split across "steps", so the
    # detected step-0 gpu (10000) < the real 30000ns.
    assert gap_gpu[0] < 30000, "gap detection should fragment step 0 (the bug)"

    # Timestamp alignment: sidecar ts gives the real step windows. step0=[0,50), step1=[50,51), step2=[51,..).
    # Map sidecar ts_us (e.g. 0, 50, 51) with nsys0=0 (first kernel at 0).
    bounds = _ts_boundaries_from_sidecar(
        sidecar_ts_us=[0, 50, 51], nsys0_ns=0, last_kernel_end_ns=51500)
    ts_steps = _partition_by_ts_boundaries(kernels, bounds)
    assert ts_steps[0]["gpu_ns"] == 30000   # step 0 WHOLE (3 chunks × 10000)
    assert ts_steps[1]["gpu_ns"] == 500
    assert ts_steps[2]["gpu_ns"] == 500


def test_extract_nsys_per_step_timestamp_path(tmp_path):
    """End-to-end: a mock nsys sqlite + sidecar → _extract_nsys_per_step with
    sidecar_ts_us recovers the real per-step gpu (no fragmentation)."""
    # Build a mock sqlite matching the real schema.
    db = tmp_path / "nsys_profile.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER, demangledName INTEGER,
            gridX INTEGER, gridY INTEGER, gridZ INTEGER,
            blockX INTEGER, blockY INTEGER, blockZ INTEGER,
            staticSharedMemory INTEGER, dynamicSharedMemory INTEGER);
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
    """)
    conn.execute("INSERT INTO StringIds VALUES (1, 'kernel_a')")
    # step 0 (ts=0us): big prefill, 3 chunks with 5us gaps; step 1 (ts=60us): decode.
    rows = [
        (0, 10000, 0, 1), (15000, 25000, 0, 1), (30000, 40000, 0, 1),  # step0
        (60000, 60500, 0, 1),                                            # step1
    ]
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start,end,deviceId,demangledName) VALUES (?,?,?,?)",
        rows)
    conn.commit(); conn.close()

    # sidecar ts_us: step0 at 0, step1 at 60 (us). nsys0 = first kernel = 0ns.
    # (ts_us*1000 would be ns; here ts_us are already the ns-scale boundaries we intend,
    #  so pass them as the sidecar timestamps — the function uses (ts[i]-ts[0])*1000.)
    per_step = _extract_nsys_per_step(tmp_path, n_steps=2, sidecar_ts_us=[0, 60])
    assert per_step is not None
    assert len(per_step) == 2
    # step 0 gpu = 30000ns = 30us = 0.03ms (NOT fragmented)
    assert abs(per_step[0]["nsys_gpu_ms"] - 0.03) < 1e-6, per_step[0]["nsys_gpu_ms"]
    # step 1 gpu = 500ns = 0.0005ms
    assert abs(per_step[1]["nsys_gpu_ms"] - 0.0005) < 1e-6, per_step[1]["nsys_gpu_ms"]


def test_extract_nsys_falls_back_without_sidecar(tmp_path):
    """When sidecar_ts_us is None, fall back to gap detection (backward compat)."""
    db = tmp_path / "nsys_profile.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, demangledName INTEGER);
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO StringIds VALUES (1,'k');
    """)
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start,end,deviceId,demangledName) VALUES (?,?,?,?)",
        [(0, 1000, 0, 1), (2000, 2500, 0, 1)])
    conn.commit(); conn.close()
    per_step = _extract_nsys_per_step(tmp_path, n_steps=2, sidecar_ts_us=None)
    assert per_step is not None and len(per_step) == 2
