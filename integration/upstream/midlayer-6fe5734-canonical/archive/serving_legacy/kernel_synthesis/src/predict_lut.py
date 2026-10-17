"""Prediction LUT cache with code-version detection.

Caches kernel prediction results keyed by (kernel_name, grid, block,
shared_mem, category, seq_len, batch_size, tp_size, routing_sig). Persists
to disk as JSON with a SHA256 code-version header — any change to prediction-
relevant source files or the C extension invalidates the cache.

routing_sig carries the fused-MoE per-expert token histogram (tok_per_expert).
Under expert parallelism the same fused_moe_kernel entry has an identical grid
on every rank (grid is derived from total tokens, not per-rank routing), so
without routing_sig rank1 would hit rank0's cached value and every EP rank
would predict the same latency. Including the routing histogram in the key
keeps per-rank MoE load asymmetry (the whole point of EP) distinct in the
cache. It is None for non-MoE kernels, leaving their keys unchanged.
"""

import hashlib
import gzip
import pickle
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]

# On-disk cache format tag. predict_lut.py is NOT in _CODE_FILES, so editing
# this file does NOT bump compute_code_version(); bumping this constant is how
# we invalidate LUT files written by an older key/serialization scheme. v2:
# native-tuple keys + pickle payload (v1 was md5(repr)-hex keys + JSON).
_KEY_FORMAT = "v2-tuple-pickle"

_CODE_FILES = [
    "src/kernel/ops/common.py",
    "src/kernel/ops/gemm.py",
    "src/kernel/ops/fused_moe.py",
    "src/kernel/ops/flash_attn.py",
    "src/kernel/ops/norm.py",
    "src/kernel/ops/comm.py",
    "src/kernel/formula.py",
    "src/kernel/kernel_base.py",
    "src/kernel/lru_cache.py",
    "src/kernel/hardware_spec.py",
    "src/serve_predict/predict_kernel_time_from_trace.py",
    # ModelConfig dataclass + loader: changes here alter model params available
    # to the recipe (e.g. max_position_embeddings drives the IndexSelect LUT key)
    # so they must invalidate the regen cache.
    "src/serve_predict/model_config.py",
    # Recipe / trace-expansion code: changes here alter the regenerated trace
    # (which sub-kernels an op emits, e.g. FA splitkv_combine), so they must
    # invalidate both the LUT and the regen cache keyed on compute_code_version().
    "src/serve_trace/analytic_mirror.py",
    "src/serve_trace/operator_records.py",
    # SyntheticKernelEntry factories (_make_rotary/_make_norm/_moe_* etc.) that
    # decide which sub-kernels each op expands into — same rationale as above
    # (e.g. Llama rope → ELEMENTWISE vs vLLM ROTARY changes the trace).
    "src/dse/model_sim.py",
]

_BINARY_FILES = [
    "build/lru_sim.so",
]


def compute_code_version() -> str:
    h = hashlib.sha256()
    for rel in sorted(_CODE_FILES):
        p = _ROOT / rel
        if p.exists():
            h.update(rel.encode())
            h.update(p.read_bytes())
    for rel in sorted(_BINARY_FILES):
        p = _ROOT / rel
        if p.exists():
            h.update(rel.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _make_key(kernel_name: str, grid: list, block: list,
              shared_mem: int, cat: str, seq_len: int,
              batch_size: int, tp_size: int, routing_sig=None):
    """Native hashable key (tuple) — used directly as a dict key.

    Previously this md5-hashed repr() of the tuple into a hex string; the hash
    was pure overhead (called once per kernel, ~9.3M times on a big case →
    ~10× slower than keying on the tuple itself, benchmarked) and added a
    (negligible but real) collision surface. Python dicts hash tuples of
    hashable atoms directly, so the mapping is identical and lossless. grid /
    block / routing_sig are tuple-ified so lists and tuples collide correctly
    and the key stays hashable.
    """
    return (kernel_name, tuple(grid), tuple(block), shared_mem,
            cat, seq_len, batch_size, tp_size,
            tuple(routing_sig) if routing_sig is not None else None)


class PredictionLUT:

    def __init__(self, code_version: str):
        self.version = code_version
        self._cache: dict = {}
        self._hits = 0
        self._misses = 0

    def get(self, kernel_name: str, grid: list, block: list,
            shared_mem: int, cat: str, seq_len: int,
            batch_size: int, tp_size: int, routing_sig=None) -> Optional[float]:
        key = _make_key(kernel_name, grid, block, shared_mem,
                        cat, seq_len, batch_size, tp_size, routing_sig)
        val = self._cache.get(key)
        if val is not None:
            self._hits += 1
        else:
            self._misses += 1
        return val

    def put(self, kernel_name: str, grid: list, block: list,
            shared_mem: int, cat: str, seq_len: int,
            batch_size: int, tp_size: int, predicted_ns: float,
            routing_sig=None):
        key = _make_key(kernel_name, grid, block, shared_mem,
                        cat, seq_len, batch_size, tp_size, routing_sig)
        self._cache[key] = predicted_ns

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses,
                "entries": len(self._cache)}

    def save(self, path: Path):
        # gzip(level=1) + pickle. Pickle because tuple keys aren't JSON-
        # serializable and the value dict is huge (up to ~3.4M entries). gzip-L1
        # because the keys are highly redundant (128-element MoE routing tuples
        # repeat across steps): it shrinks the TP4EP4 LUT 1122 MB→30 MB (37×)
        # for only ~3s extra dump and ~3s extra load — a large disk win at
        # negligible time cost. protocol 5 for fast large-container pickling.
        data = {
            "code_version": self.version,
            "key_format": _KEY_FORMAT,
            "entries": self._cache,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            pickle.dump(data, f, protocol=5)
        tmp.replace(path)  # atomic: never leave a half-written LUT on crash

    @classmethod
    def load(cls, path: Path, current_version: str) -> Optional["PredictionLUT"]:
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, OSError, EOFError, ValueError, gzip.BadGzipFile):
            # Unreadable, or an old JSON-format LUT (v1, not gzip). Rebuild.
            return None
        if not isinstance(data, dict) or data.get("key_format") != _KEY_FORMAT:
            # Older key/serialization scheme — keys are incompatible; rebuild.
            return None
        stored_version = data.get("code_version", "")
        if stored_version != current_version:
            print(f"  [predict] LUT version mismatch "
                  f"(stored={stored_version[:8]}.. vs current={current_version[:8]}..)"
                  f", regenerating...")
            return None
        lut = cls(current_version)
        lut._cache = data.get("entries", {})
        return lut
