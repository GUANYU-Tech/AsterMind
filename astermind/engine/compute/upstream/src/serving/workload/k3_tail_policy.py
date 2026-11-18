"""Fixed H200 vLLM0.30 target defaults, not host CPU environment settings.

latent_moe_runner.py sha256 f28044a7f0b1ddb56e783d328fb9b684a08d37381e0afc2589e920f53499edac
Source env defaults: shared stream enabled; threshold 256. Overrides are not
represented by this fixed-source policy and must not be guessed.
"""
DEFAULT_POLICY = "vllm_0_30_h200_source_defaults"

def resolve_k3_tail(tier, rows, *, policy=DEFAULT_POLICY):
    if type(rows) is not int or rows <= 0:
        raise ValueError("K3 tail requires positive actual fused-output rows")
    if policy != DEFAULT_POLICY:
        raise NotImplementedError("K3 automatic tail requires its fixed target-source defaults; unknown target env/policy")
    if tier not in (None, "ALLREDUCE_OVERLAP", "COLUMN_PARALLEL"):
        raise ValueError("K3 tail must be an explicit supported tier or source default")
    if tier is not None:
        return tier
    return "ALLREDUCE_OVERLAP" if rows <= 256 else "COLUMN_PARALLEL"
