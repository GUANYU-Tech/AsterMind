#!/usr/bin/env python3
"""Build the 4-wide (Mb,Nb,Kb,depth) A800 serving GEMM tile LUT, EXACT 5-key.

Why this exists (Phase 2, T-D)
------------------------------
The H-layer direct-pricing path has no synthesized kernel name, so the tile
(Mb,Nb,Kb,depth) must come from a shape-keyed lookup. This LUT is that lookup:
  Key   = "[M_tok, N_out, K, has_bias, dtype]"   (EXACT, no pad64)
  Value = [Mb, Nb, Kb, depth]  (gemm.py convention; depth may be None for
                                cutlass_relu names, which _parse_pipeline_stages
                                also can't parse → name mode gets None too)

SOURCE = gemm_kernel_lut_a800.json (the 5-key exact table build_gemm_kernel_lut.py
already correlates from serving nsys, picking the DOMINANT cuBLAS kernel per
(M,N,K,has_bias,dtype)). We parse each stored kernel name ONCE with gemm.py's
own _full_tile (== _parse_tile_dims + _parse_pipeline_stages) and store the
4-tuple. So a lut4 hit returns byte-for-byte what name mode would parse from
that kernel → lut4 ≡ name on every seen serving shape, by construction.

Why not re-correlate nsys directly (the T-D 53dd712 attempt): an s16816-only
nsys filter + TC-filtered op set shifts the per-step index when non-s16816
GEMM-family kernels (gemvx/xmma) interleave, mis-pairing 826 shapes and picking
a different (wrong) majority. gemm_kernel_lut uses all-family + its own
_consistent gate + per-position has_bias — the canonical correlation name mode
actually queries — so deriving from it inherits exactly that.

§6.1's "54% pad64 collision" was under pad64 keys (merging different exact
shapes); EXACT 5-key has 0 collision (one kernel per key in gemm_kernel_lut).

Output: src/serve_trace/gemm_tile_lut4_a800.json
Usage:  python scripts/serve_predict/build_gemm_tile_lut4_a800.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.kernel.validation.legacy_ops.tile_strategy import _full_tile  # noqa: E402

KERNEL_LUT = ROOT / "src" / "serve_trace" / "gemm_kernel_lut_a800.json"
LUT4_OUT = ROOT / "src" / "serve_trace" / "gemm_tile_lut4_a800.json"


def main():
    raw = json.load(open(KERNEL_LUT))
    serial = {}
    skipped = 0
    for key, entries in raw.items():
        if key == "_meta":
            continue
        if not isinstance(entries, list) or not entries:
            skipped += 1
            continue
        name = entries[0][0] if isinstance(entries[0], (list, tuple)) else entries[0]
        ft = _full_tile(name)
        if ft is None:
            skipped += 1
            continue
        mb, nb, kb, depth = ft
        serial[key] = [int(mb), int(nb),
                       (int(kb) if kb is not None else None),
                       (int(depth) if depth is not None else None)]
    LUT4_OUT.write_text(json.dumps(serial, indent=0, sort_keys=True))
    print(f"derived {len(serial)} exact-5-key entries from gemm_kernel_lut "
          f"({KERNEL_LUT.name}); skipped {skipped} (no parseable tile)")
    print(f"wrote {LUT4_OUT}")


if __name__ == "__main__":
    main()

