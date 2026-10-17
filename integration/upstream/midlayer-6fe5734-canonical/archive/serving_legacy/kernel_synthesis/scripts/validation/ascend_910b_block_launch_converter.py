#!/usr/bin/env python3
"""Build a per-(kernel_name, shape) -> block_dim LUT from op_profile.json.

The regen emits aclnn kernels with hardcoded Block Dims (rotary=11, Cat=48,
RmsNorm=48...). CANN auto-tiles per-shape, so the real Block Dim varies.
op_profile.json (distilled from msprof) has `OpName|"shape"|bd` -> details.
This restructures it into {kernel_name: {shape_str: block_dim}} for the regen
to look up by constructing the same shape string from op_attr.

Usage: build_block_lut_910b.py --case-dir <dir>  (writes block_lut.json)
"""
import json
import sys
from pathlib import Path


def build(case_dir):
    case_dir = Path(case_dir)
    src = case_dir / "op_profile.json"
    if not src.exists():
        return 0
    d = json.load(open(src))
    # {kernel_name: {shape_str: block_dim}}  (highest-count bd wins on conflict)
    lut = {}
    counts = {}  # {name: {shape: {bd: count}}}
    for key, v in d.items():
        if not isinstance(v, dict):
            continue
        nm = v.get("op_name")
        sh = v.get("input_shapes")
        bd = v.get("block_dim")
        c = v.get("count", 0)
        if nm is None or sh is None or bd is None:
            continue
        counts.setdefault(nm, {}).setdefault(sh, {})[bd] = (
            counts.setdefault(nm, {}).setdefault(sh, {}).get(bd, 0) + c)
    for nm, shapes in counts.items():
        lut[nm] = {}
        for sh, bds in shapes.items():
            # highest-count bd (deterministic; CANN gives one bd per shape)
            lut[nm][sh] = max(bds, key=lambda b: bds[b])
    # Some op_profile shapes collide across kernels that share the aclnn name +
    # input_shapes but get DIFFERENT Block Dims — notably the QK-norm
    # aclnnContiguous_Slice: q-norm (32 heads) and k-norm (8 heads) both
    # serialize to "{tokens},48,128;3;3" (the 48 is a CANN tiling param, NOT the
    # head count) but q-norm gets the HIGH bd (more data) and k-norm the LOW bd.
    # Emit per-shape LO/HI maps so the caller can pick by the head count it knows
    # from the real input shape.
    lut["__lo__"] = {nm: {sh: min(bds) for sh, bds in shapes.items()}
                     for nm, shapes in counts.items()}
    lut["__hi__"] = {nm: {sh: max(bds) for sh, bds in shapes.items()}
                     for nm, shapes in counts.items()}
    out = case_dir / "block_lut.json"
    json.dump(lut, open(out, "w"), indent=0)
    return len(lut)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    a = ap.parse_args()
    n = build(a.case_dir)
    print(f"[block_lut] {n} kernels -> {a.case_dir}/block_lut.json")


if __name__ == "__main__":
    main()
