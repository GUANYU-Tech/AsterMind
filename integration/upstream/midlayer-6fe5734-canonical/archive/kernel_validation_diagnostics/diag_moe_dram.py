#!/usr/bin/env python3
"""Compare Bug-B DRAM options vs measured dram_read_bytes across FUSEDMOE
prefill (large TOK) cases, to pick the physically-correct overflow model."""
import sys, os, math
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
from evaluation.kernel.dataset import load_config, load_kernel_dataset
from evaluation.kernel.profiler_observation import extract_ncu_metrics
from tools.kernel.validation.reference_models.fused_moe import _ascend_grouped_projection_traffic, predict_moe_tile_rows
from src.kernel.hardware_spec import HWSpec
import numpy as np

cfg = load_config("configs/kernel_dataset/ascend_910b.yaml"); raw = load_kernel_dataset(cfg)
hwp = os.path.join(cfg["modeling"]["hw_config"], f"{cfg['modeling']['hw_name']}.yaml")
hw = HWSpec(hwp)
l2_cap = hw.l2_size_bytes
tile_rows = hw.memory_subsystem.get("gemm_weight_reread_tile_rows")

print(f"{'TOK':>5} {'H':>5} {'INT':>5} {'proj':>4} {'W_comp':>8} {'wpe':>7} {'frac':>5} {'wrr':>8} {'dram_curr':>9} {'dram_fix':>8} {'dram_meas':>9} {'xbar':>8}")
for k, v in raw.items():
    oi = v.get("op_info", {})
    if oi.get("Type") != "FUSEDMOE": continue
    a = oi["Attr"]; B=a["TOK"]; K=a["H"]; N=a["INT"]; topk=a["TOPK"]; E=a["E"]
    if B < 2048: continue
    tpe = a.get("tok_per_expert") or []
    rows = float(sum(tpe)) if tpe else float(B*topk)
    counts = list(tpe) if tpe else None
    e_active = sum(1 for c in counts if c>0) if counts else min(E,max(1,int(rows)))
    for kd in v["kernels"]:
        ncu = extract_ncu_metrics(kd)
        if "groupedmatmul" not in ncu["kernel_name"].lower(): continue
        kl = ncu["kernel_name"].lower()
        if kl.endswith("_w1"):
            W = float(e_active*K*(2*N)*2); wpe=float(K*(2*N)*2)
        elif kl.endswith("_w2"):
            W = float(e_active*N*K*2); wpe=float(N*K*2)
        else:
            W = float(e_active*K*N*2); wpe=float(K*N*2)
        A = float(rows*K*2); C = float(rows*N*2)
        ascend = getattr(hw,"aic_aiv_separated",False)
        ramp = (lambda c: predict_moe_tile_rows(c,K,N)) if ascend else None
        l2_total, _ = _ascend_grouped_projection_traffic(ncu["kernel_name"],rows,e_active,K,N,
                    counts=counts, tile_rows=tile_rows, tile_rows_fn=ramp)
        # weight reread bytes = l2_total - (A+C)*2 ... l2_total is already *2 (bf16). A,C here are element*...
        # l2_total includes (rows*K + weight_reread_elems + rows*N-or-2N)*2. Extract weight reread:
        # compulsory_l2 (no reread) = (A_e + W_e + C_e)*2 where A_e=rows*K, W_e=e_active*wpe_dims, C_e=rows*Nout
        if kl.endswith("_w1"):
            comp_l2 = (rows*K + e_active*(2*N)*K + rows*(2*N))*2
        elif kl.endswith("_w2"):
            comp_l2 = (rows*N + e_active*N*K + rows*K)*2
        else:
            comp_l2 = (rows*K + e_active*K*N + rows*N)*2
        weight_reread = l2_total - comp_l2  # extra weight traffic beyond compulsory (already *2)
        frac = 1.0 - l2_cap/wpe if wpe>l2_cap else 0.0
        dram_curr = (W+A+C) + (frac*W if wpe>l2_cap else 0)
        dram_fix = (W+A+C) + (frac*weight_reread if wpe>l2_cap else 0)
        asc = kd.get("Ascend") or {}
        dram_meas = asc.get("dram_read_bytes",0) or 0
        xbar_meas = asc.get("gm_to_l1_bytes",0) or 0
        proj = "w1" if kl.endswith("_w1") else "w2" if kl.endswith("_w2") else "uns"
        print(f"{B:>5} {K:>5} {N:>5} {proj:>4} {W/1e9:>8.2f} {wpe/1e6:>7.1f} {frac:>5.2f} {weight_reread/1e9:>8.2f} {dram_curr/1e9:>9.2f} {dram_fix/1e9:>8.2f} {dram_meas/1e9:>9.2f} {xbar_meas/1e9:>8.2f}")
