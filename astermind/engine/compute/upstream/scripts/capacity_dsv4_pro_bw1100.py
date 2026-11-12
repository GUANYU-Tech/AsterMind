#!/usr/bin/env python3
"""Compute the 36 additional declared V4 decode capacity points on CPU."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.serving.metrics.flash_decode_steady import scenario_hardware
from src.serving.workload.model_config import load_model_config
from src.serving.kv_cache.model_residency import flash_fp4_decode_capacity

rows = []
for name, hw, bw in (("pro", "hygon_bw1100", False), ("pro", "nvidia_rtx_pro_5500", False),
                     ("pro", "hygon_bw1100", True), ("flash", "hygon_bw1100", True)):
    model = load_model_config(f"configs/model/deepseek-v4-{name}.json")
    hardware = scenario_hardware(bw)
    for scheme in ("tp8_ep8", "dp8_ep8", "dp8_fpga"):
        for mb in (1, 2, 3):
            capacity = flash_fp4_decode_capacity(model, deployment=scheme, concurrency=128 * mb,
                hbm_capacity_bytes=hardware[hw].resource_by_role("dram").size_bytes,
                initial_context_tokens=100000, decode_tokens=32)
            rows.append({"model": name, "hardware_id": hw, "bw216": bw, "scheme": scheme,
                "microbatches": mb, "capacity": capacity})
output = Path("experiment_logs/bw1100_pro5500_fpga/pro/new_capacity_table.json")
output.write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} cases to {output}")
