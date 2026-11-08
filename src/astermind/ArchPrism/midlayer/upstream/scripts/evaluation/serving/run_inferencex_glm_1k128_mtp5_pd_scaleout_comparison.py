#!/usr/bin/env python3
'Compare colocated and scale-out PD GLM-5.2 1K/128 sweeps with MTP off/on.\n\nPD KV handoff uses the registered H200 scale-out ProtoBridge send_recv\nprofile (400GbE RoCEv2 theory preset, development-only) instead of the\nscale-up NVLink profile.  Colocated groups are reused verbatim from the\nTP8/EP1 sweep because they involve no inter-pool transfer.'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.inferencex_glm_1k128_mtp5_pd_scaleout import main

if __name__ == "__main__":
    main()
