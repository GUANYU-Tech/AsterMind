#!/usr/bin/env python3
"Closed-loop colocated vs PD GLM-5.2 1K/128 sweeps with MTP off/on.\n\nEvery group runs under the InferenceX-style zero-think-time closed-loop\nclient contract: C clients, each with at most one outstanding request, the\nsuccessor injected at the client's own completion instant\n(--closed-loop-client-count C).  This replaces the earlier static-backlog\nsweeps, whose arrival semantics placed all admission waiting in TTFT for\ncolocated but in TPOT for PD and made the two curves incomparable.\n\nPD KV handoff uses the registered H200 scale-out ProtoBridge send_recv\nprofile (400GbE RoCEv2 theory preset, development-only).  Client-visible\nTTFT/TPOT are attributed to the terminal pool: the decode slot wait after\na prefill handoff is TTFT, not an inter-token gap."
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.inferencex_glm_1k128_mtp5_closedloop import main

if __name__ == "__main__":
    main()
