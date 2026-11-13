#!/usr/bin/env python3
"Closed-loop colocated vs PD Kimi-K2.6 1K/128 sweeps, MTP off only.\n\nEvery group runs under the InferenceX-style zero-think-time closed-loop\nclient contract: C clients, each with at most one outstanding request, the\nsuccessor injected at the client's own completion instant\n(--closed-loop-client-count C).  Both deployments use the H200 Kimi\nTP8/PP1/EP8 scope: one colocated pool, or one prefill plus one decode\npool with the same per-role topology (the standard vLLM\nDeepSeek-V3-family single-node layout).  Kimi-K2.6 declares zero MTP\nlayers, so only the MTP-off configuration exists.\n\nPD KV handoff uses the registered H200 scale-out ProtoBridge send_recv\nprofile (400GbE RoCEv2 theory preset, development-only).  Client-visible\nTTFT/TPOT are attributed to the terminal pool: the decode slot wait after\na prefill handoff is TTFT, not an inter-token gap."
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.inferencex_kimi_1k128_closedloop import main

if __name__ == "__main__":
    main()
