#!/usr/bin/env python3
"Closed-loop colocated vs PD DeepSeek-V3 1K/128 sweeps with MTP off/on.\n\nEvery group runs under the InferenceX-style zero-think-time closed-loop\nclient contract: C clients, each with at most one outstanding request, the\nsuccessor injected at the client's own completion instant\n(--closed-loop-client-count C).  This mirrors the GLM-5.2/Kimi-K2.6\nInferenceX comparisons so the curves are directly comparable.\n\nDeepSeek-V3 is the official FP8 release (block-wise 128x128 e4m3 weights,\ndynamic activations).  The simulator's FP8 weight contract models one\nstorage byte per element and has no scale-granularity dimension, which the\nconfig loader states explicitly; no per-tensor requantization variant is\ninvented.\n\nMTP on means the official single NextN (deepseek_mtp) draft layer\nproposing 1 token per step, top-1, with an explicit declared acceptance\nassumption (dsv3-top1-depth1-mean0.87-v1): the DeepSeek-V3 technical\nreport (arXiv:2412.19437) reports 85-90% MTP acceptance across generation\ntopics; 0.87 is the approximate midpoint and is an ASSUMPTION, not a\nmeasured trace on this workload.\n\nPD KV handoff uses the registered H200 scale-out ProtoBridge send_recv\nprofile (400GbE RoCEv2 theory preset, development-only).  Client-visible\nTTFT/TPOT are attributed to the terminal pool: the decode slot wait after\na prefill handoff is TTFT, not an inter-token gap."
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.inferencex_deepseek_v3_1k128_closedloop import main

if __name__ == "__main__":
    main()
