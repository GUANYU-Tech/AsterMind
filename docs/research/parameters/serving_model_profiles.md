# Serving model profile constants

OrbitServe freezes the following batch-one analytical constants in
`src/orbitserve/serving/model_profiles.py`. They are modeling inputs, not
measured end-to-end throughput. Rates are produced only after combining them
with the selected single-satellite hardware profile, L2 formation, parallelism,
efficiency factors, and communication model.

| Canonical model | Total parameters | Active parameters/token | Layers | Hidden size | Prefill A (FLOPs/token) | Prefill B (FLOPs/token²) | KV bytes/token | Max context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama3.3-70b` | 70B | 70B | 80 | 8192 | 136,902,082,560 | 2,621,440 | 327,680 | 131,072 |
| `deepseek-v3` | 671B | 37B | 61 | 7168 | 71,182,974,976 | 4,997,120 | 70,272 | 128,000 |
| `kimi-k3` | 2.8T | 104B | 93 | 7168 | 208,000,000,000 | 1,474,560 | 27,648 | 1,048,576 |

The full-prompt function is `F_prefill(P) = A * P + B * P²`. Weight and
activation storage default to BF16 (2 bytes). DeepSeek V3 KV stores the MLA
compressed latent and RoPE key. Kimi K3's quadratic term covers its 24 full MLA
layers; its fixed KDA recurrent state and vision tower are excluded from the
per-token, text-only approximation.

These coefficients must be treated as explicit bounded assumptions in paper
experiments. Hardware efficiency and formation link parameters should be
swept independently rather than modifying the frozen model dictionary.
