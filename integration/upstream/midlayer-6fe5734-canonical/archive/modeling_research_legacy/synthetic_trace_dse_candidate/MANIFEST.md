# 可吸收代码清单 — from archive/system-e2e-sim-hw-dse

整理日期: 2026-06-24
来源分支: `archive/system-e2e-sim-hw-dse` (commit 789e162, refactor 前快照)
对比基准: `dev_0624`

## 结论摘要

archive 分支共改动 8 个文件。逐一核对后：**6 个已被 dev 吸收**（无需动作），
**只有 `ModelSimulator` + `hw_dse.py` 这条"合成-trace DSE"路径未吸收**，且有保留价值。
本目录已把这 2 个文件（+ 其测试）的 import 路径重写为 dev 当前布局，并验证可运行。

## 一、已被 dev 吸收（无需归档，仅记录）

| archive 文件/改动 | dev 现状 | 对应 |
|---|---|---|
| `src/modeling/diff_formula.py` 的 `eff_cap_per_resource` / `grad_capacity_scaled` 链式法则 | 已逐行存在于 `src/dse/diff_formula.py` | P61 |
| `src/modeling/ops/norm.py` 的 ReducePredictor / GatherPredictor | 已存在于 `src/kernel/ops/norm.py` | P62 |
| `src/emulation/shape_inference.py` / `infer_gemm_dimensions` | 已存在于 `src/serve_trace/shape_inference.py` | (refactor) |
| `src/modeling/kernel.py` predictor 注册 | 已存在于 `src/kernel/kernel_base.py::create_predictor` | (refactor) |
| `src/emulation/trace_predictor.py` 改动 | 已并入 `src/serve_predict/predict_kernel_time_from_trace.py` | P35 等 |

archive 的 commit message 自己也说这些 "later re-derived on dev"。

## 二、未吸收 · 有价值（本目录的 3 个文件）

### 是什么
`ModelSimulator`：从 model config **解析式生成**一条合成 transformer kernel trace
（逐层 norm→qkv→rotary→kv_cache→FA→o_proj→norm→gate_up→SiLU→down + lm_head + MoE FFN），
不需要跑 vLLM emulation 抓真实 trace，直接出 e2e GPU 时间 + 可微梯度，喂给硬件 DSE。

### 和 dev 现有路径的区别
- dev `scripts/dse/run_dse.py`：DSE 基于**真实 vLLM emulation trace**（`annotate_trace_diff`）。
- 本路径 `hw_dse.py` + `ModelSimulator`：**免 emulation 的解析路径**，快速 what-if / DSE。

### 文件
| 本目录文件 | 建议落位 | 行数 | 说明 |
|---|---|---|---|
| `model_sim.py`    | `src/dse/model_sim.py`          | 613 | ModelSimulator 主体 |
| `hw_dse.py`       | `scripts/dse/hw_dse.py`         | 320 | 合成-trace 硬件 DSE 驱动 |
| `test_model_sim.py` | `tests/dse/test_model_sim.py` | 308 | 测试 |

### 已做的适配（相对 archive 原文件）
- import 路径全部重写到 dev 布局：
  - `src.modeling.hw_spec` → `src.kernel.hw_spec`
  - `src.modeling.diff_formula` → `src.dse.diff_formula`
  - `src.modeling.kernel` → `src.kernel.kernel_base`
  - `src.emulation.trace_predictor`(ModelConfig/load_model_config) → `src.serve_predict.model_config`
  - `src.emulation.trace_predictor`(TracePredictor) → `src.serve_predict.predict_kernel_time_from_trace`
  - `src.emulation.shape_inference` → `src.serve_trace.shape_inference`
  - `hw_configs/` → `configs/hw/`
- 全部 dev API 签名已核对一致（`predictor.forward` / `formula.predict_kernel` /
  `build_synthetic_ncu` / `aggregate_gradients` / `infer_gemm_dimensions`），仅路径变化，无逻辑改写。

### 验证（临时 stage 到 dev 跑过，已清理）
- `model_sim.py` import OK；功能 OK：Qwen3-0.6B-like dense 模型 predict() = 2.78ms，
  per-category 分解正常；predict_diff() 返回完整 agg_grad。
- `hw_dse.py --help` CLI 正常。
- `test_model_sim.py`：**26 passed**。

## 三、归档前需注意（决定是否真的并入 dev）

1. **零拟合/准确性顾虑**：`ModelSimulator` 生成的是"假设的" kernel 序列，**不反映框架真实调度**——
   这与 dev serve_trace 管线的设计初衷（design.md 贡献 2："trace 反映框架实际行为而非人工假设"）相悖。
   合成 trace 的 e2e 精度**从未对齐过真实 trace**（archive 是 exploration，未验证）。
   → 若要真正并入 dev，建议先验证：同一 (model, batch, seq) 下合成 trace 的 e2e GPU 时间
   vs 真实 emulation trace 的误差 < X%；达标才并，否则只作"快速粗估"工具并显著标注。

2. **测试冗余**：`test_model_sim.py` 里的 `TestReduceGatherPredictors`（2 个）与 dev 已有的
   ReduceGather 测试重叠，可删；其余 `TestPredictDiff` / `TestGradientSignConsistency` /
   trace 生成测试是 ModelSimulator 独有，保留。

3. 这是参考快照，**未 push、未并 main/dev**。是否并入由你决定。
