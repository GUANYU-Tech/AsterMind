# Serving lowering 缺的 helper-launch contract：形状、判据与为什么不能参数化

本文只描述**契约的形状**，不含实现，也不改动任何代码。写作动机是两条独立路径指向同一处缺口：910B step decomposition 工具把自身标记为 `blocked_pending_source_authored_helper_contract`（`scripts/evaluation/serving/diagnostics/report_ascend_910b_sharegpt_step_decomposition.py:170`），其 `required_scope` 列出三类 helper；C500 batch-scaling 一侧得出 mixed 步的缺口不在任何已建模 kernel 类的定价上，而在"步里有时间没被任何 kernel 计价"。两者要的是同一件东西：**lowering 在某些调用点必须声明它发射 0 个、1 个还是 N 个 kernel**，而不是给残差配一个参数。

## 1. 缺口的准确位置：不是定价层，是 emit 层

区分两个不同的缺失，避免用错层的修复：

| 缺失 | 已有归属 | 症状 |
|---|---|---|
| launch 已进 `KernelPlan`，但 host 侧准备时间未计 | `src/serving/runtime_prepare.py`（source-static occurrence rule，自身不带 duration） | host 侧欠计 |
| launch **从未进入** `KernelPlan` | **无归属** | 既不被 device 定价，也不被 host 准备；步里出现无人认领的时间 |

本文只针对第二种。这类发射在模型里**连类别都没有**，因此任何对既有 kernel 类定价参数的调整都不可能触及它——这正是 C500 那侧"不在已建模 kernel 类的定价上"的结论的结构含义。

## 2. 三类调用点及其 0/1/N 判据

判据全部读自 vLLM / vllm_ascend 源码的分支谓词，不含任何按卡开关，也不含任何由误差反推的量。行号取自 `/share_data/yangxinhao/chip_modeling/vllm_src_3ver`。

### 2.1 attention 输出写回（TensorMove）

**计数（2026-09-23 订正）：切片赋值分支在 ATen 语义层 1 次；paged `out=` 分支 0 次；发射到 trace 的 kernel 计数由 torch_npu/aclnn 派发层决定，不等于 ATen 层计数，见下方实测相位律。**

> **原断言「每次 attention layer 调用恰好 1 次，无例外」已被 canonical 轨迹实测证伪并作废，不得继续引用。** 证伪与逐 case/phase 计数见 `docs/serving_accuracy_diagnosis.md` §14（`30b39fce7`，wt-c500-batch-scaling Goal 3 交付），证据在 `experiment_logs/serving/helper_launch_20260923/`。实测发射律（910B canonical，每 rank 每 phase 代表步）：decode 全 case 0（paged `out=` 与 fused 消去两个候选分支下计数均为 0，机制不可判）；qwen3-30b prefill 0；qwen3-4b/8b prefill/mixed 2L−1（其一在写回位置、另一 unattributed）。偏差机制在 torch_npu/aclnn 派发层，python 源码不可判，按 G5 留界；实现侧消费这层计数只能以版本化离散发射 observation 为源，不得按 ATen 层计数或常数补齐。

`vllm_ascend/attention/attention_v1.py` 的切片赋值分支以一次写入调用方持有的 `output` 缓冲的赋值收尾：`output[:num_tokens] = attn_output[:num_tokens]`（`:774`、`:839`、`:974`、`:980`、`:1023`、`:1341`）、`output[:batch_size] = attn_output[:batch_size]`（`:758`）、`output[:batch_size] = attn_output`（`:1176`）、`output[:num_decode_tokens] = attn_out.squeeze(2)`（`:1223`）、`output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]`（`:1275`）。**例外分支为原表漏列**：`forward_paged_attention`（`:842`）不以切片赋值收尾，而是 `torch_npu._npu_paged_attention(..., out=output)`（`:851-859`）原位写出，ATen 层计数 0；capture 下的 `full_graph_pa`（`:627`）同样经 `out=` 收尾（`:674`、`:683`）。

判据是 ATen 语义而非形状：右侧 `attn_output` 始终是 `torch_npu.npu_fused_infer_attention_score(...)` / `..._v2(...)` 新返回的张量，从不是 `output` 自身，因此 `output[:n] = ...` 在 ATen 语义上是一次真实的 device 拷贝，不可能被 aliasing 消去。**但 ATen 语义有一次拷贝不蕴含 trace 有一次发射**：派发层可以消去（decode、30b prefill 实测 0）也可以增殖（4b/8b prefill/mixed 实测 2），上方相位律即其实测结果。

**分支谓词大部分是 source-static 的、且 Serving 已经持有**：`_EXTRA_CTX.capturing`、`attn_metadata.attn_state == AscendAttentionState.DecodeOnly / PrefillNoCache`、`self.sliding_window is not None`、`self.sinks is None`、`attn_metadata.seq_lens.shape[0] == query.size(0)`。例外是 paged 分支：`using_paged_attention`（`attention/utils.py`）额外要求 `cudagraph_mode == FULL_DECODE_ONLY` 且 `runtime_shape in get_ascend_config().pa_shape_list`，均为运行时配置而非源码静态事实；且 910B canonical trace 中没有任何名字含 Paged 的 kernel、decode 步跑的是 `FusedInferAttentionScore`，故 decode 实际走哪条 python 分支不可由 trace 判定——两候选分支的写回计数均为 0，计数可判、机制不可判（G5）。

同一文件里的**反例**决定了这条规则不能由形状推出：`query = query[:num_tokens]`（`:785`）与 PrefillNoCache 路径上的 `key = key[:num_tokens]` / `value = value[:num_tokens]` 与上面在文本上同形，但它们是 dim-0 切片**视图**，发射 0 个 kernel。形状、batch、卡都相同，计数却是 0 与 1 之差。

附带一处条件物化：`_forward_fia_slidingwindow` 中 `key = self.key_cache.flatten(2, 3).contiguous()`（`:739-740`），仅在该分支出现，需单独声明。

lowering 现状：`build_workload.py` 在 `FLASH_ATTN` 之后直接 append `{prefix}:o-proj`（`:2573`、`:4753`），中间没有任何节点。

### 2.2 MoE routing mask 三元组

**计数：EP 关闭时 0；EP 开启时每次 MoE layer 调用恰好 3 次。**

> 2026-09-23 订正：3 指源码层算子数（advanced indexing / `!= -1` / `* mask`），成立；发射到 trace 是**连续 4 kernel 串** Index→NotEqual→Cast→Mul 各 1/层，Cast 来自 bool mask 与 topk_weights 的类型提升，逐 phase 实测 48/串/层精确，见 `docs/serving_accuracy_diagnosis.md` §14。


`vllm_ascend/ops/fused_moe/token_dispatcher.py:344-348`：

```python
if expert_map is not None:
    global_num_experts = len(expert_map) + global_redundant_expert_num
    mask = expert_map[topk_ids] != -1
    topk_weights = topk_weights * mask
```

`expert_map[topk_ids]` 是 advanced indexing → Index；`!= -1` → NotEqual；`* mask` → Mul。谓词 `expert_map is not None` 等价于 EP 生效，是一个二值的 source 事实，不是连续量。

同文件 `:339-343`，`apply_router_weight_on_input` 为真时额外 1 次 Mul（`hidden_states = hidden_states * topk_weights.to(...)`，其断言要求 topk == 1）。

`vllm_ascend/ops/fused_moe/experts_selector.py` 上由 selector 变体决定的条件发射：grouped-topk 路径的 `topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)`（`:155`，其上还有 `scatter_`、`expand`、`reshape`）；zero-expert 路径的 `torch.where`（`:322`、`:330`、`:331`）；`_renormalize_topk_weights` 的除法。

这些谓词在 lowering 里**已经被解析并挂在节点上**：`build_workload.py:1669-1697` 的 ROUTER 节点属性已携带 `topk_method`、`score_function`、`normalize_topk_probability`、`expert_group_count`、`topk_group_count`。也就是说本类所需的输入一个都不缺，缺的只有 occurrence 声明本身。

### 2.3 model 输入/输出边界

**计数：每个产生采样的 step 恰好 1 次 gather。**

`sample_hidden_states = hidden_states[logits_indices]` 是 advanced indexing，必定物化，不是视图。该调用点在三卡的 vLLM 树里逐一存在：`a800/vllm/v1/worker/gpu_model_runner.py:2330,2348`、`c500/vllm/v1/worker/gpu_model_runner.py:2101`、`910b/vllm/v1/worker/gpu_model_runner.py:3822,3828`、`c500_0.24/vllm/vllm/v1/worker/gpu_model_runner.py:4366,4372`，以及 `vllm_ascend/worker/model_runner_v1.py:1398,1405,1409`。**这是 hardware-independent lowering 属性的直接证据**：谓词与计数在 vLLM 层共享，只有最终落地的算子名按卡不同。

同一边界上的条件发射：`hidden_states[:-pad_size, :]`（`model_runner_v1.py:1762`，`_all_gather_hidden_states` 内，谓词 `pad_size > 0`）；输入准备侧的 `torch.index_select`（`:665`、`:676`）与 `.copy_`（`:712`、`:746`、`:753`）；`draft_token_ids = self.input_ids.gpu[logits_indices]`（`:944`）。反例同样必须声明为 0：`hidden_states[:num_scheduled_tokens]`（`:1048`）是视图发射 0，而 `hidden_states[token_indices]`（`:1083`）是 gather 发射 1。

lowering 现状：`build_workload.py` 在 `final-norm` 之后直接按 `target_logit_rows` append `lm-head`（`:2812-2839`），**产生这些行的 gather 没有被发射**。而同一函数的 MTP 分支已经把结构完全相同的一对发射了出来：`GATHER` / `variant="embedding"` / `gather_factor=1` / `index_bytes=8` / `index_count` / `scattered=True` / `source_symbol`（`:2396-2414`），以及紧随其后的 copy 形 `ELEMENTWISE` / `variant="masked-fill-zero"` / `operation="copy"` / `reads=1` / `writes=1`（`:2924-2934`）。主路径缺的正是它自己在 MTP 分支上已有的东西。

## 3. 契约的形状

对每个调用点需要声明的字段（与 `runtime_prepare.py` 的 `RuntimePrepareRule` 同构：只声明**发生**，不携带 duration）：

- `source_symbol`：调用点的源码符号，与 `build_workload.py` 现有约定一致（`"vllm.FusedMoE.forward"`、`"vllm.GlmMoeDsaForCausalLM.compute_logits"` 等）。
- `predicate`：决定计数的 source 谓词，逐条可回指到上节的文件与行。
- `count`：`0`、`1` 或一个由谓词求值得到的整数；**`0` 必须被显式授权**，不能是"未声明"的默认值。
- `kind` / `variant` / 形状来源：复用已有词汇（`GATHER` + `gather_factor`/`index_bytes`/`index_count`/`scattered`，`ELEMENTWISE` + `operation`/`reads`/`writes`），形状由已在 lowering 中的 token 数、hidden、topk 等直接算出。
- 未覆盖分支：**fail-closed**，抛错而非记 0。

最后一条是本契约最关键的结构要求。这里 `0` 是合法取值（视图不发射），因此"静默记 0"与"该分支没人写过"在数据上不可区分；一旦允许默认 0，契约就退化成它本要替代的那种沉默。这与仓库对 `.get(...)` / 缺 key 补默认值的禁令是同一条规则。

验收也由此确定，且与 duration 无关：声明出的 per-step 算子计数必须与 trace 中对应算子的计数**逐步精确相等**。这是一个可证伪的整数等式，而不是一个误差指标。

## 4. 为什么参数化不能替代

**（1）计数不是任何可观测量的函数。** `output[:n] = attn_output[:n]` 发射 1，`query = query[:n]` 发射 0，二者的形状、batch、模型、卡完全相同。不存在以 (shape, batch, model, hardware) 为自变量的函数能把这两者分开——区别只在它是哪一个 ATen 调用。参数按可观测量索引，而这个量不在可观测量的张成里。

**（2）需要表达的是分支，不是标量。** `expert_map is not None` 使计数在 TP-only 下为 0、在 EP 下为 `3 × n_moe_layers`。用一个 per-step 常数覆盖，等于为每种并行配置各存一个数，也就是工具自己禁止的 case/model 常数（`forbidden_shortcut`：*Do not fit the phase closure gap or add a case/model constant*）。

**（3）拟合它对它要关闭的残差不可识别。** 用于拟合的量与被关闭的量是同一个 closure gap：任何常数按构造都能关闭它，在任何卡上都"成立"，因而零证伪力。契约给出的是独立于 duration 的整数计数等式，可以被 trace 单独否证。

**（4）事实在 vLLM 层跨卡共享，参数会把一个事实存成三份。** §2.3 的调用点在 a800 / c500 / c500_0.24 / 910b 四棵树上逐一存在；按卡各存一个常数会把同一个 lowering 事实复制三遍，并在 vLLM 版本变动时静默失配。谓词随版本一起移动，常数不会。

**（5）typed 路径已经存在，常数会绕过它。** §2.3 所需的节点形状在 MTP 分支上已被完整发射。一个每步常数会丢掉这些发射本来就能算出的形状依赖（token 数、hidden、index 宽度、scattered 与否），用一个不随负载变化的数替换一个随负载变化的量——在 batch scaling 这类恰恰考察负载依赖的场景里，这是把待测的结构直接抹平。
