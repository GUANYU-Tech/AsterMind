# 访存导出与诊断入口

当前导出读取生产 owner 的同一份 `MemoryAccessProgram`，不复制 predictor 公式，不从性能真值生成访问，不把逻辑字节数称为 DRAM 流量。声明目录、一次成功导出、完整 source/provider 验收是三个不同结论。旧分支的阶段证明不自动成为当前版本的验收结果。

## 可运行入口

```bash
python scripts/export_memory_access.py --m 64 --n 128 --k 64 --dtype bf16 \
  --hardware-id nvidia_a800 --granularity tile --tile 32 64 32 \
  --program-id gemm0 --call-id call0 --binding-id binding0 \
  --output experiment_logs/evaluation/memory_interface/example.jsonl --report
python scripts/export_memory_access.py inventory \
  --output-dir experiment_logs/evaluation/memory_unification/inventory
python scripts/export_memory_access.py report \
  --input experiment_logs/evaluation/memory_interface/example.jsonl
python scripts/export_memory_access.py expand-tiles \
  --input experiment_logs/evaluation/memory_interface/example.jsonl \
  --output experiment_logs/evaluation/memory_interface/expanded.jsonl \
  --max-accesses 10000 --max-dependencies 100000
python scripts/export_memory_access.py compare-exports \
  --before experiment_logs/evaluation/memory_interface/example.jsonl \
  --after experiment_logs/evaluation/memory_interface/expanded.jsonl \
  --output experiment_logs/evaluation/memory_interface/comparison.json
```

GEMM 旧参数形式继续可用，也可显式写 `gemm` 子命令。它调用 source owner 的逻辑 GEMM 构造器，独立对象绑定不代表 Serving 地址；用户指定 tile 不冒充实测 launch。`transfer` 保留原来的 `--kind MEMCPY|KV_OFFLOAD --hardware-id --bytes --direction --algorithm --link [--tier] --program-id --output [--report]` 参数，直接调用当前 `KernelSpec`、`PhysicalTransferConfig` 与公共 Kernel 访问接口，非法物理配置保持拒绝。

`inventory` 生成当前已注册 KernelKind 和 source 构造/准入函数的路径、行号、条件；它不把所有 shape、硬件、variant 的笛卡尔积标成支持。实际 source 适用域由生产 owner 对具体 KernelSpec 判断，目录中的函数数不是通过验收的算子数。

`report` 输出 `coverage.json` 和 `report.md`；codec round-trip 必须相等。requested bytes、候选访问字节与 unique footprint 分开记录，条件未决时不填写精确请求字节，地址未决时不填写精确 unique。独立逐字节 reference 只覆盖 Span/Strided 的小案例；超过界限或条件未决写 `unsupported`，不影响生产契约本身的适用范围。物理流量、时间和全局验收不能由导出成功推出。不同案例应使用独立目录，避免同目录 `coverage.json`/`report.md` 被下一次报告替换。

`expand-tiles` 只做有上限的解释展开。动态 cardinality、未绑定 partition、依赖设备索引值的程序继续由生产展开器拒绝，不能用候选区域冒充已执行访问。`compare-exports` 要求相同 program、binding 和 hardware identity，分别报告完整程序是否相等及逻辑字节变化；它不证明两次执行的数值或性能相同。

## 程序化 Kernel 与 Serving 接口

```python
from evaluation.reporting.memory_tiles import export_kernel_memory
from evaluation.serving.memory_unification import write_memory_plan_report

# kernel/hardware/bindings 来自当前生产接口，不从历史结果合成。
program = export_kernel_memory(kernel, hardware, output_path,
                               bindings=bindings, program_id="call0:memory")
# plan 是实际执行、完成 bind_trace_memory 后的 KernelPlan。
result = write_memory_plan_report(plan, hardware_by_id, output_directory)
```

`export_kernel_memory` 支持当前生产 builder 已准入的 family；无 bindings 时使用显式 standalone 绑定，Serving 必须提供真实 call binding。未知族拒绝，不回退为伪造字节数。`summarize_memory_plan`/`write_memory_plan_report` 遍历 `KernelPlan.logical_calls()`，保留每个真实调用的 rank、pool、source ports、producer、allocation identity 及缺口原因。未绑定为 `partial`，已有绑定但访问生成未支持为 `unsupported`；错误形状、错误身份和缺硬件配置继续抛错。共享 allocation 不逐 call 累加成显存峰值，也不新建 storage owner。报告不会重建模型或执行预测公式。

`evaluation.kernel.memory_frozen_evaluation.read_snapshot_json` 保留普通 JSON/gzip/zstd 的互斥读取，冲突表示直接拒绝。旧 campaign 的冷归档打包、源码 checkout/revision 执行和固定模型批量诊断在原源码中复现；当前轻量诊断不隐式执行历史树。

## 旧命令迁移范围

旧完整源码对应提交 `564bdba1a5014d7b58a8d6a135f2a1a47a3ba616`，可用 `git show 564bdba1a5014d7b58a8d6a135f2a1a47a3ba616:scripts/export_memory_access.py` 查看原 CLI。临时 backup tag 已退休；本地历史归档为 `.git/dev-integration-20261026/history_refinement/retired-local-history.bundle`，若旧提交对象不在当前仓库，可用 `git bundle unbundle <归档路径>` 恢复对象。其他 checkout 需要另行取得该历史归档，不能假设 dev 自带旧分支全部提交。下面列出原命令的处置；“历史复现”表示该精确 campaign 命令没有移植成当前版本的等价功能，不能将旧证书当作当前通过。若需要其旧行为，应在独立历史 checkout 使用原 owner、输入及证据。代码备份不保证被清理的实验产物仍存在。

通用能力迁移如下：原无子命令 GEMM 和 `transfer` 保留；当前 source export、bounded tile diagnostics、对同一 program 的比较由上述通用接口覆盖；真实 Serving storage 诊断改为 traced `KernelPlan` API。固定 22-function smoke、各族 function-acceptance、源修复前后逐价比较、历史 U/T gate 汇总、固定 QPS16 campaign 不在新版本静默执行。`--validation-dir/--reference-review` 的旧 gate 自动汇总也只在历史源码保留；当前报告不根据一组旧日志声明整个 Goal 完成。

| 原 CLI 命令 | 当前处置 |
|---|---|
| `archive-completed-snapshot` | 历史复现；精确 campaign/验收语义未移植 |
| `capture-graph-gemm` | 历史复现；精确 campaign/验收语义未移植 |
| `capture-observed-router-callers` | 历史复现；精确 campaign/验收语义未移植 |
| `capture-sglang-topk1-vocab` | 历史复现；精确 campaign/验收语义未移植 |
| `combine-function-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `compare-grouped-copy` | 历史复现；精确 campaign/验收语义未移植 |
| `compare-mixed-gemm` | 历史复现；精确 campaign/验收语义未移植 |
| `compare-observed-router-callers` | 历史复现；精确 campaign/验收语义未移植 |
| `compare-tile-runtime-dictionaries` | 历史复现；精确 campaign/验收语义未移植 |
| `compare-transfers` | 历史复现；精确 campaign/验收语义未移植 |
| `compare-unknown-activation-packing` | 历史复现；精确 campaign/验收语义未移植 |
| `cuda-extent-partition-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `cuda-extent-state-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `fa3-combine-retained-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `fa3-prefill-function-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `fa3-prefill-prepare-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `fa3-scheduler-layout-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-cann-cast-runtime-coverage` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-cann-gemm-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-cann-split-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-frozen-stage` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-gemv-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-generated-fill-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-generated-matrix-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-generated-offload-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-generated-paged-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-generated-quantizer-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-generated-reduction-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-generated-rotary-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-generated-scalar-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-gpu-gemm-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-metadata-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-moe-fill-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-moe-tensor-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-native-routing-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-normalization-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-rotary-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-scalar-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `finalize-simple-tile-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `freeze-baseline` | 历史复现；精确 campaign/验收语义未移植 |
| `freeze-graph-gemm-revision` | 历史复现；精确 campaign/验收语义未移植 |
| `freeze-source-revision` | 历史复现；精确 campaign/验收语义未移植 |
| `frozen-serving-qps16` | 历史复现；精确 campaign/验收语义未移植 |
| `gpu-extent-contract-boundary` | 历史复现；精确 campaign/验收语义未移植 |
| `gpu-extent-function-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `gpu-extent-recursive-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `interface-smoke` | 历史复现；精确 campaign/验收语义未移植 |
| `layernorm-contiguous-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `layernorm-contract-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `layernorm-function-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `marlin-binding-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `marlin-compiler-binding` | 历史复现；精确 campaign/验收语义未移植 |
| `metax-extent-index-limit-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `metax-extent-partition-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `metax-tiny-extent-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `mhc-abstract-source-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `mhc-post-prenorm-function-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `observed-router-main-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `pair-cann-bsh-native-calls` | 历史复现；精确 campaign/验收语义未移植 |
| `pair-frozen-snapshots` | 历史复现；精确 campaign/验收语义未移植 |
| `pair-graph-gemm` | 历史复现；精确 campaign/验收语义未移植 |
| `progress-report` | 历史复现；精确 campaign/验收语义未移植 |
| `public-router-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `reduction-contract-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `reduction-function-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `retry-tile-runtime-probe` | 历史复现；精确 campaign/验收语义未移植 |
| `reuse-static-tile-runtime` | 历史复现；精确 campaign/验收语义未移植 |
| `revalidate-moe-reduction-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `revalidate-native-routing-scalar-dependency` | 历史复现；精确 campaign/验收语义未移植 |
| `router-factory-boundary` | 历史复现；精确 campaign/验收语义未移植 |
| `router-function-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `summarize-paired` | 历史复现；精确 campaign/验收语义未移植 |
| `summarize-serving-qps16` | 历史复现；精确 campaign/验收语义未移植 |
| `summarize-tile-runtime-probe` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-activation-caller-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-alignment-caller-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-attention-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-attention-retained-selection` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-attention-source-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-bsh-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-bsh-before` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-bsh-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-cast-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-cast-retained-inputs` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-cast-runtime-coverage` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-gemm-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-metadata-followup` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-mla-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-mla-split-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-paged-gqa-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-rotary-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-rotary-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-rotary-retained-inputs` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-split-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-split-descriptor-selection` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-split-graph-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-split-graph-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-split-graph-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-tnd-aggregate-inputs` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-tnd-combine-price-projections` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-tnd-general-evidence` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-tnd-price-projections` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-tnd-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-tnd-stack-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cann-transpose-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-case-status` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-communication-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-alignment-caller-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-alignment-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-alignment-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-alignment-retained-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-auxiliary-alignment-compose` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-auxiliary-callers` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-matrix-alignment-compose` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-regular-alignment-retained-inputs` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-small-alignment-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-small-alignment-retained-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-cuda-small-alignment-retained-inputs` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-declared-variant-gaps` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-deterministic-runtime-coverage` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-dsa-attention-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-dsa-group-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-dsa-group-examples` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-eager-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-epilogue-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-examples` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-expert-map-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fa2-domain-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fa2-followup` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fa2-retained-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fa2-retained-selection` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-flat-softmax-before` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fp8-aligned-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fp8-aligned-callers` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fp8-counter-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fp8-counter-examples` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fp8-native-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fp8-native-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-fp8-native-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-functional-checkpoint` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-functional-inventory` | 当前声明目录改用 `inventory`；旧variant/domain与历史证书联表仅历史复现，未宣称完全等价 |
| `tile-gemv-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-fill-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-grouped-moe-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-matrix-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-matrix-class-replay` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-offload-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-paged-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-quantizer-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-reduction-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-rotary-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-generated-scalar-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-gpu-gemm-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-grouped-mla-geometry-proof` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-grouped-mla-mixed-proof` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-grouped-moe-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-grouped-moe-generated-selection` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-grouped-moe-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-grouped-moe-retained-selection` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-grouped-moe-source-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-h200-alignment-binding-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-h200-alignment-revalidate` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-h200-cuda-alignment-compose` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-h200-fill-completion` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-index-cache-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-index-cache-binding-evidence` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-indexer-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-marlin-examples` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-marlin-row-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-matrix-retained-snapshot-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-memory-binding-checkpoint` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metadata-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metadata-source-check` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metax-metadata-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metax-metadata-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metax-metadata-retained-inputs` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metax-mla-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metax-mla-metadata-control-reference` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metax-mla-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metax-mla-retained-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-metax-varlen-followup` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-mhc-function-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-mla-followup` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-moe-alignment-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-moe-count-sort-after` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-moe-count-sort-before` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-moe-count-sort-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-moe-fill-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-moe-metadata-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-moe-tensor-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-moe-vector-retained-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-moe-vector-retained-inputs` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-native-cache-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-native-cache-binding-evidence` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-native-fp8-caller-revalidation` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-native-routing-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-normalization-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-observed-matrix-retained-selection` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-paged-attention-binding-evidence` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-paged-cache-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-pairnorm-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-pim-normalization-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-pim-normalization-before` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-predicate-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-progress-report` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-range-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-reduction-binding-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-reduction-parent-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-retained-paged-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-rotary-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-rotary-prediction-supplement` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-route-weight-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-route-weight-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-runtime-probe` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-scalar-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-short-context-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-short-context-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-simple-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-slot-predicate-examples` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-softmax-native-caller-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-softmax-partition-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-sparse-composed-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-split-class-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-split-class-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-split-class-runtime` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-c500-topk-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-cann-gemm-replay-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-gemm-elementwise-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-gemm-elementwise-replay-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-gpu-norm-replay-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-layernorm-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-mhc-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-mhc-replay-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-moe-metadata-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-moe-metadata-replay-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-pairnorm-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-reduction-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-reduction-replay-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-router-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-router-graph-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-row-norm-piecewise-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-row-norm-replay-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-simt-router-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-simt-router-graph-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-v32-norm-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-v32-q-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-static-v4-inverse-capture` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-target-cache-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-tensor-kv-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-tensor-kv-before` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-tensor-kv-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-tensor-rotary-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-tensor-rotary-before` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-tensor-rotary-controls` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-transfer-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-transfer-precision-fields` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-unbound-encoding-dependency-checks` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-unfused-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-v32-norm-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-v32-q-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-v32-query-checkpoint` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-v4-decode-examples` | 历史复现；精确 campaign/验收语义未移植 |
| `tile-varlen-runtime-coverage` | 历史复现；精确 campaign/验收语义未移植 |
| `transfer` | 保留原参数，使用当前公共 Kernel 访问接口 |
| `v4-prefill-aux-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `v4-prefill-function-acceptance` | 历史复现；精确 campaign/验收语义未移植 |
| `v4-prefill-model-comparison` | 历史复现；精确 campaign/验收语义未移植 |
| `v4-prefill-public-comparison` | 历史复现；精确 campaign/验收语义未移植 |
