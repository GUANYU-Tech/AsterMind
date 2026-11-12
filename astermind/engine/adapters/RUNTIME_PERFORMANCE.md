# Runtime performance changes

本轮保持原生调度、算子依赖、路径持久化、逐跳传播及 FlowSim 完成回调语义。
性能测量使用同一份物理数据、同一 GLM 配置和同一请求流。

## 本地依赖补丁与构建

公司要求第三方目录只保留本地。可审阅的修改保存在
`astermind/engine/adapters/patches/astermind-runtime-performance.patch`；预测缓存的自有源码为
`astermind/engine/prediction_cache.py`。补丁入口会检查上下文并保留冲突的本地修改。

在仓库根目录、已配置的 Linux / WSL Python 环境中执行：

```bash
python astermind/engine/adapters/apply_runtime_optimizations.py --check
python astermind/engine/adapters/apply_runtime_optimizations.py
python astermind/engine/adapters/build_flowsim.py --output-dir local/native/runtime-performance
```

使用新内核时，将构建输出目录放在该进程的 `PYTHONPATH` 首位；新启动的进程应检查
`_flowsim.__file__` 是否指向该目录。构建目录包含源文件指纹、编译参数和二进制指纹。
已启动的进程继续使用其加载的内核。

## 优化边界

- FlowSim：链路容量和忙碌时间在状态变化时惰性结算，报告时补齐；计数仅重置上一轮占用的链路。
  无聚合共享池的独立单跳采用精确闭式速率，其余情况保留通用 max-min 求解器。
- 路由：复用按拓扑修订及类型过滤的有序结构邻接表；重复的完成时刻距离查找合并。
  路径缓存、流量相关时序计算、接触窗口检查、故障重选和排序规则保持原语义。
- 预测：每个原生 ServingRuntime 使用最多 2,048 项 LRU。每次用完整 KernelSpec 和当前
  HWSpec 实例状态生成相等性键，包含嵌套值及映射顺序；只有深度不可变的结果可以共享。
  未知输入或无法编码的输入回退原预测器；异常不缓存。没有缓存执行的开始/完成时刻。
  `ServingRuntime(prediction_cache_entries=0)` 可关闭预测结果缓存。
- Cell 完成记录：按不可变字段直接生成相同的字典，避免对标量和节点 ID 元组递归深拷贝。
- 事件详情构造提取到 `_record_event()`，便于单独计时；记录内容、事件优先级和回调顺序保持原语义。

## 诊断与验证

`FLOWSIM_PROFILE=1` 在新内核中启用细分计时；通过
`EventSimulator.performance_stats()` 读取速率重算、字节推进、完成搜索和完成处理耗时及计数。

`astermind.engine.metrics.performance.WallProfiler` 提供可选的嵌套 wall timer。
子调用耗时从父调用中扣除；指定 `root_label` 后，只记录该根作用域内部的调用。
未包装的代码保留在外围作用域自身耗时中。诊断有额外开销，比较时应说明计时设置。

```bash
python -m pytest tests/unit/runtime/test_flowsim_statistics.py \
  tests/unit/runtime/test_flowsim_transport.py \
  tests/unit/runtime/test_prediction_cache.py \
  tests/unit/runtime/test_performance_accounting.py \
  tests/unit/runtime/test_persistent_routing.py
```

正确性检查包括中断/取消、变带宽、共享池、多跳、闲置链路积分、报告重复读取、
硬件/KV/专家分布变化、不可变结果隔离、异常回退，以及完整仿真的请求/算子/通信流时间戳比较。
同规模实验还核对全部 320 个 rank、算子依赖、流完成数与最终在途流数量。
