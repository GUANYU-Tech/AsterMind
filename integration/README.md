# Mid-Layer 外部适配

本目录保存统一仿真器当前使用的 Cell、Scheduler、session/KV 与通信任务适配。计算源码通过 Git 子模块引用 `Popuffu/midlayer_hw_model` 的 `6fe573479c37a931744a6442e7a4b3503eec02e6`，不修改其源码或复制一套 kernel 模型。

```bash
git submodule update --init --recursive
```

运行入口和当前范围见 [simulator/README.md](../simulator/README.md)。本目录只随此次接入提供运行所需适配，不包含其他开发线的 K3/V4、DP/DCP 或 PD/AF 实验模块。部分适配文件保留了这些可选开发入口；它们不属于本次统一仿真器支持范围，不能单独启用缺省关闭的扩展配置。

`requirements-scheduler-latest.lock` 记录现有 Python 3.12 CPU 环境。`runtime/vllm_0_31_identity.json` 固定 vLLM 0.31.0 官方 wheel 的 URL、SHA256 和 Python 源码清单哈希，运行时仍严格校验，不因发布跳过校验。

已有兼容隔离环境可直接使用。新环境按锁文件配置，CPU PyTorch 使用官方 CPU wheel 索引，例如：

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r integration/requirements-scheduler-latest.lock
python -m pip install -r simulator/requirements.txt
python -m simulator.build_midlayer
python -m network_sim.build_native --upstream-tests
```

该入口运行真实 CPU Scheduler 状态机和解析计算模型，不加载权重或运行 GPU executor。固定 M-L 内部的 Git-LFS 数据若按指针检出，只有实际用到的校准数据完整时才能运行对应场景；需要时在子模块中执行 `git lfs pull`。原生消费者的完整性校验保持不变。
