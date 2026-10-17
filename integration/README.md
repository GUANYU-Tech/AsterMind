# Mid-Layer 外部适配

本目录保存统一仿真器当前使用的 Cell、Scheduler、session/KV 与通信任务适配。`upstream/midlayer-6fe5734-canonical/` 直接包含固定提交 `6fe573479c37a931744a6442e7a4b3503eec02e6` 的完整项目文件和已展开的校准数据；不是 Git 子模块，不依赖访问另一个 M-L 仓库。

M-L 的源码、配置与数据保持原样。仅调整快照中的 Git 属性，保持原始字节并让原 LFS 数据作为本仓库普通文件保存；不上传 `.git` 或本机构建产物。来源和逐文件 SHA256 见 `MIDLAYER_MANIFEST.json`，版本入口见快照根目录 `.orbitserve_snapshot.json`。

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

该入口运行真实 CPU Scheduler 状态机和解析计算模型，不加载权重或运行 GPU executor。此快照原有 Git-LFS 数据已逐项展开并校验后随仓库保存；无需单独运行 `git lfs pull`。原生消费者的完整性校验保持不变。
