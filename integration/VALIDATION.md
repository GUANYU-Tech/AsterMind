# develop 接入验证（2026-10-17）

在独立仓库副本中使用固定 M-L 快照，并从源码构建 M-L LRU、FlowSim 与 ns-3 后验证。使用已有 Python 3.12 CPU Scheduler 隔离环境，不运行 GPU 实测。

- `python -m simulator.checks`：14 项通过，无跳过。
- `python -m unittest discover -s network_sim/tests -v`：39 项通过；1 项 Windows DLL 交接打包测试在 Linux 下跳过，需要本地 Windows 构建产物。
- `python -m network_sim.build_native --upstream-tests`：12 项原 FlowSim 回归通过。
- `sequential_cells.json`：四 Cell、12 请求全部完成。
- `session_migration_tp2.json`：TP2→TP2 逐 rank KV 迁移完成，目标命中 32 token 前缀。
- 两个端到端产物通过 `simulator.audit` 的计算/通信依赖、数据量和完成时序核查；实际结果与本地构建日志不进入版本库。

另抽查了原仓库模块边界与运行时结果契约：跨域导入规则、源代码公共接口引用和四项结果契约检查通过。原 `develop` 已有两项接口命名断言与实现不一致：`ResolvedHardware`/`resolve_experiment_hardware` 对应实现已经使用 `ResolvedArchitecture`/`resolve_architecture`；`WORKLOAD_BUNDLE_SCHEMA` 也不在当前 workload 的公共导出中。相关源码和测试均未由本次接入修改，因此不将本次验证描述成全仓测试通过。未按正确包映射安装时，原 runtime engine 测试还会缺少 `orbitserve.experiments`；项目正常安装使用已有 pyproject 包映射。

本次新增 `.gitattributes` 规则保持轨道参考数据的原始字节，避免 Windows 换行转换破坏现有 SHA 校验。M-L 使用原始固定提交，不修改其源码；外部版本桥接、launch 选择与精度边界见 `simulator/README.md`。

M-L 后续改为本仓库内完整快照：逐项比较全部项目文件；原 LFS 数据校验真实 payload 的 SHA256 和大小后直接纳入普通 Git 文件，源码与配置仍按原始 Git blob 字节验证。仅快照 `.gitattributes` 的分发属性有调整，详见 `MIDLAYER_MANIFEST.json`。
