# 框架、参考实现与配置的发布范围

## 分开管理

| 内容 | 放置方式 | 基础发布包 |
|---|---|---|
| 通用框架 | 模型、拓扑、运行时、native 接口、FlowSim 复用代码 | 包含 |
| 基础参考 | 静态连接、simple_keep／O-S 参考选择、公平带宽共享、FC／Ring／Switch | 包含 |
| 最简单示例 | `examples/public_basic.json` + `public_transfers.json` | 仅这两个配置文件 |
| 私有研究配置 | `private/` 或外部自选路径 | 排除 |
| 实验拓扑与策略 | `experimental/`，Mesh／8-neighbor／复杂网关或轨道研究配置 | 排除 |
| 运行数据／二进制 | `runs/`、`build/` | 排除；使用者自行构建 |

FC／Ring／Switch 只是图生成便捷方法，不等于假定模型必然使用某种并行方式。自由连接始终保留，不公开研究配置也不妨碍其他用户表达自己的图。

当前没有实现 8-neighbor／Mesh 自动生成器，也没有复制研究场景。它们只在开发记录中标注为**实验性、基础发布时删除／排除**。公开文档不将它们列为已实现能力。

## 本地基础打包

```powershell
python -m network_sim.package_release --output network_sim/build/network_sim_basic.zip
```

脚本按明确的文件允许列表打包框架和最小示例，忽略新增私有配置；不会靠 `.gitignore` 推断哪些文件可以公开。必须修改允许列表才能扩大发布范围。公开包自带源码、测试和构建说明，不带本机路径、构建记录或研究结果。

本次只制作本地发布预览，没有向任何平台发布。源码复用授权／LICENSE 的正式确认见 `reuse.md`；该检查不通过时不能将预览包当作可直接正式发布的项目。

## 内部计算侧交接包

`python -m network_sim.package_release --handoff --output network_sim/build/network_sim_topology_v0.2.zip`。使用独立允许列表增加 `examples/handoff_dynamic.json` 和 Windows x64 DLL。仍不包含研究配置、实验目录、原始项目整体、运行日志或本机路径。基础公开包不含这两项。此交接包是本地内部交付，不是已经向外发布。
