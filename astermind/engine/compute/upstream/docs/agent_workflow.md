# Agent 操作流程

本文承接根 [AGENTS.md](../AGENTS.md) 中按场景触发的流程。只在执行对应动作时阅读相关章节；本文不独立增加权限、审批或验收范围，冲突时以根规范和用户当前任务为准。普通代码或文档修改不需要执行整份手册。

## 分工与 worktree

主 agent 负责本地 `main`、`dev` 的审阅、整理与集成；其他 agent 只在自己分配的 worktree 和对应分支工作。所有 Git 写操作只限本地，禁止 push、远端分支/标签修改和 PR 操作。

1. 新 feature 从明确的 `dev` commit 建立独立分支与 worktree，记录 Goal、owner、baseline 和收尾条件。分支命名为 `wt-<task>-<date>`，目录放在仓库同级的 `midlayer-wt-<task>-<date>`；不再新增 `worker/*` 等前缀，历史命名随对应任务收尾处理。
2. 独立 Goal 使用 [唯一模板](goal_template.md)：writer 在分配的 worktree 保存本地 `GOAL.md`，只读 Goal 可用任务消息承载，不为此写共享 checkout 或新建 worktree；短小只读咨询只需明确问题、范围和输出。核实 case、路径、分支和指标，推断允许推翻，失败先例不限定解法；写清首版、最终 gates、测量方式和必要环境。本地 Goal 默认不提交。
3. 各 worker 在自己的 worktree 中独立修改任务所需代码，共同文件也可同时修改；不等待文件锁、hunk 审批、其他 worker 完成或合并后再开工。主 agent 负责合并、冲突解决及组合验证；worker 只说明改动与依赖，不自行 merge/rebase、修改他人 worktree/分支或发布 canonical。
4. Worker 完成或受阻时提交可复用成果，说明接口变化、验证、未完成项和风险。主 agent 复用已有审阅证据；只有涉及重采、canonical 替换、schema/runner 语义或破坏性操作的结论需要独立复核，不重复完整实现审计。

## 集成与存储

### 代码集成与目录收尾

- 按最终能力整理提交，核实基础版、移植版和修复之间的等价关系，不把重复实现当成不同 feature；提交粒度按功能和可审阅性决定，遵守本轮约定的集成方式。
- 集成后处理来源 worktree：确认 owner/任务进程已停止，未提交和未跟踪的独有内容已处置，必要代码、证据及引用可追溯。共享软链接不按副本搬运，也不跟随链接删除目标；canonical 和 pinned raw 继续按数据规则保留。
- 结束时选择审阅集成后清理、否决后清理，或经用户确认保留并记录原因和下一步；本轮已有保留指令即可，不重复请示。没有明确处置的 dirty worktree 或半成品分支不能当作已收尾。

### 按产物规模记录空间

| 场景 | 记录范围 |
|---|---|
| 普通代码/文档集成 | 检查本轮新增文件和非预期大文件；没有数据产物就写“无数据新增”，不扫描全仓或共享 Git/LFS |
| 引入 canonical、raw 或大产物 | 按本轮涉及的目录/campaign 记录来源、目标、大小、用途、必须保留的原因、可再生性，以及主仓库有无等价内容 |
| 删除 worktree | 核实独有内容、活跃任务和引用；报告实际释放空间，不把软链接目标或仍被共享的对象算作释放 |

需要详细计量时，在动作前后记录时间、涉及目录的逻辑字节数与实际磁盘占用，例如分别使用 `du --apparent-size --block-size=1 -s <目录>` 和 `du --block-size=1 -s <目录>`，不加跟随软链接的选项。只比较本轮相关路径；按代码/文档、`experiment_logs/`、`profiler_output/`、canonical 分类，记录新增、压缩、迁移和删除。

Git/LFS 是共享存储。仅在本轮确实引入相关对象且需要归因时记录其变化；并发 worker、硬链接或共享对象导致无法确定的部分说明限制，不把全库增长归给当前集成，也不展开历史对象审计来补齐数字。

可再生日志不因清理 worktree 而整包搬回。必要结果保留最小交付物及再生命令；不可再生证据放 `profiler_output/` 并登记。发布 canonical 或整理 raw 时遵守 [数据登记与生命周期](profiler_output_registry.md) 的相关章节。

## 硬件执行与收尾

本节只在已授权范围内指导执行，不构成设备或 Modal 使用授权。设备、容器、可写工程路径和付费边界以根规范为准。以下固定环境检查与目录步骤适用于本地/SSH 机器；Modal 的启动、容器临时目录和停止方式按 [Modal 使用说明](modal_readme.md)，同样记录运行身份、版本和命令，并回传数据。

1. **开始前：** 确认约定容器和设备空闲、每卡唯一 owner，记录 hardware、owner、device、baseline、command、PID/tmux 和 output。部署并记录 canonical 精确 commit 或经任务允许的 snapshot；远端仅作执行副本。
2. **任务隔离：** 在指定远程工程路径下建立 `<worktree 名或任务名>-<日期>` 目录，与本地命名对齐。代码、采集 raw、日志和中间产物全部放入本任务目录；不复用他人目录。`/tmp` 只作为本次临时工作区，结束清空。
3. **执行：** 从最小 case 验证主链，再按目标扩展。独立任务按已获授权直接并行，不新增全局 CPU/worker/GPU 并发上限或统一排队。C500 profiler 确有 host-global 冲突时只协调相关采集，其余任务继续；实际资源不足同样局部处理。设备访问范围和付费预算保持原授权，不以并发为由自行扩额或换未授权型号。
4. **收尾：** 拉回结果并按 raw/candidate/可再生结果分类；清理本任务进程、临时目录和代码副本。Pinned raw 必须拉回、本地只压不删；其余产出按 campaign 累计达到 200MB 时拉回或告知用户。结束时确需留远端的内容列清单，交用户决定；硬件 worker 不直接发布 canonical。

三硬件 vLLM 源码参考在 `/share_data/yangxinhao/chip_modeling/upstream`，按来源仓库和版本组织。C500 profiler 使用 [C500 profiling](metax_c500/profiling.md)；H200 采集和转换参考 [H200 capture](nvidia_h200/capture.md)。文档内的历史机器、目录和运行命令不自动授权新环境或 Modal 执行。

## 验证与报告

### 选择验证范围

本地代码测试统一走 `python scripts/run_tests.py`，入口负责固定 Conda 环境；定向调用方式查 README 或入口帮助，不直接调用 pytest。

| 阶段 | 默认范围 |
|---|---|
| 日常开发内循环 | 修改文件/直接 contract 或 smoke；优先二十秒内，必要时说明原因并采用最小可用范围 |
| 代码 Goal 最终收尾、main 集成或 main 发布候选验收 | 一次无路径参数的 `python scripts/run_tests.py`，默认 96 workers、小批次按需派发（`--dist=load --maxschedchunk=1`）；复用精确 commit 与依赖未变的通过结果，失败只重跑失败项和直接依赖 |
| 普通 dev 集成 | 集成本身不额外触发全套测试；执行直接受影响的验证和本轮验收，每大轮仍交付下述标准报告 |
| Predictor、公式、HWSpec、kernel/launch canonical 变化，发布或用户明确要求 | 完整 Kernel evaluation，另行记录运行成本 |
| Serving 修改 | ShareGPT QPS16；涉及多卡时覆盖单卡和多卡，按独立 case 和适用硬件组织 |
| 完整 ShareGPT 全 case / Production Trace | 仅在用户明确要求时运行 |
| 纯文档、命名或测试清理 | 直接相关检查，不运行性能 evaluation |

常驻测试只留稳定物理不变量、owner 公共 contract、fail-closed 边界和少量端到端 smoke。微基准 runner/parser 实现细节、一次性报告快照、固定 SHA/文本布局和重复参数穷举不进入主门禁。`slow` 用于高成本回归，`scheduler` 只用于真实 vLLM Scheduler 子进程；真实 Scheduler E2E 不超过 20 个代表性 nodeid。新增单项超过五秒的测试拆为快速 contract 与少量集成 smoke，或明确归入高成本层。

### 大轮集成报告

每一大轮功能集成统一交付标准 Kernel 与 Serving report，不逐 commit 重跑。报告不自动扩大到完整 ShareGPT 或 Production Trace，范围沿用根规范与本轮约定；未变化的精确版本结果可复用。

- 固定合入前的比较版本，以及数据、case、硬件范围、环境、并发数和 CPU affinity；若条件不同，明确哪些数值不可直接比较。
- 区分预测时延、仿真器实际 wall time、评估/数据整理和绘图耗时。记录指标定义与聚合方式，避免把逐请求配对误差与平均指标误差混用。
- 标明完成、失败、unsupported 和未运行范围，报告精度及运行速度变化、原因和合理性；墙钟异常先检查资源竞争与复用条件，不能凭单次波动宣布回归。
- 用户要求看结果时，在回复开头给出可点击的报告绝对路径；标准报告分别提供 `kernel_report.md` 与 `serving_report.md`。

## 文件与文档组织

优先复用既有 owner、canonical 名称和 `vendor_model` hardware ID，不引入模糊名称或同义别名。Profiler workload 使用 `*_runner.py`；薄 CLI 按动作或职责命名。`microbench/<hardware_id>/` 按 `compute`、`memory`、`operators`、`runtime` 及其真实资源/机制组织，`calibration/`、`references/` 只保存支撑证据。

`docs/` 根保存常青设计、流程与契约文档；绑定日期、commit 或实验轮次的快照放 `docs/records/` 并标明历史用途，硬件流程放 `docs/<hardware_id>/`。不新增活动 session、handoff、临时 TODO 或一次性诊断文档。Markdown 自然语言段落保持一个物理行，不按显示宽度断行；标题、列表、表格、引用、公式和代码按结构分行。
