# Midlayer repository contract

Midlayer 是 LLM 推理性能仿真器，不是通用 Python 软件包。本文只保留不可违反的仓库底线；README 和 `docs/` 负责设计说明、操作流程与导航。

用户当前任务的明确指令优先。根 `AGENTS.md` 是唯一强制规范，嵌套规范只能收紧；README、文档、注释和现有代码均不能覆盖它。用户可在当前任务中明确解除需要确认的限制；其他禁止项只有在用户明确要求修改本规范时才可例外。

## 文档导航

- `README.md` 提供仓库入口和常用命令；`docs/serving_system.md` 与 `docs/kernel_performance_model.md` 说明 Serving System 和 Kernel 性能预测两层设计；`docs/dev_progress.md` 记录当前能力和计划。
- `docs/metax_c500/profiling.md` 规定 C500 profiler 流程；canonical 数据的具体 schema、字段和编码见对应目录 README 与 manifest。文档只负责说明和操作指导，冲突时以本规范为准。

## 目标与推进速度

- 主 agent 为独立 worker 撰写 Goal 时必须使用唯一统一模板 `docs/goal_template.md` 并填写 hard acceptance gates；除非用户明确豁免，任一 gate 未通过都只能报告 checkpoint/blocked，不能标记 completed，模板中的通用完成语义不得省略或另建专用模板替代。
- Goal 只写事实、目标和红线，引用的指标、case、路径和分支必须在撰写时核实有效。不得预设根因或指定候选机制，主 agent 的推断须标注为推断并允许 worker 推翻；已知失败先例、已否定候选和环境前置只作证据，不得转写成对解法的限定。跨硬件共享模型、零拟合来源、owner 边界和 fail-closed 限定的是结果形态而非路径，仍须写明。
- 每个 gate 必须写明取得该指标的方式，现有工具产不出时说明如何得到（为报告增加分组统计输出不算修改评估口径）。数值目标若直接指向某个可调偏移量则不得设，改为验收机制来源、覆盖率、方向一致性或结构不变量——误差百分比靶子会诱导按误差凑参数，与零拟合底线冲突。
- 每项操作必须直接服务于当前 prompt/Goal 的交付物或验收条件；无法说明直接关系的全仓盘点、历史比较、SHA 清点、预防性重构和额外报告不得执行。
- 普通任务应在五分钟内完成必要定位并进入实现；连续十五分钟没有代码、可运行结果、精度改善或明确 blocker 时，停止扩展调查并收缩到最小可交付路径。
- 面向最终可运行、可比较的结果推进。先闭合一个最小但真实的端到端 case，再扩展模型、case、硬件和细节覆盖；不得为了预先穷举或完美实现每个中间 step，长期阻塞可运行主链路。
- 实现深度按当前目标和证据需求控制。先交付能验证核心假设的最小完整路径，再把未覆盖能力显式标为 partial、unsupported 或后续工作；不得以“框架更完整”为由无限扩大任务范围。
- 动手前先估收益上限：用最终评判这件事的同一指标，算出该残差被完全关闭时最多能带来多少变化，上限可忽略就不做，转去主要矛盾。上限估不出来时，先做能估出它的最小实验，而不是直接进入实现。
- 精度优化先按总误差贡献排序，优先处理对目标指标影响最大的算子、阶段、shape 或机制；低贡献长尾只能在主要误差源已有结论后处理。每轮优化必须说明预期影响，并用同一评估口径给出实际变化。
- 非阻塞问题不得中断主目标，也不得用 silent fallback 伪造完成。
- 严格审计只用于 canonical 数据发布或替换、schema/核心 contract、HWSpec、硬件真值、破坏性操作及用户明确要求的审计任务。普通代码、文档、命名、局部公式和测试修复不得默认进行全仓历史审计、全文件 SHA 清点、无关分支比较或 provenance 重建。

## Owner 与依赖

- `src/`：kernel、launch selection、Serving 和全部生产预测公式。
- `evaluation/` 负责 case、observation、metric、comparison、gate 和 report，不得复制 predictor 公式；`dataset_pipeline/` 负责 profiler 采集、解析、转换、统计和一致性校验，不得产生预测值或 import predictor/Serving。
- `scripts/`：薄 CLI 与硬件 `*_runner.py`；活动代码不得 import `scripts`。
- `tests/` 按 owner 镜像；`configs/` 只存人工配置；`microbench/` 只存独立微基准；`archive/` 只供追溯，活动代码不得 import 或执行。

允许依赖为 `scripts → src/evaluation/dataset_pipeline`、`evaluation → src`、`dataset_pipeline → src contract/HWSpec/launch derivation`、`tests → 对应 owner 或 CLI subprocess`；`src` 不得反向依赖外围层，`evaluation` 与 `dataset_pipeline` 互不依赖。复用实现必须进入真实 owner，禁止顶层公共抽屉。单 kernel evaluation 与 Serving 统一调用 `src.kernel.kernel_base.predict(KernelSpec, HWSpec, method)`；硬件差异只由 HWSpec 和接口一致、互不横向 import 的 hardware launch package 表达。

## Forward 零拟合底线

- Production Forward 是零训练数据的解析模型。公式、参数和离散 launch 决策只能来自硬件规格/架构文档、编译器或算子源码的静态语义、与 evaluation workload 独立的机制微基准，以及只描述 variant、tile、grid/block、split 等离散 launch 事实的版本化 observation。
- 禁止用 production 算子、kernel 或 Serving workload 的任何完整或分域 duration 训练、回归、调参、插值、建立 LUT 或 residual correction；也禁止根据 MAPE/MAE 选择参数、特征、阈值、分段点或公式。holdout、LOFO 和 cross-validation 只能评估，不能使拟合合法化。
- 微基准只能测量 launch/synchronization、latency/bandwidth、并发容量等独立机制，并输出有单位、物理含义和可复测来源的参数；不得复刻完整 production 算子后把其时延用于 Forward。
- Profiler observation 可提供 launch、traffic、occupancy、cache hit 和 pipe duty 等非 duration 证据；禁止 per-shape、per-kernel-name 经验时延和静默 fallback。无法用独立物理证据解释的误差应保留并报告 WONTFIX/unsupported。
- `oracle` 和显式 ML baseline 只能在 evaluation 边界消费 observation，其真值、拟合参数和中间时间不得进入 Forward、HWSpec 或 production launch selector。修改 Forward 时只审计本次新增或变更参数的来源；发现违规依赖必须停止扩展，不能再用一层拟合掩盖。

## Contract 与数据

- Contract、配置和 dataset 必须在 owner 边界完整校验；活动 Python 禁止 `.get(...)`、`setdefault`、带默认值的 `pop/getattr`、捕获 `KeyError` 后继续或缺 key 补默认值。Optional 字段必须由 schema 定义并在边界规范化；分派必须穷举，禁止 fallback、吞异常和伪造成功。
- Commit SHA、文件 SHA256 和 source snapshot identity 只在 canonical provenance 或第三方源码绑定的唯一 owner 数据中维护。测试不得重复硬编码同一真实 SHA，也不得因普通 predictor、公式、YAML 或 Serving 实现变化更新成批 SHA；测试只校验字段格式、引用可解析、内容地址关系及篡改时 fail-closed。测试 fixture 可使用明显的合成占位值。
- `kernel_dataset/` 与 `launch_data/` 只保存 Git LFS 跟踪、使用版本化 canonical schema 的数据。Launch observation 不得包含或消费 duration；lookup key 只含物理发射字段，同一物理 key 对应不同 launch 时必须失败。具体格式由对应 README 和 manifest 定义。
- `serving_dataset/` 使用统一版本化 schema 并由 Git LFS 跟踪；Forward 只读请求、动态输入和正常配置，`alignment/` 真值只供 evaluation。未观测到 prefix-cache 命中 H 时不得写成 observed zero；性能等价代表值必须有结构化证据。
- Profiler 产物只写 `profiler_output/`，evaluation 记录只写 `experiment_logs/`，`src/` 禁止数据 payload。未经明确授权，不得删除、重写或批量移动 canonical 数据或 `pipeweave/`。

## 文件、命名与文档

- 优先扩展已有 owner；新增顶层目录、owner、组织轴、跨 owner 公共模块、第二套 schema 或 production `KernelKind` 必须先获用户确认。复用 canonical 名称和 `vendor_model` hardware ID，禁止模糊名称和同义别名。
- Profiler workload 脚本使用 `*_runner.py`，薄 CLI 使用准确的动作或职责名称。`microbench/<hardware_id>/` 按 `compute`、`memory`、`operators`、`runtime` 及其真实资源/机制组织，`calibration/` 和 `references/` 只保存支撑证据。
- `src/kernel/predictors/protobridge/whitebox_surrogate_v2.py` 是唯一获准保留上游命名与实现习惯的第三方原样源码，必须与 `PROVENANCE.md` 的 commit、源码及许可证 SHA256 一致；例外不覆盖任何 Midlayer 自有 adapter、配置、调用方或测试。
- 活动 `docs/` 不新增 session、handoff、临时 TODO 或一次性诊断记录；明确标记的历史快照只作追溯。
- Markdown 自然语言段落在源码中保持为单个物理行，不按显示宽度手工断行；标题、列表、表格、引用、公式和代码块仍按 Markdown 结构分行。

## 硬件执行

| 硬件 | 访问与环境 | 工程路径 |
|---|---|---|
| `nvidia_a800` | 本地；vLLM 使用 Conda `r2r_qwen3_next` | `/share_data/yangxinhao/chip_modeling/midlayer` |
| `metax_c500` | `ssh muxi`；profiler 容器 `yxh_c500_profile`；vLLM 容器 `yxh_vllm_serving_c500` | `/mnt/disk2/yxh/midlayer_hw_model` |
| `ascend_910b` | `ssh 910b-1`；容器 `yxh_midlayer_serving_910b1` | host `/gpfs/midlayer_yxh`，容器 `/mnt/gpfs/midlayer_yxh` |

- 三硬件 vLLM 源码参考位于 `/share_data/yangxinhao/chip_modeling/upstream`（按来源仓库和版本组织；`vllm_src_3ver/` 保留为指向它的兼容符号链接，供既有记录中的路径继续解析）。C500 profiler、采集和数据处理必须遵守 `docs/metax_c500/profiling.md`。
- 执行前确认约定容器和设备空闲，并部署、记录 canonical checkout 的精确 commit 或 snapshot。远程目录只作执行副本；不得自行换容器、占用他人设备或从远程副本反向合并代码。
- 硬件任务记录 hardware、owner、device、baseline、command、PID/tmux 和 output；每张卡仅一个 owner，结束时清理相关进程。硬件执行只生成 raw/candidate，canonical 数据由主 agent 校验并发布。

## Git、Worktree 与 Agent

- 只有位于 canonical checkout `/share_data/yangxinhao/chip_modeling/midlayer` 且在 `main` 分支上工作的 agent 才是主 agent，并且只有它能操作 `main` 和发布 canonical 数据；任何其他 checkout、worktree 或分支中的 agent 都是 worker，即使其会话自称主 agent 也禁止修改 `main`。修改前只检查工作树、目标 owner、直接相关实现和测试并保留无关用户改动；未经当前任务明确授权，禁止 push、force-push、远端分支或 PR。
- 两个以上真正独立的写任务可并行；共享 owner、schema/config、核心 contract 或有先后依赖的修改必须串行集成。每个 writer 只修改分配的 worktree 和文件范围，不得 merge、rebase、push 或发布 canonical 数据。
- 新建 worktree 时记录 Goal、owner、baseline 和收尾条件；结束时只能选择审阅集成后清理、否决后清理或经用户确认保留并记录下一步，禁止遗留无人负责的 dirty worktree 和半成品分支。Worker 完成或受阻时提交可复用成果并报告范围、接口变化、验证和风险。
- 主 agent 只对可能导致重采、canonical 数据替换、schema/runner 语义变化或破坏性操作的结论做独立复核，不重复普通实现审计。
- 后台任务按预计时长以 30 秒至 15 分钟间隔轮询；无失败、完成、实质变化或用户介入需求时不重复汇报。
- 用户要求巡检时逐 pane 判断状态并处置：正在工作的不打断也不汇报；因 API 错误、额度或超时停止的直接发 continue；停在提示符只是跑完一轮的发一条推进指令；提出需要主 agent 裁决的问题时先在仓库核实其陈述再给实质答复，不复述自己的推理过程；报告 checkpoint 或完成的按验收标准评估，达标即集成，未达标指出差距让其继续；崩溃或 pane 消失只汇报不自行重启。只有做了干预或存在用户必须知道的事时才输出。
- tmux 交互：文本和回车必须分两次发送，部分 CLI 需要补第二次回车。判断输入行是否有待发内容必须用 `tmux capture-pane -e` 保留转义序列，`❯` 后紧跟 `ESC[2m`（dim）的是自动补全建议、该 pane 实为空闲，正常亮度才是真排队输入。对真排队输入只发回车，绝不追加文本——追加会把用户已排队的指令改坏。
- Worker 跑完一轮停在提示符时优先恢复它的 goal 循环，而不是每轮手写一条指令：两个 CLI 都用 `/goal <目标文本>` 启动，Codex 另有 `/goal resume` 可恢复 paused 或 stalled 的 goal，两者都用 `/goal clear` 关闭当前 goal；目标文本只写目标与红线、不写具体步骤。状态行有 `◎ /goal active` 或 `Pursuing goal` 即为自驱中，不要打断。goal 被误标 `✔ Goal achieved` 会使循环终止、worker 从此只能等喂指令，发现横幅与其正文的未达标结论矛盾时先纠正再重起循环。只有 worker 提出需要主 agent 裁决的问题、或存在它拿不到的跨 worktree 事实与用户决定时，才发具体指令。

## 验证与交付

- 日常快速验证硬上限二十秒，只运行修改文件或小 owner 的语法检查、直接相关 contract/smoke 和 `git diff --check`。普通提交不得自动扩大到整个 owner、repository、全仓测试或 Kernel/Serving evaluation；PipeWeave 仅在显式选择其模式时验证。
- 批量测试、evaluation 和硬件实验中相互独立的 case、模型与硬件必须并行执行，禁止逐项串行等待。大型 Serving 实验按 case 分配独立 worker，并根据可用 CPU、内存和硬件资源尽可能并行；跨 A800、C500、910B 的独立任务应同时在三台机器运行。只有真实数据依赖、设备冲突、内存限制或 C500 profiler 的 host-global 串行约束允许降低并发，并须在启动时说明原因。
- 同一精确 commit 已通过且相关依赖未变化的结果必须复用。失败后只重跑失败项和直接依赖；纯文档、命名或测试清理不运行性能 evaluation。
- Goal 收尾或普通 `main` 集成运行一次全部受维护测试（无路径参数的 `python scripts/run_tests.py`，默认 48 workers）；开发内循环仍只跑直接 contract/smoke，失败后只重跑失败项。所有本地测试命令必须通过该入口执行，由入口固定使用 Conda `r2r_qwen3_next`，禁止依赖 shell 当前的 `python` 或 `pytest`。测试只保留稳定物理不变量、owner 公共 contract、fail-closed 边界和少量端到端 smoke；微基准 runner/parser 实现细节、一次性实验/报告快照、固定 SHA/文本布局以及重复参数穷举不得进入常驻主门禁。性能 evaluation 与代码测试分开：完整 Kernel evaluation 只在 predictor、公式、HWSpec、kernel/launch canonical 数据变化、发布或用户明确要求时运行。Serving 修改默认以 ShareGPT QPS16 作为代表性负载，并按独立 case 和适用硬件并行验证；涉及多卡能力时同时运行单卡 QPS16 与多卡 QPS16。完整 ShareGPT 全 case evaluation 和 Production Trace evaluation 只有在用户明确要求时才运行。
- `slow` 只用于高成本回归，`scheduler` 只用于真实 vLLM Scheduler 子进程；真实 Scheduler E2E 不超过 20 个代表性 nodeid。新增单项超过五秒的测试必须拆成快速 contract 与少量集成 smoke，或明确归入高成本层。
- 当用户要求“看 report”“看报告”“看结果”或使用同义表达时，默认完成或刷新本任务约定产物后，在最终回复开头直接给出对应 report 文件或结果目录的可点击绝对路径；standard report 若分为 Kernel 与 Serving，则分别给出 `kernel_report.md` 和 `serving_report.md`。可按任务需要附带简短说明，但应以方便用户立即打开报告为优先，避免无关或冗长信息。
- 交付时只需报告结果、修改范围、schema/API 变化、已执行与未执行验证、已知风险、未提交改动、commit 和 push 状态。
