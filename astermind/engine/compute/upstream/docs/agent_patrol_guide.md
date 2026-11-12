# Agent 巡检手册（主 agent 专用）

多 agent 巡检、tmux 交互与 worker goal 循环的操作流程。仅主 agent 在承担巡检/调度任务时阅读；普通 worker 无需加载本文件。

## 巡检处置

用户要求巡检时逐 pane 判断状态并处置：正在工作的不打断也不汇报；因 API 错误、额度或超时停止的直接发 continue；停在提示符只是跑完一轮的发一条推进指令；提出需要主 agent 裁决的问题时先在仓库核实其陈述再给实质答复，不复述自己的推理过程；报告 checkpoint 或完成的按验收标准评估，达标即集成，未达标指出差距让其继续；崩溃或 pane 消失只汇报不自行重启。只有做了干预或存在用户必须知道的事时才输出。

## tmux 交互

文本和回车必须分两次发送，部分 CLI 需要补第二次回车。判断输入行是否有待发内容必须用 `tmux capture-pane -e` 保留转义序列，`❯` 后紧跟 `ESC[2m`（dim）的是自动补全建议、该 pane 实为空闲，正常亮度才是真排队输入。对真排队输入只发回车，绝不追加文本——追加会把用户已排队的指令改坏。

## Worker agent 工具

Worker 可用三种 CLI：Kimi Code（`kimi`）、Codex（`codex`）、Claude Code（`claude`）。三者都在分配的 worktree（统一命名 `wt-<task>-<date>`，目录为仓库同级的 `midlayer-wt-<task>-<date>`）中以 `GOAL.md` 为任务说明启动；选型只看各 CLI 当前的额度、网络稳定性与任务匹配度，无强约束时沿用同类任务已验证可用的工具。自驱能力按工具区分：Kimi，Codex，Claude Code 都有内置 `/goal <目标文本>` 循环，Codex 另有 `/goal resume` 可恢复 paused 或 stalled 的 goal；启动 prompt 可要求它按 `GOAL.md` 持续自驱直到 hard gates 通过。

## Worker goal 循环

Worker 跑完一轮停在提示符时优先恢复它的 goal 循环，而不是每轮手写一条指令：用 `/goal <目标文本>` 启动、用 `/goal clear` 关闭当前 goal；目标文本只写目标与红线、不写具体详细的实现方案步骤，给执行的worker以充足的探索空间和发挥空间。Kimi 状态行有 `◎ /goal active`、Codex 有 `Pursuing goal` 即为自驱中，不要打断。goal 被误标 `✔ Goal achieved` 会使循环终止、worker 从此只能等喂指令，发现横幅与其正文的未达标结论矛盾时先纠正再重起循环。只有 worker 提出需要主 agent 裁决的问题、或存在它拿不到的跨 worktree 事实与用户决定时，才发具体指令。
