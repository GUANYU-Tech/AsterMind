# Modeling research legacy snapshot

本目录保存从仓库外暂存目录
`/share_data/yangxinhao/chip_modeling/docs_sav` 归档的历史研究材料。
归档日期为 2026-08-11，归档基线为 commit
`3b2866cb3749b3a01454ea17bcd0c0fea5fdb362`。

这些文件仅用于追溯，不是活动实现、当前设计规范或 production Forward 的参数来源。
活动代码不得 import 或执行本目录中的 Python 文件。归档时只按职责重新组织路径，并规范了
Markdown 的行尾空白；未改写研究结论或候选代码。

## 内容

- `nvidia_ampere/`：Ampere CTA swizzle 和 SASS 静态分析的研究笔记。
- `synthetic_trace_dse_candidate/`：2026-06-24 整理的合成 trace DSE 候选实现、测试和原始清单。

`synthetic_trace_dse_candidate/` 与现有 `archive/dse_legacy/` 有共同历史，但不是同一份实现；
前者记录当时准备吸收到旧 `dev` 布局的候选版本，因此单独保留。其 import 路径和 API
针对当时的仓库结构，不能据此假定它在当前仓库中仍可运行。
