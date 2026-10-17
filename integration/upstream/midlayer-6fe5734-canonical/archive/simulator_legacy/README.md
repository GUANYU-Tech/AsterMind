# Timeloop simulator legacy archive

本目录保存已退出活动主路径的 Timeloop 数据集生成链路，仅用于追溯既有
UCB Gemmini 与 Groq LPU ground truth 的来源。活动评估直接读取
`kernel_dataset/{ucb_gemmini,groq_lpu}/`，不再调用 Timeloop。

- `configs/`：归档生成器所需的 Timeloop 二进制、环境与硬件参数快照。
- `scripts/`：历史数据集生成入口；默认路径仍可能写入受保护的数据集，禁止在
  未备份和未审核输出路径时直接运行。
- `timeloop/`：architecture、mapspace、mapper 和 GEMM problem template。
- `validation/`：历史搜索诊断脚本及结果。

这些文件不属于生产 `src/`、活动 `scripts/` 或活动 `configs/`，也不纳入默认
测试与编译流程。若要复现实验，需要先按本机安装位置修改归档配置中的
`timeloop_bin` 与 `timeloop_env_prefix`。
