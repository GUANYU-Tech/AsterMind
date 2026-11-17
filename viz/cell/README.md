# 计算 Cell 前端

从本机保留的完整 Cell 编辑器恢复，外观与六步操作流程保留。
由 `../web_server.py` 在 `/cell/` 提供页面与同源接口；执行适配位于
`../cell_adapter.py`、`../cell_jobs.py`、`../cell_worker.py`。

运行 `python -B -m viz`，详见上级 README。不要再启动旧 mid-layer Web API。
`presentation.json` 只保存编辑器 schema 与初始值，运行数据均由 V1 实际生成。
旧功能不兼容项及 device-only 请求边界说明见上级 README 和网页技术文档。

结果页沿用独立 Cell 前端在 `5f91d59` 之前的 System、Rank、算子、通信、内存分栏结构，
恢复指标卡、区间分解、时间线和曲线。`cell_result_view.py` 只读取已保存的 V1 报告，
通过 `/cell/api/simulations/<job>/analysis` 提供图表摘要和有界算子 / 时间线分页。
不修改执行器或预测器。任务完成用时与仿真截止时间分开；并行区间取并集，
不把算子耗时相加为 Wall time。当前报告没有完整临时 Tensor 生命周期与 Roofline
利用率时，使用真实 KV 事件、权重驻留和算子构成展示，不填充演示数值。
