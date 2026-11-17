# 计算 Cell 前端

从本机保留的完整 Cell 编辑器恢复，外观与六步操作流程保留。
由 `../web_server.py` 在 `/cell/` 提供页面与同源接口；执行适配位于
`../cell_adapter.py`、`../cell_jobs.py`、`../cell_worker.py`。

运行 `python -B -m viz`，详见上级 README。不要再启动旧 mid-layer Web API。
`presentation.json` 只保存编辑器 schema 与初始值，运行数据均由 V1 实际生成。
旧功能不兼容项及 device-only 请求边界说明见上级 README 和网页技术文档。
