# Production trace preprocessing provenance

本目录保留生成当前 Production CSV 的历史上游预处理脚本，仅供来源追溯和人工审计，不属于活动 pipeline，仓库代码不得 import 或执行它。

`filter_production_trace.py` 由用户于 2026-08-22 上传，脚本版本为 `2.3.0`，归档时 SHA256 为 `4eea0b59a70c674cca0887a885d4c3e10583739b093aa8a0b809a82bc7234eba`。它从 3 份未入库的敏感 XLSX 中删除明确失败或无有效计算负载的请求，移除状态与错误等列，使用确定性 seed 重建秒内到达顺序，并生成 6 份主 CSV、2 份一秒 debug CSV 及一份处理报告。用户已确认仓库外现存 Production CSV 均来自该成功请求筛选流程。

当前 canonical dataset 保存经严格转换后的四列请求，位于 `serving_dataset/production/` 并通过 Git LFS 跟踪；由于原始数据未记录模型与运行 setting，其中的 latency、TTFT 和 generation speed 不作为 alignment ground truth 发布。原始 XLSX、九列中间 CSV 和处理报告不作为 canonical payload。本归档保存预处理代码；缺少原始 XLSX 时不能独立复现最初的筛选步骤。

该脚本会替换目标目录中的同名输出，并删除其上一次报告登记、本次未重新生成的旧文件，因此不得在仓库或 canonical 数据目录中直接运行。如确需审计复现，只能在隔离的本地临时目录中显式指定输入和输出路径。

当前活动的严格转换实现位于 `dataset_pipeline/serving_dataset/production_raw_adapter.py`，对应薄入口位于 `scripts/evaluation/serving/build_production_candidates.py`；活动转换从已经筛选好的九列 CSV 构建本地 candidate，不负责重新执行本归档中的 XLSX 筛选逻辑。
