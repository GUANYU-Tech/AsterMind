# CAL'26 论文图表

本目录重新生成 CAL'26 论文中全部数据图，以及两张表所依据的报告。图 1（overview）是手绘结构图，没有数据来源。

## 用法

```
python scripts/paper_plot/cal26/collect_data.py     # 约 15 分钟，四项采集并行
python scripts/paper_plot/cal26/plot_figures.py     # 读取最近一次采集
```

`python scripts/paper_plot/cal26/collect_data.py --validate-only` 只检查图 5 的 suite 配置、共享输入和 CLI 参数，不启动 evaluation，也不创建结果目录。`python scripts/paper_plot/cal26/plot_figures.py --run-dir <目录> --figure production_tradeoff` 可单独刷新图 5。

论文使用的版本是 tag `cal26-submit-review-ver`；复现论文数字须先检出该 tag。之后的提交（例如 H200 与 DeepSeek-R1 SGLang 对齐）会改变图 5 与 DeepSeek-R1 的模拟结果。

当前 main 的图 5 采集使用已发布的 DeepSeek-V3/R1 recipe suite，选择四种 vLLM setting（colocated/PD × MTP off/on）、C=4/8/16/32/64，共 20 次正式运行；C=1/2 不在当前共享输入中。结果位于 `<run-dir>/dsr1_sweep/vllm_<group>/cN.json`，采集 manifest 版本为 2；绘图显式兼容版本 1 的旧目录与七档并发，未知版本报错。原论文数字仍以论文 tag 为准。

图 5 使用 P90 interactivity 和 input+output throughput，并按部署全部 GPU 归一化：colocated 为 8，PD 为 16。suite 自带曲线的 PD 分母为 8 张 decode GPU，两者不能直接混用。PD 模拟使用 vLLM 后端；InferenceX SGLang 实测对照仅筛选 colocated。SGLang 自身支持 PD，当前 Midlayer 的 SGLang mirror 尚未实现 PD。

`collect_data.py` 写入 `experiment_logs/paper_plot/cal26/<stamp>/`，`manifest.json` 记录生成时的 main commit、工作树是否干净以及每项输入的路径；`plot_figures.py` 默认把图写到该目录的 `figs/`，可用 `--run-dir` 指定采集目录、`--figure-dir` 指定输出目录。

## 图表与数据来源

| 论文 | 输出 | 数据 |
|---|---|---|
| 表 1 Kernel MAPE | `standard_report/kernel_report.md` | 标准报告 Kernel 部分（1027 case，含 Roofline/PipeWeave） |
| 表 2 Serving MAPE | `standard_report/serving_report.md`、`serving_results` | 标准报告 45 个 ShareGPT case，每个设置取三个 QPS 的平均 |
| 图 2 | `memory_traffic.pdf` | `evaluation.kernel.memory_traffic_report` |
| 图 3 | `serving_mechanism_ablation.pdf` | `evaluation.serving.metrics.diagnostics.mechanism_ablation`，910B Qwen3-4B TP1 QPS4 |
| 图 4 | `serving_request_comparison_<model>.pdf` | 标准报告逐请求 TTFT/TPOT；论文使用 qwen3-4b，其余四个模型一并生成 |
| 图 5 | `production_tradeoff.pdf` | DeepSeek-V3/R1 recipe H200 8K/1K 闭环 sweep 与 InferenceX R1 原始测量 `profiler_output/serving/inferencex_alignment/raw/dsr1_all_20260924.json` |
