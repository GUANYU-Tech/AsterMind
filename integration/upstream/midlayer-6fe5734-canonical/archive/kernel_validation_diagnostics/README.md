# Kernel validation diagnostic archive

本目录保留旧 predictor 并行实现时期的一次性 Oracle、DRAM 与 dispatch 调参脚本，仅供
追溯历史结论。它们依赖已退出活动路径的模块，不属于当前 evaluation API，也不纳入
默认编译和测试。新的误差分析必须基于 `evaluation/kernel/` observation 与
`src/kernel/` predictor contract 实现。
