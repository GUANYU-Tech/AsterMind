# Modal 使用说明

Modal 用于按需运行云端 GPU 任务。执行前按 [AGENTS.md](../AGENTS.md) 取得用户授权，明确卡型、卡数、时限、GPU-hours 和预算；使用能验证当前目标的最小已授权配置。通用用法查[官方文档](https://modal.com/docs/guide)，价格查[定价页](https://modal.com/pricing)。

先在本地尽量完成最小验证，包括语法、编译、输入输出和可运行的小 case；通过后再提交 Modal。不要把本地能发现的问题留到付费云端调试。本地缺少目标硬件或环境时，明确未验证项，只将必要的最小验证放到 Modal 上执行。

## 本机环境

客户端已安装在 `/home/yangxinhao/.local/share/midlayer-modal-client`，PATH 已加入 `~/.bashrc`，账号已配置，无需重复安装或登录。新终端找不到命令时执行 `source ~/.bashrc`；需要重新登录时使用 `modal setup`，按提示在浏览器完成认证。

使用代理时，在客户端环境安装代理支持依赖：

```bash
/home/yangxinhao/.local/share/midlayer-modal-client/bin/python -m pip install 'modal[api-proxy-support]'
```

## T4 / nsys 配置

使用单张 Tesla T4，容器镜像选 `nvidia/cuda:12.6.3-devel-ubuntu22.04`，添加 Python 3.11，并安装 `cuda-nsight-systems-12-6`。以下命令采集 CUDA trace，关闭 CPU sampling。

这是 T4 采集链路示例，不代表 H200 性能或完整 vLLM 已通过验证。在云端容器的独立任务临时目录准备 `probe.cu`，执行：

```bash
nvcc -O2 -arch=sm_75 probe.cu -o probe
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true -o trace ./probe
nsys stats --report cuda_gpu_kern_sum --format csv trace.nsys-rep
```

`sm_75` 对应 T4。以生成 `.nsys-rep` 且摘要出现预期 kernel 为通过条件，不能只检查 `nsys --version`。每次记录实际 GPU、软件版本、代码版本和命令；采集耗时不等于无 profiler 的性能真值。

## 回传与清理

- 小指标由云端函数 `return`，本地通过 `.remote()` 接收；小报告可读取为 `bytes` 返回，落盘后校验 SHA256。云端普通文件不会自动同步。
- 原始报告和日志取回本地 `profiler_output/`，按[登记规则](profiler_output_registry.md)管理。大文件需要持久存储时，先明确用途、费用和清理方式。
- 按授权限制并发容器数、GPU 数量、重试次数和函数超时，并控制包含构建、排队及清理的总时限。结束时停止本任务应用、清理任务临时目录；异常时可执行 `modal app stop <app-id>`，确认停止成功。
- 本地文件不产生 Modal 存储费；Volume 会在应用停止后继续保留，可能持续计费。短实验无需创建 Volume。费用估算与实际账单分开报告。
