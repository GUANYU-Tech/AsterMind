# L2 实验

每个实验点有自己的 `case.json` 和 `run.sh`。直接运行该目录的脚本就会生成完整场景、执行仿真并保存结果。

距离扫描的三个 1 km 点和同部署的 1000 Gbps 带宽点共用结果。以下是 23 个独立运行目录。

| # | 实验组 | 部署 | 拓扑 | Gbps／方向 | km | 目录 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 1 | GLM 扩容 | TP8／DP4／EP32 | Ring | 400 | 1 | [`ring400`](glm52-expansion/ring400/) |
| 2 | GLM 扩容 | TP8／DP4／EP32 | Ring | 600 | 1 | [`ring600`](glm52-expansion/ring600/) |
| 3 | GLM 扩容 | TP8／DP4／EP32 | FC | 400 | 1 | [`fc400`](glm52-expansion/fc400/) |
| 4 | DeepSeek 带宽 | TP8／PP4 | Ring | 200 | 1 | [`tp8-pp4-200gbps`](deepseek-v4-bandwidth/tp8-pp4-200gbps/) |
| 5 | DeepSeek 带宽 | TP8／PP4 | Ring | 400 | 1 | [`tp8-pp4-400gbps`](deepseek-v4-bandwidth/tp8-pp4-400gbps/) |
| 6 | DeepSeek 带宽 | TP8／PP4 | Ring | 600 | 1 | [`tp8-pp4-600gbps`](deepseek-v4-bandwidth/tp8-pp4-600gbps/) |
| 7 | DeepSeek 带宽 | TP8／PP4 | Ring | 800 | 1 | [`tp8-pp4-800gbps`](deepseek-v4-bandwidth/tp8-pp4-800gbps/) |
| 8 | DeepSeek 带宽 | TP8／PP4 | Ring | 1000 | 1 | [`tp8-pp4-1000gbps`](deepseek-v4-bandwidth/tp8-pp4-1000gbps/) |
| 9 | DeepSeek 距离 | TP8／PP4 | Ring | 1000 | 0.5 | [`tp8-pp4-0.5km`](deepseek-v4-distance/tp8-pp4-0.5km/) |
| 10 | DeepSeek 距离 | TP8／PP4 | Ring | 1000 | 10 | [`tp8-pp4-10km`](deepseek-v4-distance/tp8-pp4-10km/) |
| 11 | DeepSeek 距离 | TP8／PP4 | Ring | 1000 | 100 | [`tp8-pp4-100km`](deepseek-v4-distance/tp8-pp4-100km/) |
| 12 | DeepSeek 带宽 | TP16／PP2 | Ring | 200 | 1 | [`tp16-pp2-200gbps`](deepseek-v4-bandwidth/tp16-pp2-200gbps/) |
| 13 | DeepSeek 带宽 | TP16／PP2 | Ring | 600 | 1 | [`tp16-pp2-600gbps`](deepseek-v4-bandwidth/tp16-pp2-600gbps/) |
| 14 | DeepSeek 带宽 | TP16／PP2 | Ring | 1000 | 1 | [`tp16-pp2-1000gbps`](deepseek-v4-bandwidth/tp16-pp2-1000gbps/) |
| 15 | DeepSeek 距离 | TP16／PP2 | Ring | 1000 | 0.5 | [`tp16-pp2-0.5km`](deepseek-v4-distance/tp16-pp2-0.5km/) |
| 16 | DeepSeek 距离 | TP16／PP2 | Ring | 1000 | 10 | [`tp16-pp2-10km`](deepseek-v4-distance/tp16-pp2-10km/) |
| 17 | DeepSeek 距离 | TP16／PP2 | Ring | 1000 | 100 | [`tp16-pp2-100km`](deepseek-v4-distance/tp16-pp2-100km/) |
| 18 | DeepSeek 带宽 | TP8／DP4／EP32 | Ring | 200 | 1 | [`wideep-200gbps`](deepseek-v4-bandwidth/wideep-200gbps/) |
| 19 | DeepSeek 带宽 | TP8／DP4／EP32 | Ring | 600 | 1 | [`wideep-600gbps`](deepseek-v4-bandwidth/wideep-600gbps/) |
| 20 | DeepSeek 带宽 | TP8／DP4／EP32 | Ring | 1000 | 1 | [`wideep-1000gbps`](deepseek-v4-bandwidth/wideep-1000gbps/) |
| 21 | DeepSeek 距离 | TP8／DP4／EP32 | Ring | 1000 | 0.5 | [`wideep-0.5km`](deepseek-v4-distance/wideep-0.5km/) |
| 22 | DeepSeek 距离 | TP8／DP4／EP32 | Ring | 1000 | 10 | [`wideep-10km`](deepseek-v4-distance/wideep-10km/) |
| 23 | DeepSeek 距离 | TP8／DP4／EP32 | Ring | 1000 | 100 | [`wideep-100km`](deepseek-v4-distance/wideep-100km/) |

正式运行使用 128 个同时到达的请求，每个 1000 输入、100 输出 token。每星 8 张 H200，星内用 NVLink Surrogate，星间按最小传播时延选路并运行动态带宽竞争。额外逐跳处理时间和启动时间设为 0。每副本最多调度 32 条请求，最大上下文 1152。场景显式关闭分块和前缀缓存，每批 token 预算由代码推导为每副本 36864。DeepSeek V4 使用已试跑的 H200 显存估算配置，WideEP 使用 Direct 专家分发。

## 公司服务器

将发布包传到 `/data1/jiateng` 并解压，从解压后的仓库根目录执行：

```bash
bash tests/L2/l2.sh setup company
bash tests/L2/glm52-expansion/fc400/run.sh
```

安装步骤复用 `/data1/$USER/orbitserve-l2-env/ring600-ready` 指向的已有 Conda 和 vLLM 0.30 CPU 环境，并在当前源码下编译 CPU 辅助库。结果保存在 `/data3/l2-*`。

## Verda

```bash
git clone --single-branch --branch feature/L2_experiment https://github.com/GUANYU-Tech/OrbitServe.git OrbitServe-L2
cd OrbitServe-L2
bash tests/L2/l2.sh setup verda
bash tests/L2/glm52-expansion/fc400/run.sh
```

在 Verda 的 Ubuntu 26.04 root 环境运行安装命令。依赖装在 `~/orbitserve-l2-experiments-env`，结果保存在 `~/orbitserve-l2-results/`。

## 运行选项与结果

- `bash tests/L2/glm52-expansion/fc400/run.sh --validate` 校验正式场景与拓扑。
- `bash tests/L2/glm52-expansion/fc400/run.sh --smoke` 跑 4 条请求，每条 96 入／2 出。
- 省略选项运行正式负载。换一个实验时只需替换脚本路径。
- 运行目录包含 `source-commit.txt`、`config.json`、`run.log`、`batches.jsonl` 和 `result.json`。`batches.jsonl` 逐批写入实际耗时和模拟耗时。最终结果包含实际调度预算、每请求 TTFT／TPOT／E2E 和通信统计。
