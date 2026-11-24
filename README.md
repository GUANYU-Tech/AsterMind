# AsterMind

<p align="center">
  <img src="assets/astermind-logo.png" alt="AsterMind logo" width="144">
</p>

English · [简体中文](README.zh-CN.md)

**End-to-end simulation of LLM serving on low Earth orbit constellations**

AsterMind simulates how satellites serve inference requests from users on Earth. It follows requests from gateway ingress through model execution to token delivery, with satellite motion, changing link availability, and contention for network capacity on a shared timeline.

[Technical Report](#technical-report) · [Features](#features) · [Build and Install](#build-and-install) · [Quick Start](#quick-start) · [Contributing](#contributing) · [Contact](#contact)

## Technical Report

**AsterMind Technical Report** — Public PDF link pending.

## Features

- **Simulation scope.** L2 models LLM inference in one multi-satellite Cell. L3 models constellation-wide LLM serving from gateway ingress to inference and response delivery.
- **Simulation modes.** Detail Mode models individual operators for fine-grained simulation. Fast Mode estimates groups of operators per batch. L2 supports both modes. L3 supports Fast Mode only.
- **Model deployment.** Configure tensor, pipeline, and expert parallelism and map ranks to GPUs and satellites.
- **Network and metrics.** Model changing links, routing, and bandwidth sharing. Inspect TTFT, TPOT, and end-to-end latency.

## Architecture

![AsterMind architecture showing workload replay, offline network snapshots, scheduling, Cell execution, and communication](assets/astermind-architecture-en.png)

## Build and Install

Start from x86-64 Ubuntu 24.04 with internet access and a user who can run `sudo`.

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates
git clone --branch version/AsterMind-V1.0 https://github.com/GUANYU-Tech/OrbitServe.git AsterMind
cd AsterMind

bash scripts/install-ubuntu-deps.sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/build.sh
```

The build script compiles the C and C++ helpers, prepares the vLLM source checkout, and installs the frontend assets. The simulator runs on the host CPU without model weights.

## Quick Start

### File Structure

```text
astermind/builder/                Scenario configuration, orbits, and physical datasets
astermind/engine/compute/         Model execution and serving
astermind/engine/comm/            Collective communication and transfer simulation
viz/                             Local frontend for L2 and L3
```

### Open the Frontend

Start the local service. The 3D previews require a browser with WebGL 2.

```bash
bash scripts/run.sh
```

Open the [Chinese homepage](http://localhost:8766/) or the [English homepage](http://localhost:8766/en/) and enter the simulation platform. Check [service health](http://localhost:8766/health) for an `ok` status.

| Simulation scope | Frontend | Example |
| --- | --- | --- |
| **L2 · Single Cell** | [Compute Cell Simulation (计算 Cell 仿真)](http://localhost:8766/cell/) | [DeepSeek-V4-Pro browser walkthrough](#l2--deepseek-v4-pro). |
| **L3 · Constellation** | [Full Scenario Simulation (完整场景仿真)](http://localhost:8766/workspace) | Example pending. |

### L2 · DeepSeek-V4-Pro

The browser walkthrough uses the vLLM 0.30 CPU scheduler to simulate **DeepSeek-V4-Pro** with **TP8 × PP4** across four satellites. Each satellite has eight simulated H200 GPUs. The L2 task models inference and communication inside the Cell. It uses a zero-byte client endpoint for request injection and completion.

1. Open [Compute Cell Simulation](http://localhost:8766/cell/) and select **New Task (新建任务)**.
2. In **Resources (星座资源)**, select **NVIDIA H200**, **four satellites**, and **eight GPUs per satellite**. Use a ring with **1 km** spacing.
3. In **Network (通信网络)**, use one **eight-GPU SU group per satellite** and fixed **400 Gbps** inter-satellite links.
4. In **Model and Workload (模型负载)**, select **deepseek-v4-pro** and **Fast Mode**. Use two synthetic requests with **16 input tokens** and **2 output tokens** each.
5. In **Parallel Deployment (并行部署)**, keep **Colocated (同池部署)** and choose **Manual Configuration (手动配置)**. Set **TP 8, PP 4, DP 1** with EP off. Map each stage to a separate satellite.
6. In **Scheduling (调度运行)**, set **Model Length Limit (模型长度上限)** to **1152 tokens**, choose **deepseek-v4-h200-memory.json** as the **Memory Policy (显存策略)**, and turn off **Chunked Prefill (分块 Prefill)**. Select **Submit Task (发布任务)**. Open **Simulation Results (仿真结果)** to see both completed requests and the compute and communication timelines.

### L3 · Constellation Simulation Example

Sample dataset and run instructions pending.

## Contributing

Bug reports, fixes, and runnable scenarios are welcome. For an issue, include your AsterMind commit, scenario or minimal input, run command, and expected and observed behavior. For a pull request, describe the change and include the checks you ran.

## Contact

- Maintainer 1 — [Name], [Email]
- Maintainer 2 — [Name], [Email]
