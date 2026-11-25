# <img src="assets/astermind-logo.png" alt="" width="36" align="absmiddle"> AsterMind

English · [简体中文](README.zh-CN.md)

**End-to-end simulation of LLM serving on low Earth orbit constellations**

AsterMind simulates how satellites serve inference requests from users on Earth. It follows requests from gateway ingress through model execution to token delivery, with satellite motion, changing link availability, and contention for network capacity on a shared timeline.

[Technical Report](#technical-report) · [Features](#features) · [Overview](#overview) · [Build and Install](#build-and-install) · [Quick Start](#quick-start) · [Contributing](#contributing) · [Contact](#contact)

## Technical Report

**AsterMind Technical Report** — Public PDF link pending.

## Features

- **Simulation scope.** L2 models LLM inference in one multi-satellite Cell. L3 models constellation-wide LLM serving from gateway ingress to inference and response delivery.
- **Simulation modes.** Detail Mode models individual operators for fine-grained simulation. Fast Mode estimates groups of operators per batch. L2 supports both modes. L3 supports Fast Mode only.
- **Model deployment.** Configure tensor, pipeline, and expert parallelism and map ranks to GPUs and satellites.
- **Network and metrics.** Model changing links, routing, and bandwidth sharing. Inspect TTFT, TPOT, and end-to-end latency.

## Overview

![AsterMind architecture showing workload replay, offline network snapshots, scheduling, Cell execution, and communication](assets/astermind-architecture-en.png)

## Build and Install

**Prerequisites.** x86-64 Linux with glibc 2.39 or newer (for example, Ubuntu 24.04), Git, a C/C++ toolchain, and `libnuma.so.1`. The virtual environment uses Python 3.12. The setup script can download Python 3.12 or Node.js with npm when needed and uses `curl` to fetch tools. The examples run on the CPU without model weights or a GPU.

```bash
git clone --branch version/AsterMind-V1.0 https://github.com/GUANYU-Tech/OrbitServe.git AsterMind
cd AsterMind

bash scripts/install-ubuntu-deps.sh
bash scripts/create-venv.sh
source .venv/bin/activate
```

Install the Python dependencies and build AsterMind.

```bash
python -m pip install -r requirements.txt
bash scripts/build.sh
```

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

Open the [homepage](http://localhost:8766/en/).

| Simulation scope | Frontend | Example |
| --- | --- | --- |
| **L2 · Single Cell** | [Compute Formation Simulation](http://localhost:8766/cell/) | [DeepSeek-V4-Pro](#l2--deepseek-v4-pro-example) |
| **L3 · Constellation** | [End-to-End Simulation](http://localhost:8766/workspace) | Example pending. |

### L2 · DeepSeek-V4-Pro Example

Run **DeepSeek-V4-Pro** in **Detail Mode** with **TP8 × PP4** across four satellites with eight simulated H200 GPUs each.

1. Open [Compute Formation Simulation](http://localhost:8766/cell/), switch to **EN**, and select **New Task**.
2. In **Constellation Resources**, select **NVIDIA H200**, **four satellites**, and **eight GPUs per satellite**. Set the formation to **Ring** with **1 km** spacing.
3. In **Communication Network**, use one **eight-GPU SU group per satellite** and set the inter-satellite links to **Fixed-Bandwidth Mode** at **400 Gbps**.
4. In **Model and Workload**, select **deepseek-v4-pro** and **Detail Mode**. Set two synthetic requests at **16 requests/s**, each with **16 input tokens** and **2 output tokens**.
5. In **Parallel Deployment**, select **Colocated** and **Manual Configuration**. Set **TP 8, PP 4, DP 1** with EP off. Map each pipeline stage to a separate satellite.
6. In **Scheduling and Execution**, set **Maximum Model Context Length** to **1152 tokens** and turn off **Chunked Prefill**. Select **Submit Task**. Open **Simulation Results** to view both requests and their compute and communication timelines.

### L3 · Constellation Simulation Example

Sample dataset and run instructions pending.

## Contributing

Bug reports, fixes, and runnable scenarios are welcome. For an issue, include your AsterMind commit, scenario or minimal input, run command, and expected and observed behavior. For a pull request, describe the change and include the checks you ran.

## Contact

- Maintainer 1 — [Name], [Email]
- Maintainer 2 — [Name], [Email]
