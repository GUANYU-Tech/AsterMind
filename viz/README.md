# Astermind 星鉴前端 · V1.0 适配

完整场景与计算 Cell 前端位于同一个 `viz` 目录、使用同一个 HTTP 服务。
保留原界面的布局、页面与编辑流程；执行适配使用当前分支的 AsterMind V1 后端。
本次恢复没有修改 `viz` 外的源文件，没有提交或推送。

## 在 WSL / Linux / OS 启动

```bash
conda activate OS
cd /mnt/d/OrbitServe
python -B -m viz
```

打开 http://localhost:8766/ 。计算 Cell 入口为 http://localhost:8766/cell/ 。
健康信息为 http://localhost:8766/health ，显示实际 Python、平台、源码位置和提交。
关闭服务使用启动终端的 Ctrl+C；先让当前任务完成或通过页面停止任务。
当前交互会话不能在服务重启后续算，已保存配置与报告仍可读取和重新运行。
无需启动旧的 4173 服务，也不使用 Windows Python。

## 源码与依赖

本次验证基线为 `ae6ee46a93c3d61be8ed00303b0a990d540f6553`，分支 `v1.0-lyz-1`。
`runtime_environment.py` 从当前 Git HEAD 导出 `astermind`、`workloads`、`examples`
到 `viz/.runtime/sources/<commit>`。这份源码字节与提交一致，未修改算法。
这样避免当前工作区换行转换导致 V1 内置数据校验不一致，同时不改动目录外文件。
因此，未提交的后端源码改动不会被该入口加载；前端文件修改则直接生效。

所有 Python 执行来自 WSL 的 Conda OS。当前 OS 缺少的依赖只读复用本机已安装目录，
搜索顺序见 `runtime_environment.py`；可用 `ASTERMIND_FRONTEND_DEPS` 指定依赖目录，
或将兼容依赖放入 `viz/.runtime/python`。V1 所需的两个 C/C++ 扩展使用原源码在 viz/.runtime 内自动编译，并记录源文件哈希。需要本机 cc / g++；不会把编译产物写入目录外。
这不是独立安装包：换机器需要准备 V1 所需的依赖和编译扩展。
ISLSim / GSLSim 使用本机已配置的依赖路径，只读加载。

Three.js、字体等第三方资源仅保留本地，并由 `viz/.gitignore` 排除。
新机器需要另行准备这些前端资源；本次不将 third-party 内容提交 Git。

## 页面与运行流程

- `/`：统一首页；`/workspace`：完整场景概览。
- `/configuration`、`/physical`：物理场景编辑、TLE 导入、物理数据生成与预览。
- `/scene`：模型、GPU/Rank、请求、故障编辑；保存、批量运行及交互运行。
- `/live`：实际 V1 时间线、拓扑、事件、暂停/推进和故障操作。
- `/results`、`/experiments`：报告、运行记录、下载、回放与重新运行。
- `/cell/`：原六步 Cell 编辑器、三维预览、配置校验、任务运行、报告和产物。
- `/cell/usage`：任务查询、详情、提交配置与生效配置。
- `/assets/guide.html`：当前适配说明。

配置、上传文件、缓存、日志、运行结果全部写入 `viz/.runtime`，不写入旧 `local`。
Cell 结果来自 V1 原始报告，缺少的旧指标显示不可用，不以演示值替代。
原来的静态原生报告入口保存在 `native_replay.html`，导出时仍提供独立报告。

## 适配语义与明确限制

完整场景默认使用 V1 官方 vLLM Scheduler、Runtime Comm，以及
`IntraSatelliteRoofline` 卫星内通信模型。默认参数改变是为了真实接入 V1，
不代表旧后端执行语义与 V1 完全相同。

Cell 的旧 device-only 请求被映射到显式零载荷地面边界；不计算星地接入耗时。
仿真截止时间为最后一个请求到达后至少 60 秒。请求的实际完成/失败仍由 V1 决定。
完整星地业务请使用完整场景工作区。流数表示真实通信准入尝试数，不能与旧
FlowSim 流数直接作等价比较。

下列旧功能当前不能等价运行，接口会明确拒绝或页面说明不可用：
SGLang、AF/PD+AF、旧 FlowSim ACK、SimGrid/ns-3 独立后端、旧双引擎对照、
MTP、异步 Scheduler、旧 Host/KV offload、CUDA Graph、闭环 admission、
逐卡异构硬件覆盖、自定义跨 pool routes，以及当前适配器未覆盖的 CP/SP/DP/EPLB。
旧独立 serving/collective 演示入口不能作为 V1 运行器使用。
这些功能需要分别确认 V1 扩展方案，不能通过恢复页面声称已经实现。

V1 通信事件描述整条路径共享资源，回放显示路径与真实开始/结束时刻；
不会伪造旧 FlowSim 的逐跳排队、到达及重传时间。旧逐跳传输索引不能等价复用。
交互断链/恢复仍受 V1 终端占用、链路获取和时间边界约束，拒绝时显示真实原因。

文件来源见 `migration-manifest.json`；逐文件职责见 `FILE_CHANGES.md`。
验证证据与边界检查保存在 `viz/.runtime/verification/`；恢复结论见 `RESTORATION.md`。
