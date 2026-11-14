# 前端恢复验收记录

验收日期：2026-11-14。分支：v1.0-lyz-1。V1 基线：ae6ee46a93c3d61be8ed00303b0a990d540f6553。

## 结论

原统一首页、完整场景工作区和 Cell 六步编辑器已恢复到 viz。
前后端同源服务为 http://localhost:8766/，运行于 WSL Linux 的 Conda OS。
前端实际执行使用 V1 源码与官方 vLLM Scheduler，未调用旧 4173 / mid-layer Web 服务。
本次仅修改 viz；未修改 builder、engine、examples、tests 或根目录构建配置；未提交、推送。

这次完成的是界面恢复与 V1 可支持流程的实际接通，不能宣称旧版本所有执行功能已 100% 等价恢复。
以下不兼容项保留明确错误提示，需要另行讨论后端方案，未越过 viz 边界实现。

## 实际运行与页面验证

| 范围 | 结果与证据 |
| --- | --- |
| Cell 六步页面、导航、编队与网络预览 | 浏览器逐步切换；预览返回真实距离、链路容量及传播延迟。 |
| Cell 提交与实际计算 | AM-20261114-b299a2c2464d4797b8ed90a4eccdcadf；Qwen3-4B，2 星，2 个请求均完成；TTFT 0.301636623 / 0.540398727 s。 |
| Cell 报告与产物 | 报告分页、请求、统计、产物预览已检查；大数组使用有界预览，完整数据仍可下载。 |
| Cell 历史与配置复用 | 浏览器从历史任务载入配置回到资源页；使用记录显示来源、硬件、提交配置及 V1 生效配置。 |
| 配置拒绝 | SGLang、AF、FlowSim ACK 与旧双引擎对照返回明确错误；失败提交保存为校验拒绝记录。 |
| TLE 与请求文件 | TLE 上传、V1 SGP4/ITRS 坐标转换已验证；canonical CSV 上传及请求指纹生成通过。 |
| 物理准备 | physical-20261114-113238-8eed45；32 颗卫星、1 地面站、20 秒、10 个物理采样帧；检查通过。 |
| 场景编辑与启动 | 浏览器生成 2 Cell / 8 GPU 的 Qwen3-8B TP2×PP2 部署，添加请求、保存、运行到结束。 |
| 完整场景报告 | native-scene-20261114-115040-609500；请求 1/1 完成，E2E 0.644182 s，TTFT 0.32944 s；请求、图表、算子、KV 校验页可用。 |
| 完整场景复跑 | native-scene-20261114-120246-481632；历史重新运行成功，1/1 请求完成。 |
| V1 因果审核 | 完整场景审核通过：1,920 个计算算子、160 个通信操作、3,882 个依赖、3 个业务传输、1,025 个分配记录、0 个失败 batch。 |
| 三维回放 | 1,830 条通信事件、609 次跨节点通信准入尝试；已检查逐时刻推进、卫星路径绑定。数字表示通信尝试，不是用户请求数或旧 FlowSim 流数。 |
| 在线控制 | 验证暂停、推进和断链命令；恢复连接受 V1 终端可用性约束，后端拒绝时保留真实错误。 |
| 静态资源与接口 | verify_frontend.py 的 38 项 HTTP 检查通过，覆盖主要页面、目录接口、报告接口、身份与结果一致性。 |
| 源码检查 | 50 个 Python 文件 AST 与 80 个 JS/MJS 模块语法检查通过；JS 使用本机 Linux Node 22.20.0。 |
| 目录边界 | 启动前记录的 3,978 个 viz 外已跟踪文件 SHA-256 全部保持一致；工作区已有的未跟踪 docs/ 未动。 |

本次未运行 UC1–UC6，没有新增根目录 tests。没有穷举所有模型、硬件、PD/MoE 组合或长时间压力场景。
删除任务等破坏性按钮没有为验收而删除用户数据。

## 不能等价恢复的旧功能

- SGLang、AF / PD+AF；旧 FlowSim ACK、SimGrid / ns-3 独立执行和双引擎比较。
- MTP、异步 Scheduler、scheduler pipeline depth、旧 Host / KV offload、CUDA Graph、闭环 admission。
- 逐卡异构覆盖、旧 pool KV 分层、自定义跨 pool routes / pipeline_p2p；当前适配器未覆盖 CP / SP / DP / EPLB。
- 原独立 serving / collective 演示运行器。页面保留，但启动明确拒绝；不是假装使用 V1 成功运行。
- 旧逐跳排队/传播动画与统计：V1 使用整条路径共享资源，只展示实际路径和真实起止事件，不伪造逐跳时间。原始 V1 报告完整保留。

Cell 的 device-only 任务使用显式零载荷客户端边界，不计算星地接入时间；完整星地业务走场景工作区。
Cell 运行截止时间为最后请求到达后至少 60 秒，失败/完成由实际 V1 仿真决定。
缺少等价来源的旧指标保持不可用。运行环境目前复用本机依赖，不是可直接分发的独立安装包。

## 文件与复核入口

逐文件用途和修改原因见 FILE_CHANGES.md，恢复来源及初始哈希见 migration-manifest.json。
启动命令、依赖处理及端点见 README.md。

本机证据保存在 .runtime/verification/：frontend-checks.json、javascript-checks.json、compatibility-checks.json、
scene-result.jpg；原始结果在 .runtime/workspace/local/results/network/ 和 .runtime/cell-jobs/。
这些证据、缓存和第三方依赖均不提交 Git。
