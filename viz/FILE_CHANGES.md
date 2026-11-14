# viz 逐文件说明

本清单只描述本次前端恢复；全部文件位于 viz 内。
“原样恢复”表示与本地保留版本复制时的 SHA-256 一致；“恢复并适配”表示其后修改了接口、路径、语义或呈现。
新增文件与原基线静态报告文件分别标注。旧功能文件保留不代表旧引擎可以在 V1 执行，具体限制见 README。

| 文件 | 性质 | 用途与原因 |
| --- | --- | --- |
| `.gitignore` | 新增 | 排除运行数据、缓存、字体及本地第三方库。 |
| `README.md` | 恢复并适配 | 说明 WSL/OS 启动方式、后端来源与兼容边界。 |
| `__init__.py` | 基线保留 | 前端模块。 |
| `__main__.py` | 新增 | 支持 python -B -m viz 在 WSL/OS 启动。 |
| `assets/analysis_embed.css` | 原样恢复 | 保留 analysis_embed 主题、布局与控件样式，维持原界面外观。 |
| `assets/analysis_embed.js` | 原样恢复 | 分析 / 嵌入交互模块；恢复对应页面操作与数据呈现。 |
| `assets/cell_model.js` | 原样恢复 | 计算 Cell / 模型交互模块；恢复对应页面操作与数据呈现。 |
| `assets/cell_topology.css` | 原样恢复 | 保留 cell_topology 主题、布局与控件样式，维持原界面外观。 |
| `assets/cell_topology.js` | 原样恢复 | 计算 Cell / 拓扑交互模块；恢复对应页面操作与数据呈现。 |
| `assets/cell_visual.js` | 恢复并适配 | 计算 Cell / 可视化交互模块；恢复对应页面操作与数据呈现。 |
| `assets/collective_scene.js` | 恢复并适配 | 集合通信 / 场景交互模块；恢复对应页面操作与数据呈现。 |
| `assets/communication_clock.js` | 恢复并适配 | 通信 / 时钟交互模块；恢复对应页面操作与数据呈现。 |
| `assets/constellation_layout.js` | 原样恢复 | constellation / 布局交互模块；恢复对应页面操作与数据呈现。 |
| `assets/current_result.js` | 原样恢复 | 当前实验 / 结果交互模块；恢复对应页面操作与数据呈现。 |
| `assets/demo_pointer.js` | 原样恢复 | 示例 / 交互指引交互模块；恢复对应页面操作与数据呈现。 |
| `assets/earth.jpg` | 原样恢复 | 原三维地球贴图资源，保持原显示效果。 |
| `assets/event_timeline.js` | 原样恢复 | 事件 / 时间线交互模块；恢复对应页面操作与数据呈现。 |
| `assets/event_view.js` | 恢复并适配 | 事件 / 展示交互模块；恢复对应页面操作与数据呈现。 |
| `assets/fault_state.js` | 原样恢复 | 故障 / 状态交互模块；恢复对应页面操作与数据呈现。 |
| `assets/fault_view.js` | 恢复并适配 | 故障 / 展示交互模块；恢复对应页面操作与数据呈现。 |
| `assets/global_multiflow.js` | 原样恢复 | 全局 / 并发流交互模块；恢复对应页面操作与数据呈现。 |
| `assets/globe.js` | 恢复并适配 | 三维地球交互模块；恢复对应页面操作与数据呈现。 |
| `assets/globe_links.js` | 原样恢复 | 三维地球 / 链路交互模块；恢复对应页面操作与数据呈现。 |
| `assets/globe_picking.js` | 原样恢复 | 三维地球 / 拾取交互模块；恢复对应页面操作与数据呈现。 |
| `assets/guide.html` | 恢复并适配 | Astermind · 技术文档页面；保留原布局、入口及功能区域。 |
| `assets/islsim_editor.js` | 原样恢复 | ISLSim / 编辑交互模块；恢复对应页面操作与数据呈现。 |
| `assets/live-interaction.css` | 原样恢复 | 保留 live-interaction 主题、布局与控件样式，维持原界面外观。 |
| `assets/live.css` | 原样恢复 | 保留 live 主题、布局与控件样式，维持原界面外观。 |
| `assets/live.js` | 恢复并适配 | 在线运行交互模块；恢复对应页面操作与数据呈现。 |
| `assets/live_event_cursor.js` | 原样恢复 | 在线运行 / 事件 / 游标交互模块；恢复对应页面操作与数据呈现。 |
| `assets/live_panel_feedback.css` | 原样恢复 | 保留 live_panel_feedback 主题、布局与控件样式，维持原界面外观。 |
| `assets/live_panel_feedback.js` | 原样恢复 | 在线运行 / 面板 / 状态反馈交互模块；恢复对应页面操作与数据呈现。 |
| `assets/live_playback_state.js` | 原样恢复 | 在线运行 / 播放 / 状态交互模块；恢复对应页面操作与数据呈现。 |
| `assets/live_runtime.js` | 恢复并适配 | 在线运行 / 运行时交互模块；恢复对应页面操作与数据呈现。 |
| `assets/live_scene.js` | 恢复并适配 | 在线运行 / 场景交互模块；恢复对应页面操作与数据呈现。 |
| `assets/live_traffic.js` | 恢复并适配 | 在线运行 / 通信路径交互模块；恢复对应页面操作与数据呈现。 |
| `assets/live_traffic_model.js` | 恢复并适配 | 在线运行 / 通信路径 / 模型交互模块；恢复对应页面操作与数据呈现。 |
| `assets/live_watch.js` | 原样恢复 | 在线运行 / 订阅交互模块；恢复对应页面操作与数据呈现。 |
| `assets/mission.css` | 原样恢复 | 保留 mission 主题、布局与控件样式，维持原界面外观。 |
| `assets/motion_frames.js` | 原样恢复 | 运动 / 数据帧交互模块；恢复对应页面操作与数据呈现。 |
| `assets/motion_state.js` | 原样恢复 | 运动 / 状态交互模块；恢复对应页面操作与数据呈现。 |
| `assets/navigation.js` | 恢复并适配 | 导航交互模块；恢复对应页面操作与数据呈现。 |
| `assets/network_options.js` | 原样恢复 | 网络 / 选项交互模块；恢复对应页面操作与数据呈现。 |
| `assets/pages.css` | 原样恢复 | 保留 pages 主题、布局与控件样式，维持原界面外观。 |
| `assets/physical_editor.js` | 恢复并适配 | 物理 / 编辑交互模块；恢复对应页面操作与数据呈现。 |
| `assets/physical_motion.js` | 原样恢复 | 物理 / 运动交互模块；恢复对应页面操作与数据呈现。 |
| `assets/physical_routes.js` | 原样恢复 | 物理 / routes交互模块；恢复对应页面操作与数据呈现。 |
| `assets/radio_editor.js` | 原样恢复 | 射频 / 编辑交互模块；恢复对应页面操作与数据呈现。 |
| `assets/random_faults.js` | 原样恢复 | 随机 / 故障交互模块；恢复对应页面操作与数据呈现。 |
| `assets/recorded_scene.css` | 原样恢复 | 保留 recorded_scene 主题、布局与控件样式，维持原界面外观。 |
| `assets/recorded_scene.js` | 恢复并适配 | 已保存记录 / 场景交互模块；恢复对应页面操作与数据呈现。 |
| `assets/recovery_demo.js` | 原样恢复 | 恢复 / 示例交互模块；恢复对应页面操作与数据呈现。 |
| `assets/request_batch.js` | 原样恢复 | 请求 / 批量交互模块；恢复对应页面操作与数据呈现。 |
| `assets/result_center.css` | 原样恢复 | 保留 result_center 主题、布局与控件样式，维持原界面外观。 |
| `assets/result_center.js` | 恢复并适配 | 结果 / 中心交互模块；恢复对应页面操作与数据呈现。 |
| `assets/result_count_help.js` | 原样恢复 | 结果 / 计数 / 说明交互模块；恢复对应页面操作与数据呈现。 |
| `assets/result_dashboard.js` | 原样恢复 | 结果 / 仪表板交互模块；恢复对应页面操作与数据呈现。 |
| `assets/result_selection.js` | 原样恢复 | 结果 / 选择交互模块；恢复对应页面操作与数据呈现。 |
| `assets/results_catalog.js` | 原样恢复 | 结果列表 / catalog交互模块；恢复对应页面操作与数据呈现。 |
| `assets/runtime_controls.js` | 恢复并适配 | 运行时 / 控制交互模块；恢复对应页面操作与数据呈现。 |
| `assets/runtime_faults.js` | 原样恢复 | 运行时 / 故障交互模块；恢复对应页面操作与数据呈现。 |
| `assets/runtime_frames.js` | 原样恢复 | 运行时 / 数据帧交互模块；恢复对应页面操作与数据呈现。 |
| `assets/runtime_overlay.js` | 恢复并适配 | 运行时 / overlay交互模块；恢复对应页面操作与数据呈现。 |
| `assets/runtime_scene.js` | 原样恢复 | 运行时 / 场景交互模块；恢复对应页面操作与数据呈现。 |
| `assets/runtime_state.js` | 恢复并适配 | 运行时 / 状态交互模块；恢复对应页面操作与数据呈现。 |
| `assets/runtime_time.js` | 原样恢复 | 运行时 / 时间交互模块；恢复对应页面操作与数据呈现。 |
| `assets/scene_editor.js` | 恢复并适配 | 场景 / 编辑交互模块；恢复对应页面操作与数据呈现。 |
| `assets/scene_replay.js` | 恢复并适配 | 场景 / 回放交互模块；恢复对应页面操作与数据呈现。 |
| `assets/serving_scene.js` | 恢复并适配 | 服务 / 场景交互模块；恢复对应页面操作与数据呈现。 |
| `assets/sharing_view.js` | 原样恢复 | 共享带宽 / 展示交互模块；恢复对应页面操作与数据呈现。 |
| `assets/solar_state.js` | 原样恢复 | 光照 / 状态交互模块；恢复对应页面操作与数据呈现。 |
| `assets/starguide.css` | 原样恢复 | 保留 starguide 主题、布局与控件样式，维持原界面外观。 |
| `assets/studio_common.js` | 恢复并适配 | 工作台 / 公共交互模块；恢复对应页面操作与数据呈现。 |
| `assets/sun_visual.js` | 恢复并适配 | 太阳 / 可视化交互模块；恢复对应页面操作与数据呈现。 |
| `assets/task_manager.js` | 原样恢复 | 任务 / 管理交互模块；恢复对应页面操作与数据呈现。 |
| `assets/time_clock.js` | 原样恢复 | 时间 / 时钟交互模块；恢复对应页面操作与数据呈现。 |
| `assets/tle_importer.js` | 原样恢复 | TLE / 导入交互模块；恢复对应页面操作与数据呈现。 |
| `assets/visual_state.js` | 恢复并适配 | 可视化 / 状态交互模块；恢复对应页面操作与数据呈现。 |
| `assets/web_capture.js` | 原样恢复 | web / 录制交互模块；恢复对应页面操作与数据呈现。 |
| `assets/workspace-overview.css` | 原样恢复 | 保留 workspace-overview 主题、布局与控件样式，维持原界面外观。 |
| `assets/workspace-overview.js` | 原样恢复 | 工作区 / 概览交互模块；恢复对应页面操作与数据呈现。 |
| `assets/workspace.css` | 原样恢复 | 保留 workspace 主题、布局与控件样式，维持原界面外观。 |
| `catalog.html` | 恢复并适配 | OrbitServe · 运行记录页面；保留原布局、入口及功能区域。 |
| `cell/README.md` | 恢复并适配 | 说明 Cell 同源入口与 V1 适配位置。 |
| `cell/THIRD_PARTY.md` | 原样恢复 | 保留资源来源、使用及依赖说明。 |
| `cell/app-path.mjs` | 原样恢复 | 应用 / 路径交互模块；恢复对应页面操作与数据呈现。 |
| `cell/assets/earth-blue-marble.jpg` | 原样恢复 | 原三维地球贴图资源，保持原显示效果。 |
| `cell/cell-studio.css` | 恢复并适配 | 保留 cell-studio 主题、布局与控件样式，维持原界面外观。 |
| `cell/config-edit.mjs` | 恢复并适配 | 配置 / 编辑交互模块；恢复对应页面操作与数据呈现。 |
| `cell/config-form.mjs` | 原样恢复 | 配置 / 表单交互模块；恢复对应页面操作与数据呈现。 |
| `cell/config-labels.mjs` | 原样恢复 | 配置 / 标签交互模块；恢复对应页面操作与数据呈现。 |
| `cell/config-layout.mjs` | 原样恢复 | 配置 / 布局交互模块；恢复对应页面操作与数据呈现。 |
| `cell/config-validation.mjs` | 原样恢复 | 配置 / 校验交互模块；恢复对应页面操作与数据呈现。 |
| `cell/fonts.css` | 原样恢复 | 保留 fonts 主题、布局与控件样式，维持原界面外观。 |
| `cell/hardware-profile.mjs` | 原样恢复 | 硬件 / 参数交互模块；恢复对应页面操作与数据呈现。 |
| `cell/index.html` | 恢复并适配 | Astermind 星鉴 · 计算 Cell 仿真页面；保留原布局、入口及功能区域。 |
| `cell/link-prediction.mjs` | 原样恢复 | 链路 / 预测交互模块；恢复对应页面操作与数据呈现。 |
| `cell/main.js` | 原样恢复 | 主入口交互模块；恢复对应页面操作与数据呈现。 |
| `cell/mission.css` | 原样恢复 | 保留 mission 主题、布局与控件样式，维持原界面外观。 |
| `cell/presentation.json` | 新增 | 保存旧编辑器 JSON schema 和默认值；不存放模拟运行结果。 |
| `cell/results.mjs` | 恢复并适配 | 结果列表交互模块；恢复对应页面操作与数据呈现。 |
| `cell/scene-data.mjs` | 原样恢复 | 场景 / 数据交互模块；恢复对应页面操作与数据呈现。 |
| `cell/scene-geometry.mjs` | 原样恢复 | 场景 / 几何交互模块；恢复对应页面操作与数据呈现。 |
| `cell/scene.js` | 原样恢复 | 场景交互模块；恢复对应页面操作与数据呈现。 |
| `cell/styles.css` | 原样恢复 | 保留 styles 主题、布局与控件样式，维持原界面外观。 |
| `cell/usage.css` | 原样恢复 | 保留 usage 主题、布局与控件样式，维持原界面外观。 |
| `cell/usage.html` | 原样恢复 | 使用记录 · AsterMind页面；保留原布局、入口及功能区域。 |
| `cell/usage.js` | 恢复并适配 | 使用记录交互模块；恢复对应页面操作与数据呈现。 |
| `cell/web.css` | 原样恢复 | 保留 web 主题、布局与控件样式，维持原界面外观。 |
| `cell/workspace-ui.mjs` | 原样恢复 | 工作区 / ui交互模块；恢复对应页面操作与数据呈现。 |
| `cell_adapter.py` | 新增 | 把原 Cell 表单映射到 V1 公共配置，校验并拒绝无法等价的选项。 |
| `cell_jobs.py` | 新增 | 保存任务、配置、使用记录、超时状态及可分页的真实产物。 |
| `cell_worker.py` | 新增 | 在 OS 子进程运行 V1 Simulation 并保存真实结果。 |
| `cli.py` | 恢复并适配 | 在 Linux/OS 调用当前 V1 物理运行入口。 |
| `collective_runs.py` | 恢复并适配 | 保留旧 Collective 页面查询接口；不启动不兼容的旧引擎。 |
| `collectives.html` | 恢复并适配 | OrbitServe · 编队与跨 cell 集合通信页面；保留原布局、入口及功能区域。 |
| `comparison.py` | 原样恢复 | 保留历史对照记录的读取和显示结构。 |
| `configuration.html` | 恢复并适配 | Astermind 星鉴 · 实验配置页面；保留原布局、入口及功能区域。 |
| `console.html` | 恢复并适配 | OrbitServe · 星座与编队配置页面；保留原布局、入口及功能区域。 |
| `console.py` | 恢复并适配 | 恢复完整场景 HTTP 路由及任务入口，改接 V1 配置/物理/运行接口。 |
| `current_experiment.py` | 原样恢复 | 保存当前选中的实验，串联页面导航。 |
| `demo_v1.py` | 新增 | 将前端演示输入转换成可执行的 V1 场景，重新生成真实数据。 |
| `event_catalog.py` | 恢复并适配 | Presentation categories; raw engine logs remain available for diagnostics. |
| `export.py` | 修改基线文件 | 保留原生报告导出能力，并适配 V1 schema 2、通信事件和审核结构。 |
| `flow_archive.py` | 原样恢复 | SQLite 保存与搜索已执行的路径。 |
| `home.html` | 恢复并适配 | OrbitServe · 实验工作台页面；保留原布局、入口及功能区域。 |
| `index.html` | 修改基线文件 | 原统一首页，提供完整场景和 Cell 两个入口。 |
| `live.html` | 恢复并适配 | OrbitServe · 在线仿真实验台页面；保留原布局、入口及功能区域。 |
| `live.py` | 恢复并适配 | 保留交互会话控制、推进与事件输出结构。 |
| `live_communication.py` | 恢复并适配 | 历史逐跳通信索引；V1 页面使用已适配的真实路径事件。 |
| `live_paths.py` | 恢复并适配 | 维护实际端到端路径及其终态，供查询与回放。 |
| `live_replay.py` | 恢复并适配 | 维护交互回放窗口和活跃传输。 |
| `maintenance.py` | 恢复并适配 | 识别有效 V1 输入与结果；不在启动时自动清理旧目录。 |
| `migration-manifest.json` | 新增 | 记录恢复来源和复制时 SHA-256，便于审阅及对照。 |
| `motion_frames.py` | 原样恢复 | 读取轨道运动帧。 |
| `native_environment.py` | 恢复并适配 | 兼容旧前端的初始化函数名，内部统一进入 V1 环境。 |
| `native_live.py` | 恢复并适配 | 保留物理结果导出；旧独立预设启动明确拒绝。 |
| `native_replay.html` | 新增 | 保留基线原生算子报告页面，避免首页替换影响导出。 |
| `native_result_store.py` | 恢复并适配 | 保存 V1 原始报告及因果审核结果。 |
| `native_scene.py` | 恢复并适配 | 通过 V1 配置、模型目录和场景构造器生成及校验 GPU 场景。 |
| `native_scene_live.py` | 恢复并适配 | 把交互暂停、推进、命令及事件订阅接到 V1 Runtime。 |
| `native_scene_result.html` | 恢复并适配 | OrbitServe · 原生场景结果页面；保留原布局、入口及功能区域。 |
| `paths.py` | 新增 | 约束所有可写路径到 viz/.runtime，满足目录外零修改。 |
| `physical.html` | 恢复并适配 | OrbitServe · 物理模型页面；保留原布局、入口及功能区域。 |
| `physical_frames.py` | 恢复并适配 | 读取物理数据帧供三维预览。 |
| `physical_preview.py` | 恢复并适配 | 物理数据索引、连接预览和路径查询。 |
| `playback.py` | 恢复并适配 | 保留运行记录读取与回放控制结构。 |
| `portal.html` | 恢复并适配 | Astermind 星鉴页面；保留原布局、入口及功能区域。 |
| `progress.html` | 恢复并适配 | OrbitServe · 运行进度页面；保留原布局、入口及功能区域。 |
| `replay.css` | 基线保留 | 保留 replay 主题、布局与控件样式，维持原界面外观。 |
| `replay.html` | 恢复并适配 | OrbitServe · 动态网络检查页面；保留原布局、入口及功能区域。 |
| `replay.js` | 修改基线文件 | 回放交互模块；恢复对应页面操作与数据呈现。 |
| `replay_pages.py` | 原样恢复 | 大型报告按批次读取，避免一次加载全部结果。 |
| `result.html` | 恢复并适配 | OrbitServe · 实验结果页面；保留原布局、入口及功能区域。 |
| `result_analysis.py` | 恢复并适配 | 从真实报告派生请求/算子指标，缺失旧指标保持为空。 |
| `result_dashboard.html` | 原样恢复 | Astermind 星鉴 · 实验结果页面；保留原布局、入口及功能区域。 |
| `results.html` | 恢复并适配 | Astermind 星鉴 · 实验结果页面；保留原布局、入口及功能区域。 |
| `runtime.html` | 恢复并适配 | OrbitServe · Runtime 流量页面；保留原布局、入口及功能区域。 |
| `runtime_environment.py` | 新增 | 用 OS Python 加载当前 HEAD 的 V1 原始源码与既有依赖，校验后端身份。 |
| `runtime_events.py` | 新增 | 把 V1 整条路径通信和请求事件转换为页面展示契约。 |
| `scene.html` | 恢复并适配 | OrbitServe · 实验场景页面；保留原布局、入口及功能区域。 |
| `scene_outputs.py` | 新增 | 从真实 V1 结果生成报告、物理帧和回放，不重新运行业务。 |
| `scene_recording.py` | 恢复并适配 | 按页保存带初始状态的回放事件，将设备路径映射到所属卫星。 |
| `scene_replay.html` | 恢复并适配 | 场景回放 · 已保存记录页面；保留原布局、入口及功能区域。 |
| `scene_validation.py` | 恢复并适配 | 使用 V1 契约校验，明确拒绝旧执行配置。 |
| `server.py` | 恢复并适配 | 保留简单启动入口，不安装目录外命令。 |
| `serving.html` | 原样恢复 | OrbitServe · 端到端并行推理页面；保留原布局、入口及功能区域。 |
| `serving_runs.py` | 恢复并适配 | 保留旧 Serving 页面查询接口；不启动不兼容的旧引擎。 |
| `studio.py` | 恢复并适配 | 保存、读取及列举物理模型和场景。 |
| `task_cleanup.py` | 恢复并适配 | 前端任务清理实现，支持 V1 配置及依赖检查。 |
| `tasks.py` | 恢复并适配 | 读取任务进程及运行状态，约束到前端工作目录。 |
| `templates/fault_recovery.json` | 原样恢复 | 保留 fault_recovery 输入模板；执行前须通过 V1 配置校验，非结果数据。 |
| `templates/fault_workload.json` | 原样恢复 | 保留 fault_workload 输入模板；执行前须通过 V1 配置校验，非结果数据。 |
| `templates/interactive_demo.json` | 原样恢复 | 保留 interactive_demo 输入模板；执行前须通过 V1 配置校验，非结果数据。 |
| `templates/sharing_demo.json` | 原样恢复 | 保留 sharing_demo 输入模板；执行前须通过 V1 配置校验，非结果数据。 |
| `templates/starlink_550.json` | 原样恢复 | 保留 starlink_550 输入模板；执行前须通过 V1 配置校验，非结果数据。 |
| `tle_import.py` | 恢复并适配 | TLE 检查、样本读取与导入表单接口。 |
| `transmission_recording.py` | 恢复并适配 | 索引实际通信事件及忙碌区间，支持 V1 路径共享语义。 |
| `validated_scene_cache.py` | 原样恢复 | 缓存已经通过配置校验的场景摘要。 |
| `verify_backend.py` | 新增 | 留下 V1 原生后端实际运行验证脚本。 |
| `verify_frontend.py` | 新增 | 检查 HTTP、Python 语法、报告一致性和目录外文件哈希。 |
| `viewer.py` | 原样恢复 | 物理三维预览页面及数据连接。 |
| `web_server.py` | 新增 | 统一场景及 Cell 的同源 HTTP、上传、静态资源和任务接口。 |
| `workbench.html` | 恢复并适配 | Astermind 星鉴 · 完整场景页面；保留原布局、入口及功能区域。 |
| `worker.py` | 新增 | 调用 V1 物理数据准备入口，保存进度与结果。 |
| `workspace.py` | 恢复并适配 | 给页面装配原统一导航和主题。 |

| `FILE_CHANGES.md` | 新增 | 逐文件说明用途、修改性质及原因。 |
| `RESTORATION.md` | 新增 | 记录验证证据、结论与不能等价恢复的功能。 |

本地忽略文件：`assets/three.module.js`、`assets/three.core.js`、`assets/OrbitControls.js`、`cell/vendor/` 和 `cell/fonts/` 提供原三维及字体依赖，仅本地保留。
`viz/.runtime/` 中的依赖副本、输入、上传、日志、数据库、截图及报告均不属于源代码提交。
