# AsterMind 星鉴前端 V1.0

## Linux

使用 x86-64 Linux。以下以 Ubuntu 24.04 为例，安装 Git、C/C++ 编译器、Python 3.12、Node.js 和 npm。

```bash
sudo apt update
sudo apt install git build-essential python3.12 python3.12-venv nodejs npm
```

进入 AsterMind V1.0 的 Git 工作区根目录，继续[安装项目依赖](#安装项目依赖)。

## Windows

在 x64 Windows 上用 WSL 2 运行仿真服务。首次安装时，在管理员 PowerShell 中运行以下命令，按提示重启并完成 Ubuntu 用户初始化。[微软 WSL 安装文档](https://learn.microsoft.com/zh-cn/windows/wsl/install)

```powershell
wsl --install -d Ubuntu-24.04
```

从 Windows 开始菜单打开 Ubuntu 24.04。在 Ubuntu 终端执行上面的 [Linux 系统依赖安装命令](#linux)，将 AsterMind V1.0 仓库克隆到 Ubuntu 文件系统中，然后在仓库根目录继续以下步骤。服务启动后，在 Windows 浏览器访问页面。

## 安装项目依赖

在仓库根目录创建 Python 虚拟环境，安装仿真依赖和 vLLM 0.31 CPU 调度依赖。默认 Qwen3-4B 场景使用 vLLM 0.31。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install \
  'numpy==2.2.6' 'astropy==6.1.3' 'astropy-iers-data==0.2026.5.11.1.8.52' \
  'sgp4==2.27' 'pydantic>=2.7,<3' 'jsonschema>=4.18,<5' 'PyYAML>=6,<7'
python -m pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu \
  -r astermind/engine/compute/scheduler_runtime/requirements-vllm-0.31.lock
```

需要 vLLM 0.11 的场景可新建独立环境，将 `pip install --no-deps` 命令中的锁文件换成
`requirements-vllm-0.11-cpu.lock`。在启动服务之前，将 `ORBITServe_VLLM011_PYTHON` 设为该独立环境的 Python 路径，即可在同一服务中运行双池和默认单池编队。vLLM 0.30 的配置方法见 [AsterMind 安装说明](../astermind/README.md#setup)。

安装 Three.js，并复制完整场景和计算 Cell 页面加载的文件。三维预览需要支持 WebGL 2 的浏览器。

```bash
npm install --prefix viz --no-save --no-package-lock three@0.180.0
three_dir=viz/node_modules/three
mkdir -p viz/cell/vendor
cp "$three_dir"/build/three.module.js "$three_dir"/build/three.core.js viz/assets/
cp "$three_dir"/build/three.module.min.js "$three_dir"/build/three.core.min.js viz/cell/vendor/
cp "$three_dir"/examples/jsm/controls/OrbitControls.js viz/assets/
cp "$three_dir"/examples/jsm/controls/OrbitControls.js viz/cell/vendor/
```

品牌字体可选，浏览器默认使用系统字体。需要还原任务查询页字体时，将下载的 OTF 文件放到
`viz/cell/fonts/`。文件名为 `Switzer-Regular.otf`、`Switzer-Bold.otf`、
`MiSans-Regular.otf` 和 `MiSans-Semibold.otf`。
[Switzer](https://www.fontshare.com/fonts/switzer) · [MiSans](https://hyperos.mi.com/font/download)

## 启动服务并运行实验

在仓库根目录运行。首次启动会自动编译所需的 C/C++ 扩展。

```bash
source .venv/bin/activate
export ORBITSERVE_ISLSIM_ROOT="$PWD/astermind/builder/islsim"
python -B -m viz
```

在浏览器打开[项目首页](http://localhost:8766/)或[英文首页](http://localhost:8766/en/)，点击「进入仿真平台」选择工作区。首页与平台共用语言设置。也可以访问[健康检查](http://localhost:8766/health)，确认返回的 `status` 为 `ok`。[Windows 浏览器访问 WSL 服务](https://learn.microsoft.com/zh-cn/windows/wsl/networking)

- **计算 Cell**。打开[计算 Cell](http://localhost:8766/cell/)，沿用默认 Qwen3-4B 配置，预览拓扑后进入任务配置并运行仿真。完成后在仿真结果页查看报告。
- **完整场景**。从首页进入工作区，在物理场景的「计算星座与编队」中设置 Cell 数量，逐个编辑编队或将一个 Cell 的配置作为整座模板，生成拓扑数据后在场景编辑页面配置业务请求。保存场景并启动仿真，完成后在实验结果页查看报告。

两个工作区都可以选择 Detail Mode 或 Fast Mode。Fast Mode 由 vLLM 0.30 调度，需要在启动服务时设置 `ORBITServe_VLLM030_SOURCE` 和 `ORBITServe_VLLM030_PYTHON`。需要显存投影的组合，可把对应硬件、并行布局和请求上限的策略 JSON 放到 `viz/.runtime/memory-policies/`，刷新页面后在调度参数中选择。

两个工作区的任务均可持续运行至完成。旧编队配置中的 `runtime.timeout_seconds` 会被忽略。完整场景的「仿真时长」定义轨道和链路需要模拟的时间跨度。

在启动终端按 `Ctrl+C` 停止服务。运行产物保存在 `viz/.runtime/`。
