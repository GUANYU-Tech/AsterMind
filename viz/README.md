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
`requirements-vllm-0.11-cpu.lock`。vLLM 0.30 的配置方法见 [AsterMind 安装说明](../astermind/README.md#setup)。

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

在浏览器打开[星鉴首页](http://localhost:8766/)。也可以访问[健康检查](http://localhost:8766/health)，确认返回的 `status` 为 `ok`。[Windows 浏览器访问 WSL 服务](https://learn.microsoft.com/zh-cn/windows/wsl/networking)

- **计算 Cell**。打开[计算 Cell](http://localhost:8766/cell/)，沿用默认 Qwen3-4B 配置，预览拓扑后进入任务配置并运行仿真。完成后在仿真结果页查看报告。
- **完整场景**。从首页进入工作区，在物理场景页面生成拓扑数据，再到场景编辑页面配置计算资源和请求。保存场景并启动仿真，完成后在实验结果页查看报告。

在启动终端按 `Ctrl+C` 停止服务。运行产物保存在 `viz/.runtime/`。
