"""
Fake CUDA library management: build, configure, and inject via LD_PRELOAD.

Builds libcudart.so.12 / libcublas.so.12 / libnccl.so.2 from C sources and
constructs the environment variables needed to run vLLM under emulation.

Architecture notes (P17.2):
- Single LD_PRELOAD=libcudart.so.12 replaces NVIDIA CUDA runtime + driver API
- libcuda.so.1 is a symlink → libcudart.so.12
- libcublas.so.12 / libcublasLt.so.12 are independent stub .so files
- __cudaPushCallConfiguration/__cudaPopCallConfiguration implement thread-local
  call config stack (required for PyTorch <<<grid,block>>> kernel launch syntax)
- cudaGetDeviceProperties uses offsetof()-verified CUDA 12.8 struct layout
- cudaMalloc returns fake pointers (0x200000000+), tracks allocated bytes
"""

import json
import subprocess
import tempfile
from pathlib import Path

import yaml

EMULATION_DIR = Path(__file__).parent
CSRC_DIR = EMULATION_DIR / "csrc"
PROJECT_ROOT = EMULATION_DIR.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
FAKE_CUDA_DIR = BUILD_DIR / "fake_cuda_libs"
HW_CONFIGS_DIR = PROJECT_ROOT / "configs" / "hw"


def build() -> Path:
    """Build fake CUDA libraries if not already built or source is newer.

    Returns:
        Path to the fake_cuda_libs directory.

    Outputs:
        build/fake_cuda_libs/libcudart.so.12
        build/fake_cuda_libs/libcuda.so.1  (symlink)
        build/fake_cuda_libs/libcublas.so.12
        build/fake_cuda_libs/libcublasLt.so.12
    """
    src_driver = CSRC_DIR / "cuda_emulate.c"
    src_runtime = CSRC_DIR / "cuda_runtime_emulate.c"
    src_cublas = CSRC_DIR / "cublas_emulate.c"
    libcudart = FAKE_CUDA_DIR / "libcudart.so.12"
    libcublas = FAKE_CUDA_DIR / "libcublas.so.12"

    newest_src = max(src_driver.stat().st_mtime, src_runtime.stat().st_mtime)
    if not libcudart.exists() or newest_src > libcudart.stat().st_mtime:
        FAKE_CUDA_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            "gcc", "-O2", "-shared", "-fPIC",
            "-Wl,--version-script=" + str(CSRC_DIR / "libcudart.ver"),
            "-Wl,-soname,libcudart.so.12",
            "-o", str(libcudart),
            str(src_driver),
            str(src_runtime),
            "-lpthread", "-ldl",
        ]
        print("[fake_cuda] Building libcudart.so.12...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Build failed: {result.stderr}")

        cuda_link = FAKE_CUDA_DIR / "libcuda.so.1"
        if cuda_link.exists() or cuda_link.is_symlink():
            cuda_link.unlink()
        cuda_link.symlink_to("libcudart.so.12")
        print("[fake_cuda] Built libcudart.so.12 + libcuda.so.1 symlink")

    if not libcublas.exists() or src_cublas.stat().st_mtime > libcublas.stat().st_mtime:
        FAKE_CUDA_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            "gcc", "-O2", "-shared", "-fPIC",
            "-Wl,--version-script=" + str(CSRC_DIR / "libcublas.ver"),
            "-Wl,-soname,libcublas.so.12",
            "-o", str(libcublas),
            str(src_cublas),
            "-lpthread",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"cublas build failed: {result.stderr}")
        print("[fake_cuda] Built libcublas.so.12")

    src_nccl = CSRC_DIR / "nccl_emulate.c"
    libnccl = FAKE_CUDA_DIR / "libnccl.so.2"
    newest_nccl = max(src_nccl.stat().st_mtime,
                      (CSRC_DIR / "libnccl.ver").stat().st_mtime)
    if not libnccl.exists() or libnccl.stat().st_mtime < newest_nccl:
        FAKE_CUDA_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            "gcc", "-O2", "-shared", "-fPIC",
            "-Wl,--version-script=" + str(CSRC_DIR / "libnccl.ver"),
            "-Wl,-soname,libnccl.so.2",
            "-o", str(libnccl),
            str(src_nccl),
            "-lpthread",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"nccl build failed: {result.stderr}")
        print("[fake_cuda] Built libnccl.so.2")

    return FAKE_CUDA_DIR


def load_emulation_config(hw_config: str) -> dict:
    """Load emulation_config section from a hw_configs YAML.

    Args:
        hw_config: hardware config name (e.g. 'A800-SXM4-80GB') or path to YAML.
    """
    if Path(hw_config).suffix in (".yaml", ".yml"):
        yaml_path = Path(hw_config)
    else:
        yaml_path = HW_CONFIGS_DIR / f"{hw_config}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"HW config not found: {yaml_path}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    if "emulation_config" not in cfg:
        raise KeyError(
            f"No 'emulation_config' section in {yaml_path}. "
            f"Add it to use CUDA emulation with this hardware config."
        )
    return cfg["emulation_config"]


def write_emulation_config_json(emu_cfg: dict, trace_output: str | None = None) -> str:
    """Write emulation config to a temp JSON file for the C library to read.

    Returns absolute path to the written file (caller is responsible for cleanup,
    or rely on _cleanup_stale_emulation_configs() for reaping).
    """
    config_dict = {"gpu": emu_cfg}
    if trace_output:
        config_dict["trace_output"] = trace_output

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="cuda_emu_",
        dir=str(BUILD_DIR), delete=False
    )
    json.dump(config_dict, tmp, indent=2)
    tmp.close()
    return tmp.name


def cleanup_stale_emulation_configs(max_age_seconds: int = 3600) -> None:
    """Remove build/cuda_emu_*.json files older than max_age_seconds."""
    import time
    now = time.time()
    if not BUILD_DIR.exists():
        return
    for fpath in BUILD_DIR.glob("cuda_emu_*.json"):
        try:
            mtime = fpath.stat().st_mtime
        except FileNotFoundError:
            continue
        if now - mtime > max_age_seconds:
            try:
                fpath.unlink()
            except FileNotFoundError:
                continue


def get_env(hw_config: str = "A800-SXM4-80GB",
            trace_output: str | None = None,
            tp_size: int = 1) -> dict:
    """Return environment variables for running a process under CUDA emulation.

    Builds fake libs if needed (including libnccl.so.2 for TP > 1), writes
    emulation config JSON, and constructs LD_PRELOAD / LD_LIBRARY_PATH /
    CUDA_EMULATE_CONFIG entries.

    Args:
        hw_config: HW config name or path to YAML.
        trace_output: Override CUDA_TRACE_OUTPUT path.
        tp_size: Tensor-parallel degree. If > 1, libnccl.so.2 is added to
                 LD_PRELOAD and VLLM_NCCL_SO_PATH is set.
    """
    fake_dir = build()
    cleanup_stale_emulation_configs()

    emu_cfg = load_emulation_config(hw_config)
    config_json_path = write_emulation_config_json(emu_cfg, trace_output)

    preload_libs = [
        str(fake_dir / "libcudart.so.12"),
        str(fake_dir / "libcublas.so.12"),
        str(fake_dir / "libcublasLt.so.12"),
    ]
    if tp_size > 1:
        preload_libs.append(str(fake_dir / "libnccl.so.2"))

    env = {
        "LD_PRELOAD": ":".join(preload_libs),
        "LD_LIBRARY_PATH": str(fake_dir),
        "CUDA_EMULATE_CONFIG": config_json_path,
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "fork",
    }
    if tp_size > 1:
        env["VLLM_NCCL_SO_PATH"] = str(fake_dir / "libnccl.so.2")
    if trace_output:
        env["CUDA_TRACE_OUTPUT"] = trace_output

    return env
