#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "$0")" && pwd)"
repo="$(cd -- "$here/../.." && pwd)"
mode_file="$here/.environment"
company_root="/data1/$USER/orbitserve-l2-env"
verda_root="$HOME/orbitserve-l2-experiments-env"
vllm_commit=ced6857afa0ea7b2e3f0846a62e1394e90f15607

usage() {
    echo "Usage: bash tests/L2/l2.sh setup {company|verda}" >&2
    echo "       bash <experiment>/run.sh [--smoke|--validate]" >&2
    exit 2
}

set_runtime() {
    if [[ "$1" == company ]]; then
        marker="$company_root/ring600-ready"
        if [[ ! -s "$marker" ]]; then
            echo "The existing company CPU environment was not found: $marker" >&2
            exit 1
        fi
        IFS= read -r env_dir < "$marker"
        python="$env_dir/orbitserve-conda/bin/python"
        sdk_env="$env_dir/vllm030-conda"
        sdk_source="$env_dir/vllm030-source"
        results_root="/data3"
    else
        env_dir="$verda_root"
        sdk_env="$env_dir/venv"
        python="$sdk_env/bin/python"
        sdk_source="$env_dir/vllm030-source"
        results_root="$HOME/orbitserve-l2-results"
    fi
    export ORBITServe_VLLM030_PYTHON="$sdk_env/bin/python" ORBITServe_VLLM030_SOURCE="$sdk_source"
}

install_verda() {
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates curl git gcc g++ libnuma1 tmux
    mkdir -p -- "$env_dir"
    if [[ ! -d "$sdk_source/.git" ]]; then
        git -c advice.detachedHead=false clone --single-branch --depth 1 --branch v0.30.0 \
            https://github.com/vllm-project/vllm.git "$sdk_source"
    fi
    test "$(git -C "$sdk_source" rev-parse HEAD)" = "$vllm_commit"
    uv="$env_dir/tools/uv"
    if [[ ! -x "$uv" ]]; then
        curl -fsSL https://astral.sh/uv/0.11.14/install.sh | UV_UNMANAGED_INSTALL="$env_dir/tools" sh
    fi
    export UV_PYTHON_INSTALL_DIR="$env_dir/python"
    if [[ ! -x "$python" ]]; then
        "$uv" python install 3.12
        "$uv" venv --python 3.12 --managed-python --seed "$sdk_env"
    fi
    arch="$(uname -m)"
    wheel="https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcpu-cp38-abi3-manylinux_2_39_$arch.whl"
    "$python" -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        'torch==2.13.0+cpu' 'numpy==2.2.6' "$wheel" -e "$repo"
    wheel_pkg="$("$python" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')"
    shopt -s nullglob
    for library in "$wheel_pkg"/*.so; do
        ln -sfn -- "$library" "$sdk_source/vllm/"
    done
    shopt -u nullglob
}

verify_runtime() {
    if [[ ! -x "$python" || ! -x "$ORBITServe_VLLM030_PYTHON" || ! -d "$sdk_source/.git" ]]; then
        echo "CPU environment is incomplete. Check the setup output." >&2
        exit 1
    fi
    test "$(git -C "$sdk_source" rev-parse HEAD)" = "$vllm_commit"
    iomp="$(find "$sdk_env" -name libiomp5.so -print -quit)"
    if [[ -n "$iomp" ]]; then
        export LD_PRELOAD="$iomp${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
}

export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$repo"
unset ORBITSERVE_MIDLAYER_ROOT ORBITServe_MIDLAYER_ROOT
cd "$repo"

case "${1:-}" in
    setup)
        [[ $# -eq 2 && ( "$2" == company || "$2" == verda ) ]] || usage
        set_runtime "$2"
        if [[ "$2" == verda ]]; then install_verda; fi
        verify_runtime
        "$python" -m astermind.common.build_native
        VLLM_TARGET_DEVICE=cpu PYTHONPATH="$sdk_source" "$ORBITServe_VLLM030_PYTHON" -c \
            'from vllm.v1.core.sched.scheduler import Scheduler; print("vLLM 0.30 CPU Scheduler ready")'
        printf '%s\n' "$2" > "$mode_file"
        echo "L2 environment ready ($2)"
        ;;
    run)
        [[ $# -ge 2 && -f "$2" ]] || usage
        case_path="$(realpath -- "$2")"
        shift 2
        if [[ ! -s "$mode_file" ]]; then
            echo "Run the L2 setup command first." >&2
            exit 1
        fi
        IFS= read -r platform < "$mode_file"
        set_runtime "$platform"
        verify_runtime
        if [[ -f "$repo/.l2-source-commit" ]]; then
            commit="$(cat -- "$repo/.l2-source-commit")"
        else
            commit="$(git -C "$repo" rev-parse HEAD)"
        fi
        family="$(basename -- "$(dirname -- "$(dirname -- "$case_path")")")"
        case_id="$(basename -- "$(dirname -- "$case_path")")"
        timestamp="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S%N)"
        mkdir -p -- "$results_root"
        out="$results_root/l2-$family-$case_id"_"$timestamp"_"$USER"
        mkdir -- "$out"
        printf '%s\n' "$commit" > "$out/source-commit.txt"
        echo "Results: $out"
        "$python" -u "$here/run.py" "$case_path" "$out" "$commit" "$@" 2>&1 | tee "$out/run.log"
        ;;
    *) usage ;;
esac
