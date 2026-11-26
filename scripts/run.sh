#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
repo=$(pwd -P)
python="$repo/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo 'Create .venv and install requirements.txt before starting AsterMind.' >&2
    exit 1
fi

source "$repo/scripts/conda-runtime.sh"
astermind_use_conda_cxx_runtime "$repo/.venv"

export ORBITSERVE_ISLSIM_ROOT="${ORBITSERVE_ISLSIM_ROOT:-$repo/astermind/builder/islsim}"
export ORBITServe_VLLM030_PYTHON="${ORBITServe_VLLM030_PYTHON:-$python}"
export ORBITServe_VLLM030_SOURCE="${ORBITServe_VLLM030_SOURCE:-$repo/.venv/source}"
if [[ ${1:-} == --deepseek-v4-example ]]; then
    export ORBITServe_V4_EXPERIMENTAL_MEMORY_PROJECTION="$repo/tests/L2/deepseek-v4-h200-memory.json"
    shift
fi
if [[ -f "$repo/examples/case-study-iii/inputs.tar.gz" ]]; then
    if ! "$python" -B "$repo/scripts/install-case-study-iii.py" --ensure; then
        echo 'Case Study III inputs could not be registered.' >&2
    fi
fi
exec "$python" -B -m viz "$@"
