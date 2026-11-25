#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
repo=$(pwd -P)
python="$repo/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo 'Create .venv and install requirements.txt before building.' >&2
    exit 1
fi

if ! "$python" -c 'import ctypes; ctypes.CDLL("libnuma.so.1")' >/dev/null 2>&1; then
    echo 'The vLLM CPU wheel requires libnuma.so.1. On Ubuntu, it is provided by libnuma1.' >&2
    exit 1
fi

"$python" -m astermind.common.build_native
VLLM030_ENV="$repo/.venv" bash scripts/install-vllm-030.sh --source-only

export PATH="$repo/.venv/bin:$PATH"
npm install --prefix viz --no-package-lock --no-audit --no-fund
three_dir=viz/node_modules/three
mkdir -p viz/cell/vendor
cp "$three_dir"/build/three.module.js "$three_dir"/build/three.core.js viz/assets/
cp "$three_dir"/build/three.module.min.js "$three_dir"/build/three.core.min.js viz/cell/vendor/
cp "$three_dir"/examples/jsm/controls/OrbitControls.js viz/assets/
cp "$three_dir"/examples/jsm/controls/OrbitControls.js viz/cell/vendor/
