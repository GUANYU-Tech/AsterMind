#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
environment="${VLLM030_ENV:-$repo/.venv-vllm030}"
python="$environment/bin/python"
source_dir="$environment/source"
tag=v0.30.0
commit=ced6857afa0ea7b2e3f0846a62e1394e90f15607
requirements="$repo/astermind/engine/compute/scheduler_runtime/requirements-vllm-0.30.txt"

if (( $# > 1 )) || [[ $# -eq 1 && "$1" != --source-only ]]; then
    echo 'Usage: bash scripts/install-vllm-030.sh [--source-only]' >&2
    exit 2
fi

if [[ $(uname -sm) != 'Linux x86_64' ]]; then
    echo 'The vLLM 0.30 CPU installer requires x86-64 Linux.' >&2
    exit 1
fi

if (( $# == 0 )); then
    if [[ ! -x "$python" ]] || ! "$python" -m pip --version >/dev/null 2>&1; then
        python3.12 -m venv "$environment"
    fi
    "$python" -m pip install -r "$requirements"
elif [[ ! -x "$python" ]]; then
    printf 'Create %s and install the vLLM 0.30 requirements first.\n' "$environment" >&2
    exit 1
fi

# The scheduler imports from the checkout. The CPU wheel provides its compiled extensions.
wheel_package=$("$python" -c '
from importlib.metadata import distribution

wheel = distribution("vllm")
if wheel.version != "0.30.0+cpu":
    raise SystemExit("Expected the vLLM 0.30.0 CPU wheel.")
print(wheel.locate_file("vllm"))
')

if [[ ! -e "$source_dir/.git" ]]; then
    git -c advice.detachedHead=false clone --depth 1 --single-branch --branch "$tag" \
        https://github.com/vllm-project/vllm.git "$source_dir"
fi
if [[ $(git -C "$source_dir" rev-parse HEAD) != "$commit" || \
      $(git -C "$source_dir" tag --points-at HEAD --list "$tag") != "$tag" || \
      -n $(git -C "$source_dir" status --porcelain --untracked-files=all) ]]; then
    printf 'Expected a clean vLLM %s checkout at %s.\n' "$tag" "$source_dir" >&2
    exit 1
fi

shopt -s nullglob
libraries=("$wheel_package"/*.so)
if (( ${#libraries[@]} == 0 )); then
    printf 'No compiled extensions found in %s.\n' "$wheel_package" >&2
    exit 1
fi
ln -sfn -- "${libraries[@]}" "$source_dir/vllm/"
printf 'vLLM 0.30 CPU scheduler ready in %s.\n' "$environment"
