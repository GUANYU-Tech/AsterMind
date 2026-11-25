#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
venv="$repo/.venv"
python="$venv/bin/python"

scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT

download() {
    if ! command -v curl >/dev/null; then
        echo 'curl is required to download missing user-space tools.' >&2
        return 1
    fi
    curl --fail --location --silent --show-error --retry 2 "$@"
}

if [[ ! -x "$python" ]]; then
    if [[ -f "$venv/pyvenv.cfg" ]]; then
        echo "The existing $venv has no working Python. Move it aside before running this script." >&2
        exit 1
    fi
    if command -v python3.12 >/dev/null && python3.12 -m ensurepip --version >/dev/null 2>&1; then
        python3.12 -m venv "$venv"
    else
        uv=$(command -v uv || true)
        if [[ -z $uv ]]; then
            download https://astral.sh/uv/0.11.14/install.sh -o "$scratch/install-uv.sh"
            UV_UNMANAGED_INSTALL="$scratch/bin" sh "$scratch/install-uv.sh"
            uv="$scratch/bin/uv"
        fi
        "$uv" venv --python 3.12 --seed "$venv"
    fi
fi

if ! "$python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    echo "The existing $venv must use Python 3.12." >&2
    exit 1
fi
if ! "$python" -m pip --version >/dev/null 2>&1; then
    "$python" -m ensurepip --upgrade
fi

export PATH="$venv/bin:$PATH"
if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
    node_archive=node-v22.23.5-linux-x64.tar.xz
    node_release=https://nodejs.org/dist/v22.23.5
    node_sha=$(download "$node_release/SHASUMS256.txt" | awk -v name="$node_archive" '$2 == name { print $1 }')
    if [[ ! $node_sha =~ ^[0-9a-f]{64}$ ]]; then
        echo "No SHA-256 checksum found for $node_archive." >&2
        exit 1
    fi
    download "$node_release/$node_archive" -o "$scratch/$node_archive"
    printf '%s  %s\n' "$node_sha" "$scratch/$node_archive" | sha256sum --check --status
    mkdir -p "$venv/node"
    tar -xJf "$scratch/$node_archive" --strip-components=1 -C "$venv/node"
    ln -sfn ../node/bin/node "$venv/bin/node"
    ln -sfn ../node/bin/npm "$venv/bin/npm"
fi

echo 'The AsterMind environment is ready. Activate .venv before installing requirements.txt.'
