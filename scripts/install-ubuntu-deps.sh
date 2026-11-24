#!/usr/bin/env bash
set -euo pipefail

missing=()
for program in git python3.12 "${CC:-gcc}" "${CC:-cc}" "${CXX:-g++}" node npm; do
    if ! command -v "$program" >/dev/null; then
        missing+=("$program")
    fi
done
if (( ${#missing[@]} )); then
    printf 'Missing required command: %s\n' "${missing[@]}" >&2
    exit 1
fi
if ! python3.12 -c 'import ensurepip' >/dev/null 2>&1; then
    echo 'Python 3.12 must include venv and ensurepip. On Ubuntu, these are provided by python3.12-venv.' >&2
    exit 1
fi
if ! python3.12 -c 'import ctypes; ctypes.CDLL("libnuma.so.1")' >/dev/null 2>&1; then
    echo 'The vLLM CPU wheel requires libnuma.so.1. On Ubuntu, it is provided by libnuma1.' >&2
    exit 1
fi
echo 'System prerequisites are available.'
