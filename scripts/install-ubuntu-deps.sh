#!/usr/bin/env bash
set -euo pipefail

failed=false
platform=$(uname -sm)
if [[ $platform != 'Linux x86_64' ]]; then
    printf 'The vLLM CPU wheel requires x86-64 Linux (found: %s).\n' "$platform" >&2
    failed=true
fi
glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
if [[ ! $glibc =~ ^glibc\ ([0-9]+)\.([0-9]+)$ ]] ||
    (( BASH_REMATCH[1] < 2 || (BASH_REMATCH[1] == 2 && BASH_REMATCH[2] < 39) )); then
    printf 'The vLLM CPU wheel requires glibc 2.39 or newer (found: %s).\n' "${glibc:-unavailable}" >&2
    failed=true
fi

missing=()
for program in git "${CC:-gcc}" "${CC:-cc}" "${CXX:-g++}" python3.12 node npm; do
    if ! command -v "$program" >/dev/null; then
        missing+=("$program")
    fi
done
if (( ${#missing[@]} )); then
    printf 'Missing required command: %s\n' "${missing[@]}" >&2
    failed=true
fi
if command -v python3.12 >/dev/null && ! python3.12 -m ensurepip --version >/dev/null 2>&1; then
    echo 'Python 3.12 must include venv and ensurepip.' >&2
    failed=true
fi
if [[ $failed == true ]]; then
    exit 1
fi
echo 'System checks passed.'
