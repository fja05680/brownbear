#!/usr/bin/env bash
# Stop stray Jupyter kernels and remove stale connection files.
set -euo pipefail

count="$(pgrep -fc ipykernel_launcher || true)"
echo "ipykernel processes before cleanup: ${count}"

if [[ "${count}" -gt 0 ]]; then
    pkill -f ipykernel_launcher || true
    sleep 1
fi

remaining="$(pgrep -fc ipykernel_launcher || true)"
echo "ipykernel processes after cleanup: ${remaining}"

runtime_dir="${HOME}/.local/share/jupyter/runtime"
if [[ -d "${runtime_dir}" ]]; then
    stale="$(find "${runtime_dir}" -maxdepth 1 -name 'kernel-*.json' | wc -l)"
    echo "stale kernel connection files: ${stale}"
    if [[ "${remaining}" -eq 0 && "${stale}" -gt 0 ]]; then
        rm -f "${runtime_dir}"/kernel-*.json
        echo "removed stale kernel connection files"
    elif [[ "${stale}" -gt 0 ]]; then
        echo "skipped removing connection files while kernels are still running"
    fi
fi

cache_dir="${TMPDIR:-/tmp}/brownbear-yfinance"
if [[ -d "${cache_dir}" ]]; then
    rm -rf "${cache_dir}"
    echo "cleared per-process yfinance cache: ${cache_dir}"
fi
