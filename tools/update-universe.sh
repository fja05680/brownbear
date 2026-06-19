#!/usr/bin/env bash
# Run UPDATE steps 1-4 (restartable). Completed steps are recorded in
# tools/.update-universe.state and skipped on the next run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

STATE_FILE="${ROOT}/tools/.update-universe.state"
FUNDAMENTALS_STEP="step3:fundamentals"

DELETE_SYMBOL_CACHE=false
RESET_FUNDAMENTALS_CACHE=false
RESET_PROGRESS=false

usage() {
    cat <<'EOF'
Usage: tools/update-universe.sh [OPTIONS]

Run UPDATE steps 1-4. If a step fails, fix the issue and re-run with no
options; completed steps are skipped (notebooks use refresh_timeseries=False
and reuse existing cache).

Typical workflow:
  First run:   tools/update-universe.sh --full
  After error: tools/update-universe.sh

Options:
  --full                      First-time run: --delete-symbol-cache,
                              --reset-fundamentals-cache, and --reset-progress
  --delete-symbol-cache       Step 0: delete symbol-cache/ and reset progress
  --reset-fundamentals-cache  Delete fundamentals_cache.json before step 3
  --reset-progress            Re-run all steps, ignoring the checkpoint file
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            DELETE_SYMBOL_CACHE=true
            RESET_FUNDAMENTALS_CACHE=true
            RESET_PROGRESS=true
            ;;
        --delete-symbol-cache)
            DELETE_SYMBOL_CACHE=true
            ;;
        --reset-fundamentals-cache)
            RESET_FUNDAMENTALS_CACHE=true
            ;;
        --reset-progress)
            RESET_PROGRESS=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [[ -f "${ROOT}/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/venv/bin/activate"
fi

if ! command -v jupyter >/dev/null 2>&1; then
    echo "jupyter not found; activate your venv and install requirements first" >&2
    exit 1
fi

if [[ "${DELETE_SYMBOL_CACHE}" == true ]]; then
    echo "Step 0: deleting symbol-cache/"
    rm -rf "${ROOT}/symbol-cache"
    RESET_PROGRESS=true
fi

if [[ "${RESET_PROGRESS}" == true && -f "${STATE_FILE}" ]]; then
    rm -f "${STATE_FILE}"
    echo "Cleared checkpoint: ${STATE_FILE}"
fi

is_done() {
    [[ -f "${STATE_FILE}" ]] && grep -Fxq "$1" "${STATE_FILE}"
}

mark_done() {
    echo "$1" >> "${STATE_FILE}"
}

cleanup_kernels() {
    "${ROOT}/tools/cleanup-kernels.sh"
}

run_notebook() {
    local notebook="$1"
    if is_done "${notebook}"; then
        echo
        echo "=== Skipping (already done): ${notebook} ==="
        return 0
    fi

    echo
    echo "=== Running ${notebook} ==="
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=-1 \
        "${notebook}"
    mark_done "${notebook}"
    cleanup_kernels
}

run_fundamentals() {
    if is_done "${FUNDAMENTALS_STEP}"; then
        echo
        echo "=== Skipping (already done): fundamentals ==="
        return 0
    fi

    echo
    echo "=== Running step 3: get_symbol_fundamentals ==="
    python3 - "${RESET_FUNDAMENTALS_CACHE}" <<'PY'
import sys

import brownbear as bb

reset_cache = sys.argv[1] == "true"
if reset_cache:
    deleted = bb.reset_fundamentals_cache()
    if deleted:
        print("Deleted fundamentals cache")
    else:
        print("No fundamentals cache to delete")

df = bb.get_symbol_fundamentals(throttle_limit=100, wait_time=30, reset_cache=False)
out = bb.ROOT / "tools" / "symbol-cache" / "fundamentals.csv"
df.to_csv(out, encoding="utf-8")
print(f"Wrote {out}")
PY
    mark_done "${FUNDAMENTALS_STEP}"
    cleanup_kernels
}

echo "Step 1: asset-class-galaxy/asset-classes.ipynb"
run_notebook "universe/asset-class-galaxy/asset-classes.ipynb"

echo
echo "Step 2: remaining universe galaxies"
for galaxy_dir in "${ROOT}"/universe/*/; do
    galaxy="$(basename "${galaxy_dir}")"
    if [[ "${galaxy}" == "asset-class-galaxy" ]]; then
        continue
    fi

    for notebook in "${galaxy_dir}"*.ipynb; do
        [[ -f "${notebook}" ]] || continue
        base="$(basename "${notebook}")"
        if [[ "${base}" == "investment-options.ipynb" ]]; then
            continue
        fi
        run_notebook "universe/${galaxy}/${base}"
    done

    investment_options="universe/${galaxy}/investment-options.ipynb"
    if [[ -f "${investment_options}" ]]; then
        if [[ "${galaxy}" == "extra-manual-galaxy" ]]; then
            echo "NOTE: ${galaxy} may need manual edits to investment-options.csv first"
        fi
        run_notebook "${investment_options}"
    fi
done

echo
echo "Step 3: symbol fundamentals"
if [[ "${RESET_FUNDAMENTALS_CACHE}" == true && -f "${STATE_FILE}" ]]; then
    grep -Fxv "${FUNDAMENTALS_STEP}" "${STATE_FILE}" > "${STATE_FILE}.tmp" || true
    mv "${STATE_FILE}.tmp" "${STATE_FILE}"
fi
run_fundamentals

echo
echo "Step 4: portfolios"
for portfolio_dir in "${ROOT}"/portfolios/*/; do
    portfolio="$(basename "${portfolio_dir}")"

    investment_options="portfolios/${portfolio}/investment-options.ipynb"
    if [[ -f "${investment_options}" ]]; then
        run_notebook "${investment_options}"
    fi

    portfolio_nb="portfolios/${portfolio}/portfolio.ipynb"
    if [[ -f "${portfolio_nb}" ]]; then
        run_notebook "${portfolio_nb}"
    fi
done

echo
echo "Done."
