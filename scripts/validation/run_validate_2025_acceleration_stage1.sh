#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/mnt/hengshui_gw/hengshui/test-gw}"
cd "$REPO"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

rm -rf outputs_hengshui/validation/acceleration_2025_stage1

/usr/bin/time -v env PYTHONPATH=src \
python ./validate_2025_acceleration_stage1.py \
  --stack outputs_hengshui/canonical/insar_stack.h5 \
  --outdir outputs_hengshui/validation/acceleration_2025_stage1 \
  --block-size 512 \
  --max-pair-day-diff 18 \
  --same-season-start-month 1 \
  --same-season-end-month 8 \
  --same-season-start-year 2017 \
  --same-season-end-year 2024 \
  --window 2020-01-01:2022-12-31 \
  --window 2021-01-01:2023-12-31 \
  --window 2022-01-01:2024-12-31 \
  --window 2023-01-01:2025-08-30

echo
echo "===== RESULT FILES ====="
find outputs_hengshui/validation/acceleration_2025_stage1 \
  -maxdepth 2 -type f -printf "%p\n" | sort
