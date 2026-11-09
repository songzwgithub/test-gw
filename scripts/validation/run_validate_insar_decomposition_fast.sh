#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/mnt/hengshui_gw/hengshui/test-gw}"
cd "$REPO"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

rm -rf outputs_hengshui/validation/decomposition
mkdir -p outputs_hengshui/validation/decomposition

/usr/bin/time -v env PYTHONPATH=src python ./validate_insar_decomposition_fast.py \
  --stack outputs_hengshui/canonical/insar_stack.h5 \
  --deformation-dir outputs_hengshui/deformation \
  --outdir outputs_hengshui/validation/decomposition \
  --recent-start 2022-01-01 \
  --holdout-years 2023 2024 2025 \
  --aggregate-km 5 \
  --projected-crs EPSG:32650 \
  --block-size 512
