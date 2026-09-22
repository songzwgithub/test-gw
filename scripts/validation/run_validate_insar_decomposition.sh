#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/mnt/hengshui_gw/hengshui/test-gw}"
cd "$REPO"

SCRIPT="${2:-./validate_insar_decomposition.py}"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: validation script not found: $SCRIPT"
  exit 1
fi

python -m compileall -q src
PYTHONPATH=src python "$SCRIPT" \
  --stack outputs_hengshui/canonical/insar_stack.h5 \
  --deformation-dir outputs_hengshui/deformation \
  --outdir outputs_hengshui/validation/decomposition \
  --recent-start 2022-01-01 \
  --holdout-years 2023 2024 2025 \
  --aggregate-km 5 \
  --projected-crs EPSG:32650 \
  --block-size 256

echo
cat outputs_hengshui/validation/decomposition/decomposition_validation_summary.json
