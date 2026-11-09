#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/mnt/hengshui_gw/hengshui/test-gw}"
cd "$REPO"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

rm -rf outputs_hengshui/validation/acceleration_2025_stage2

/usr/bin/time -v env PYTHONPATH=src \
python ./diagnose_2025_insar_real_vs_artifact.py \
  --stack outputs_hengshui/canonical/insar_stack.h5 \
  --deformation-dir outputs_hengshui/deformation \
  --stage1-dir outputs_hengshui/validation/acceleration_2025_stage1 \
  --groundwater outputs_hengshui/groundwater/groundwater_field.h5 \
  --groundwater-distance outputs_hengshui/groundwater/groundwater_nearest_well_distance_km.tif \
  --outdir outputs_hengshui/validation/acceleration_2025_stage2 \
  --sample-stride 24 \
  --block-size 512 \
  --max-pair-day-diff 18 \
  --support-distance-km 15 \
  --aggregate-km 5 \
  --projected-crs EPSG:32650

echo
echo "===== FINAL SUMMARY ====="
cat outputs_hengshui/validation/acceleration_2025_stage2/stage2_artifact_vs_real_summary.json
