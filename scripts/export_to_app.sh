#!/bin/bash
# scripts/export_to_app.sh
#
# Copies osdci-pipeline outputs to osdci-app after
# retraining. Run from the osdci-pipeline root directory.
#
# Usage:
#   cd osdci-pipeline
#   bash scripts/export_to_app.sh
#
# Assumes osdci-app is at ../osdci-app relative to
# osdci-pipeline. Edit APP_DIR below if different.

set -e

APP_DIR="../osdci-app"
BACKEND_DATA="$APP_DIR/backend/data"
FRONTEND_DATA="$APP_DIR/frontend/public/data"

echo "OSDCI — Exporting pipeline outputs to osdci-app"
echo "================================================"
echo "App dir: $APP_DIR"
echo ""

# Verify osdci-app exists
if [ ! -d "$APP_DIR" ]; then
  echo "ERROR: osdci-app not found at $APP_DIR"
  echo "Edit APP_DIR in this script to match your setup."
  exit 1
fi

# Verify all required pipeline outputs exist
REQUIRED=(
  "models/top_candidates.geojson"
  "models/basemap.csv"
  "models/cell_explanations.json"
  "data/processed/scored_cells_app.parquet"
  "models/baseline_results.json"
  "models/feature_importances.csv"
)

echo "Checking pipeline outputs..."
for f in "${REQUIRED[@]}"; do
  if [ ! -f "$f" ]; then
    echo ""
    echo "ERROR: Missing pipeline output: $f"
    echo ""
    echo "Run the full pipeline first:"
    echo "  python scripts/train_baseline.py"
    echo "  python scripts/compute_final_scores.py"
    exit 1
  fi
  echo "  ✓ $f"
done

echo ""
echo "All outputs present. Copying to osdci-app..."
echo ""

# Create destination directories if they don't exist
mkdir -p "$BACKEND_DATA"
mkdir -p "$FRONTEND_DATA"

# Backend data files
cp models/top_candidates.geojson \
   "$BACKEND_DATA/top_candidates.geojson"
echo "✓ backend/data/top_candidates.geojson"

cp models/basemap.csv \
   "$BACKEND_DATA/basemap.csv"
echo "✓ backend/data/basemap.csv"

cp models/cell_explanations.json \
   "$BACKEND_DATA/cell_explanations.json"
echo "✓ backend/data/cell_explanations.json"

cp data/processed/scored_cells_app.parquet \
   "$BACKEND_DATA/scored_cells.parquet"
echo "✓ backend/data/scored_cells.parquet (app-filtered)"

# Frontend static data files
cp models/top_candidates.geojson \
   "$FRONTEND_DATA/top_candidates.geojson"
echo "✓ frontend/public/data/top_candidates.geojson"

echo ""
echo "================================================"
echo "Export complete."
echo ""
echo "Next steps:"
echo ""
echo "  cd $APP_DIR"
echo "  git add backend/data/ frontend/public/data/"
echo "  git commit -m 'refresh: update data files from pipeline'"
echo "  git push"
echo ""
echo "Railway will redeploy osdci-app automatically on push."
