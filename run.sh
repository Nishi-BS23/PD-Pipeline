#!/usr/bin/env bash
# =============================================================================
# PD Voice Analysis Pipeline — Orchestrator
#
# Usage:
#   bash run.sh
#
# Configuration:
#   Set MODE below to 'full' or 'segment' before running.
#
#   full    : Reads raw FLAC files from mpower_voice_data_flac-*/
#             and cohort labels from final_selected.xlsx.
#             Produces embeddings_full/ for each model.
#
#   segment : Reads pre-segmented FLAC from segments/{PD,HC}/{early,middle,late}/
#             Produces embeddings/ and embeddings_aggregated/ for each model.
#             (Pre-computed embeddings are reused if they already exist on disk.)
#
# Steps executed:
#   segment mode:
#     1. Audio segmentation (raw FLAC -> segments/{PD,HC}/{early,middle,late})
#     2. Wav2Vec2 embedding extraction
#     3. HuBERT embedding extraction
#     4. Comparative analysis (dim reduction, DBSCAN, MLP, evaluation)
#   full mode:
#     1. Audio segmentation (raw FLAC -> segments/{PD,HC}/full)
#     2. Wav2Vec2 embedding extraction
#     3. HuBERT embedding extraction
#     4. Comparative analysis (dim reduction, DBSCAN, MLP, evaluation)
#
# Optional flags (edit below):
#   GPU         : GPU index for embedding extraction (set to "" for CPU)
#   N_TRIALS    : Number of random hparam search trials per model
#   FINAL_EPOCHS: Epochs for final MLP training run
# =============================================================================

# ----------- USER CONFIGURATION (edit here) ----------------------------------
MODE="full"        # 'full' or 'segment'
GPU=0              # GPU index; set to "" to force CPU
N_TRIALS=20
FINAL_EPOCHS=100
MAX_PER_CLASS=0    # 0 means no cap; N means PD=N and HC=N across full flow
SEED=42
# -----------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "============================================================"
echo "  PD Voice Pipeline Orchestrator"
echo "  MODE         = $MODE"
echo "  GPU          = ${GPU:-cpu}"
echo "  N_TRIALS     = $N_TRIALS"
echo "  FINAL_EPOCHS = $FINAL_EPOCHS"
echo "  MAX_PER_CLASS= $MAX_PER_CLASS"
echo "  SEED         = $SEED"
echo "  Project root = $PROJECT_ROOT"
echo "============================================================"

# Build GPU flag string
GPU_FLAG=""
if [ -n "$GPU" ]; then
    GPU_FLAG="--gpu $GPU"
fi

TOTAL_STEPS=4

# ---------------------------------------------------------------------------
# Step 1 (segment mode only) — Audio segmentation
# ---------------------------------------------------------------------------
if [ "$MODE" = "segment" ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "[Step 1/$TOTAL_STEPS] Audio segmentation (mode=$MODE)"
    echo "------------------------------------------------------------"
    python audio_segmentation.py --mode segment --max-per-class "$MAX_PER_CLASS" --seed "$SEED"
elif [ "$MODE" = "full" ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "[Step 1/$TOTAL_STEPS] Audio segmentation (mode=$MODE)"
    echo "------------------------------------------------------------"
    python audio_segmentation.py --mode full --max-per-class "$MAX_PER_CLASS" --seed "$SEED"
fi

# ---------------------------------------------------------------------------
# Step 2 — Wav2Vec2 embedding extraction
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[Step 2/$TOTAL_STEPS] Wav2Vec2 embedding extraction  (mode=$MODE)"
echo "------------------------------------------------------------"
python Wav2Vec2/pipeline.py --mode "$MODE" $GPU_FLAG --max-per-class "$MAX_PER_CLASS" --seed "$SEED"

# ---------------------------------------------------------------------------
# Step 3 — HuBERT embedding extraction
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[Step 3/$TOTAL_STEPS] HuBERT embedding extraction  (mode=$MODE)"
echo "------------------------------------------------------------"
python HuBERT/pipeline.py --mode "$MODE" $GPU_FLAG --max-per-class "$MAX_PER_CLASS" --seed "$SEED"

# ---------------------------------------------------------------------------
# Step 4 — Comparative analysis
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[Step 4/$TOTAL_STEPS] Comparative analysis  (mode=$MODE)"
echo "------------------------------------------------------------"
python comparative_analysis.py \
    --mode "$MODE" \
    --n-trials "$N_TRIALS" \
    --final-epochs "$FINAL_EPOCHS" \
    --max-per-class "$MAX_PER_CLASS" \
    --seed "$SEED"

echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "  Results -> results/comparative_analysis/$MODE/"
echo "============================================================"
