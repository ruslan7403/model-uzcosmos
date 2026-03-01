#!/usr/bin/env bash
# Railway entrypoint: Kaggle YOLO only (no Mapillary).
# 1) Download Kaggle YOLO traffic sign dataset
# 2) Build class-per-folder crops from it for embedding training
# 3) Train embedding model on those crops
# 4) Train YOLO detector on the same Kaggle dataset (real images + bboxes)
#
# Set DATA_ROOT (e.g. /data) when using a Railway volume.
# Required env: KAGGLE_USERNAME, KAGGLE_KEY

set -e
DATA_ROOT="${DATA_ROOT:-.}"
YOLO_DATASET_DIR="${YOLO_DATASET_DIR:-$DATA_ROOT/yolo_signs}"
CROPS_DIR="${CROPS_DIR:-$DATA_ROOT/kaggle_crops}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/output}"
EPOCHS="${EPOCHS:-100}"
YOLO_EPOCHS="${YOLO_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRIPLETS="${TRIPLETS_PER_EPOCH:-10000}"
YOLO_BATCH_SIZE="${YOLO_BATCH_SIZE:-16}"

echo "=== Install dependencies ==="
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -e .
pip uninstall -y opencv-python 2>/dev/null || true
pip install -q opencv-python-headless

# Optional: skip everything except YOLO detector training (dataset + crops must already exist)
if [ -n "${RUN_YOLO_ONLY}" ] && [ "${RUN_YOLO_ONLY}" != "0" ]; then
  echo "=== RUN_YOLO_ONLY: skip download and embedding training ==="
  mkdir -p "$OUTPUT_DIR"
  if [ ! -d "$YOLO_DATASET_DIR/train/images" ] || [ -z "$(ls -A "$YOLO_DATASET_DIR/train/images" 2>/dev/null)" ]; then
    echo "ERROR: $YOLO_DATASET_DIR has no train/images. Run full pipeline first."
    exit 1
  fi
  echo "=== Train YOLO detector on real Kaggle data ==="
  python scripts/train_yolo_detector.py \
    --real-dataset "$YOLO_DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "${YOLO_EPOCHS}" \
    --batch-size "$YOLO_BATCH_SIZE"
  echo "=== Done. YOLO: $OUTPUT_DIR/traffic_sign_yolo.pt ==="
  exit 0
fi

echo "=== Download Kaggle YOLO traffic sign dataset to $YOLO_DATASET_DIR ==="
mkdir -p "$YOLO_DATASET_DIR"
mkdir -p "$OUTPUT_DIR"
python scripts/download_kaggle_yolo_dataset.py --output-dir "$YOLO_DATASET_DIR"

echo "=== Build class-per-folder crops from Kaggle for embedding training ==="
python scripts/build_kaggle_crops_for_embedding.py \
  --yolo-dir "$YOLO_DATASET_DIR" \
  --output-dir "$CROPS_DIR" \
  --splits train val

echo "=== Train embedding model on Kaggle crops (epochs=$EPOCHS) ==="
CHECKPOINT_ARG=""
if [ -f "$OUTPUT_DIR/checkpoint.pt" ]; then
  CHECKPOINT_ARG="--checkpoint $OUTPUT_DIR/checkpoint.pt"
  echo "Resuming from $OUTPUT_DIR/checkpoint.pt"
fi
python scripts/train.py \
  --data-dir "$CROPS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --triplets-per-epoch "$TRIPLETS" \
  --embedding-dim 128 \
  --learning-rate 1e-4 \
  --margin 0.3 \
  --image-size 224 \
  --gallery-threshold 0.6 \
  --checkpoint-interval 1 \
  $CHECKPOINT_ARG

echo "=== Train YOLO detector on real Kaggle data ==="
python scripts/train_yolo_detector.py \
  --real-dataset "$YOLO_DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$YOLO_EPOCHS" \
  --batch-size "$YOLO_BATCH_SIZE"

echo "=== Done. Embedding: $OUTPUT_DIR/best_model.pth | YOLO: $OUTPUT_DIR/traffic_sign_yolo.pt ==="
