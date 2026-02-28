#!/usr/bin/env bash
# Railway entrypoint: install deps, download dataset, then train.
# Set DATA_ROOT (e.g. /data) when using a Railway volume; dataset and output go under it.

set -e
DATA_ROOT="${DATA_ROOT:-.}"
DATA_DIR="${DATA_DIR:-$DATA_ROOT/mapillary}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/output}"
EPOCHS="${EPOCHS:-100}"
YOLO_EPOCHS="${YOLO_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRIPLETS="${TRIPLETS_PER_EPOCH:-10000}"

echo "=== Install dependencies ==="
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -e .

echo "=== Download dataset to $DATA_DIR ==="
mkdir -p "$(dirname "$DATA_DIR")"
mkdir -p "$OUTPUT_DIR"
python scripts/download_dataset.py --output-dir "$DATA_DIR"

echo "=== Train model (epochs=$EPOCHS, batch_size=$BATCH_SIZE) ==="
CHECKPOINT_ARG=""
if [ -f "$OUTPUT_DIR/checkpoint.pt" ]; then
  CHECKPOINT_ARG="--checkpoint $OUTPUT_DIR/checkpoint.pt"
  echo "Resuming from $OUTPUT_DIR/checkpoint.pt"
fi

python scripts/train.py \
  --data-dir "$DATA_DIR" \
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

echo "=== Train custom YOLO detector (traffic_sign only) ==="
python scripts/train_yolo_detector.py \
  --sign-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$YOLO_EPOCHS" \
  --num-train 2000 \
  --num-val 400 \
  --batch-size 16 \
  --img-size 640

echo "=== Done. Embedding model: $OUTPUT_DIR/best_model.pth | YOLO: $OUTPUT_DIR/traffic_sign_yolo.pt ==="
