# FaceID-like Traffic Sign Recognition System

A similarity-based traffic sign recognition system that uses learned embeddings to identify signs. Instead of a fixed classifier, this system works like FaceID: it converts images into compact embeddings and compares them by similarity.

## Key Features

- **Object detection** — YOLOv8 detects traffic signs in full scene images with bounding boxes
- **Embedding-based recognition** — Each detected sign is identified by similarity in a learned embedding space
- **Visual output** — Annotated images with bounding boxes, class labels, and similarity scores
- **Open-set recognition** — Unknown signs are detected when no gallery match exceeds the similarity threshold
- **Incremental learning** — New sign classes can be added with just 1-3 reference images, without retraining the model
- **Gallery system** — Reference embeddings (prototypes) are stored and managed like face templates in FaceID

## Architecture

```
                          Detection + Recognition Pipeline
                          ================================

Scene Image ──→ YOLOv8 Detector ──→ Bounding Boxes
                                          │
                         ┌────────────────┘
                         ↓
                    Crop each sign
                         ↓
              ResNet-18 Backbone → Projection Head → 128-d Embedding
                                                          ↓
                                                 Cosine Similarity
                                                          ↓
                                              Gallery Matching → Prediction
                                                          ↓
                                          Annotated image with boxes + scores
```

The model is trained with **triplet loss**, which encourages:
- Images of the **same** sign to have similar embeddings
- Images of **different** signs to have dissimilar embeddings

## Project Structure

```
traffic_sign_recognition/
├── __init__.py          # Package exports
├── model.py             # EmbeddingNet (ResNet-18 + projection head) and TripletLoss
├── gallery.py           # SignGallery for storing/matching reference embeddings
├── dataset.py           # TripletTrafficSignDataset and transforms
├── trainer.py           # Training loop, gallery building, evaluation
├── recognizer.py        # High-level TrafficSignRecognizer API
├── detector.py          # YOLOv8-based traffic sign detector
├── visualize.py         # Bounding box drawing, labels, and score panels
├── pipeline.py          # Full detect → crop → embed → match pipeline
scripts/
├── train.py             # CLI training script
├── recognize.py         # CLI recognition script (single cropped sign)
├── detect_and_recognize.py  # CLI detection + recognition on scene images
├── enroll.py            # CLI enrollment script (incremental learning)
├── demo.py              # Full demo with synthetic data
tests/
├── test_system.py       # Comprehensive test suite
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run Demo (no data needed)

```bash
python scripts/demo.py
```

This generates synthetic traffic signs, trains the model, builds a gallery, tests recognition, demonstrates unknown detection, and shows incremental class enrollment.

### Train on Your Data

Organize images by class:
```
data/train/
    stop_sign/
        img1.jpg
        img2.jpg
    speed_limit_50/
        img1.jpg
        ...
```

```bash
python scripts/train.py --data-dir data/train --output-dir output --epochs 30
```

### Detect and Recognize Signs in a Scene Image

```bash
python scripts/detect_and_recognize.py \
    --image street_photo.jpg \
    --embedding-model output/best_model.pth \
    --gallery output/gallery \
    --output result.jpg
```

This will:
1. Detect traffic signs in the image using YOLOv8 (draws bounding boxes)
2. Recognize each detected sign via embedding similarity (labels + scores)
3. Save an annotated image with rectangles, class names, and similarity scores

For a custom YOLO model trained specifically on traffic signs:
```bash
python scripts/detect_and_recognize.py \
    --image street_photo.jpg \
    --yolo-model my_traffic_sign_yolo.pt \
    --embedding-model output/best_model.pth \
    --gallery output/gallery \
    --detect-all \
    --output result.jpg
```

### Recognize a Single Cropped Sign

```bash
python scripts/recognize.py \
    --model output/best_model.pth \
    --gallery output/gallery \
    --image test_image.jpg
```

### Add a New Sign Class (Incremental Learning)

```bash
python scripts/enroll.py \
    --model output/best_model.pth \
    --gallery output/gallery \
    --class-name "new_country_sign" \
    --images template.png sample1.jpg sample2.jpg
```

No retraining required. The new class is immediately available for recognition.

### Run Tests

```bash
pytest tests/ -v
```

## How It Works

### Training Phase
1. Images are organized by traffic sign class
2. Triplets (anchor, positive, negative) are sampled
3. The model learns to minimize distance between same-class pairs and maximize distance between different-class pairs
4. After training, all training images are embedded into a gallery

### Recognition Phase
1. A query image is embedded by the model
2. The embedding is compared (cosine similarity) against all gallery prototypes
3. The class with the highest similarity is returned, along with a confidence score
4. If no class exceeds the threshold, the sign is reported as **unknown**

### Incremental Learning
1. Provide 1+ reference images of the new sign class
2. Each image is embedded using the trained model
3. Embeddings are added to the gallery as new prototypes
4. The new class is immediately recognizable — no retraining needed

This is the same workflow as enrolling a new face in FaceID.

## Data Source

This system is designed to work with traffic sign datasets such as:
- [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign)
- German Traffic Sign Recognition Benchmark (GTSRB)
- Belgian Traffic Sign Dataset

## API Usage

### Full Detection + Recognition Pipeline

```python
from traffic_sign_recognition.pipeline import DetectionRecognitionPipeline
from PIL import Image

# Load the full pipeline (YOLO detector + embedding recognizer + gallery)
pipeline = DetectionRecognitionPipeline.load(
    yolo_model_path="yolov8n.pt",
    embedding_model_path="output/best_model.pth",
    gallery_path="output/gallery",
)

# Process a scene image and get annotated output
image = Image.open("street_photo.jpg")
annotated_img, detections = pipeline.process_and_visualize(
    image,
    output_path="result.jpg",  # saves annotated image with bboxes + scores
    with_panel=True,           # side panel with detailed similarity scores
)

# Inspect each detection
for det in detections:
    print(f"Sign: {det.recognized_class}, Score: {det.similarity_score:.2f}, "
          f"Box: {det.bbox}")
```

### Recognition Only (Pre-Cropped Images)

```python
from traffic_sign_recognition import TrafficSignRecognizer

recognizer = TrafficSignRecognizer.load(
    model_path="output/best_model.pth",
    gallery_path="output/gallery",
)

prediction, confidence, scores = recognizer.recognize_file("cropped_sign.jpg")
if prediction is None:
    print("Unknown sign")
else:
    print(f"Detected: {prediction} (confidence: {confidence:.2f})")

# Add a new class incrementally (no retraining)
recognizer.enroll_class("new_sign", ["ref1.jpg", "ref2.jpg"])
recognizer.save("output/model.pth", "output/gallery")
```
