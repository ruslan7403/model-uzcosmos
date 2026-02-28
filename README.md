# FaceID-like Traffic Sign Recognition System

A similarity-based traffic sign recognition system that uses learned embeddings to identify signs. Instead of a fixed classifier, this system works like FaceID: it converts images into compact embeddings and compares them by similarity.

## Key Features

- **Embedding-based recognition** — Signs are identified by similarity in a learned embedding space, not fixed class logits
- **Open-set recognition** — Unknown signs are detected when no gallery match exceeds the similarity threshold
- **Incremental learning** — New sign classes can be added with just 1-3 reference images, without retraining the model
- **Gallery system** — Reference embeddings (prototypes) are stored and managed like face templates in FaceID

## Architecture

```
Input Image → ResNet-18 Backbone → Projection Head → L2-Normalized Embedding (128-d)
                                                              ↓
                                                    Cosine Similarity
                                                              ↓
                                                  Gallery Matching → Prediction
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
scripts/
├── train.py             # CLI training script
├── recognize.py         # CLI recognition script
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

### Recognize a Sign

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

```python
from traffic_sign_recognition import EmbeddingNet, SignGallery, TrafficSignRecognizer

# Load trained system
recognizer = TrafficSignRecognizer.load(
    model_path="output/best_model.pth",
    gallery_path="output/gallery",
)

# Recognize a sign
prediction, confidence, scores = recognizer.recognize_file("sign.jpg")
if prediction is None:
    print("Unknown sign")
else:
    print(f"Detected: {prediction} (confidence: {confidence:.2f})")

# Add a new class incrementally
recognizer.enroll_class("new_sign", ["ref1.jpg", "ref2.jpg"])
recognizer.save("output/model.pth", "output/gallery")
```
