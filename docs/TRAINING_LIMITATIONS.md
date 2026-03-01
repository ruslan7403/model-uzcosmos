# What’s Not Perfect Yet (Training Limitations)

The pipeline has **three trained / data-dependent parts**. Any of them can cause imperfect results.

---

## 1. Traffic sign detector (YOLO) – `traffic_sign_yolo.pt`

**How it’s trained:**  
`train_yolo_detector.py` builds a **synthetic** dataset: random backgrounds (gradients, noise, simple “sky/road”) with **cropped sign images pasted** on them, then fine-tunes YOLOv8n for one class: `traffic_sign`.

**Why it’s not perfect:**
- **Synthetic only** – No real street scenes (perspective, occlusion, real lighting, real backgrounds). So it can miss signs in real photos (false negatives) or fire on non-signs (false positives).
- **Single class** – Only “is it a sign?”; no per–sign-type detection.
- **Few negative examples** – ~10% of training images have no sign. The model can still trigger on background patches that look sign-like.
- **Limited epochs / data** – With few epochs or limited sign variety, recall and precision are not fully tuned.
- **Scale/pose** – Training uses pasted crops at random scales; real scenes have different scales and angles, so some signs are missed or get duplicate/overlapping boxes.

**Improvements:** More epochs, more real (or realistic) images with boxes, more negative samples, optional NMS tuning or confidence threshold.

---

## 2. Embedding model (recognition) – `best_model.pth`

**How it’s trained:**  
Triplet loss on **class directories** (e.g. from Kaggle crops: `data/kaggle_crops/class_name`, …). The model learns to map each crop to a 128‑dim L2‑normalized vector so that same-class signs are close and different-class signs are far.

**Why it’s not perfect:**
- **Training data = clean crops** – Training uses full sign crops from the dataset. At inference you feed **YOLO crops** (often smaller, blurry, partial, different aspect). Distribution shift can lower similarity or cause confusion between similar classes.
- **Limited or short training** – Few epochs (e.g. 1 on Railway) or imbalanced classes leave some classes under-learned; similar-looking classes (e.g. archive_04 vs archive_10) can be confused.
- **Capacity** – ResNet‑18 + 128‑dim embedding is relatively small for many fine-grained sign types.
- **Only known classes** – Anything that wasn’t in the training set has no “natural” class; the model will still output a nearest class, which can be wrong.

**Improvements:** More epochs, balanced data, augmentation that mimics YOLO crops (scale, blur, crop), larger embedding or backbone, or more classes in training.

---

## 3. Gallery – `gallery.json` + `gallery.npz`

**How it’s built:**  
All (or a subset of) images from the **same class dirs** used for embedding training are embedded and stored as prototypes per class. Recognition = nearest prototype (cosine similarity) and a **similarity threshold** (e.g. 0.6).

**Why it’s not perfect:**
- **Coverage** – Only classes present in the data (e.g. from the Kaggle/crops class dirs). Any other sign type is forced to match the nearest class or fall below threshold and become “unknown”.
- **Threshold** – If `similarity_threshold` is too low, wrong classes get chosen; if too high, valid signs are marked unknown.
- **Prototype quality** – Gallery is as good as the embedding model and the images used. Noisy or rare views can bias matches.
- **No explicit rejection model** – Decision is “max similarity vs threshold”; there’s no separate “this is not any known sign” model.

**Improvements:** Tune threshold on a validation set, add more classes/images to the gallery, or add a rejection/confidence mechanism.

---

## Summary

| Component        | Output              | Main limitations                                      |
|-----------------|---------------------|--------------------------------------------------------|
| **YOLO detector** | Boxes (or none)     | Synthetic training, few negatives, no real scenes    |
| **Embedding model** | 128‑d vector        | Trained on clean crops; YOLO crops differ; capacity   |
| **Gallery**     | Class + similarity  | Only trained classes; threshold; no rejection model   |

Fixing “not working perfectly” usually means: **better/more detector data (and epochs), more embedding training (and data augmentation), and tuning the gallery threshold (and coverage).**
