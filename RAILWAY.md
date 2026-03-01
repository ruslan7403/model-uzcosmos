# Run training on Railway

This guide gets [model-uzcosmos](https://github.com/ruslan7403/model-uzcosmos) running on [Railway](https://railway.com): **Kaggle YOLO dataset** → build crops for embedding → **train embedding** → **train YOLO detector** (Kaggle-only pipeline).

---

## 1. Prerequisites

- GitHub account (repo: `ruslan7403/model-uzcosmos`)
- [Railway](https://railway.com) account
- [Kaggle](https://www.kaggle.com) account and API credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`)
- Railway CLI (optional): `npm i -g @railway/cli` and `railway login`

---

## 2. Create a new project and deploy from GitHub

1. Go to [railway.com](https://railway.com) → **Dashboard** → **New Project**.
2. Choose **Deploy from GitHub repo**.
3. Connect GitHub if needed, then select **ruslan7403/model-uzcosmos**.
4. Railway will create a service and start a build (Nixpacks will detect Python). Configure the rest before the first run.

---

## 3. Add a persistent volume (for dataset + output)

You need enough space for the Kaggle dataset, crops, and training outputs (e.g. **50–100 GB** depending on dataset size).

1. Open your service → **Variables** tab (or **Settings**).
2. Go to **Volumes** (or **Storage**).
3. Click **Add Volume** (or **Mount Volume**).
4. Set **Mount Path** to: **`/data`**  
   The script uses `/data/yolo_signs`, `/data/kaggle_crops`, and `/data/output` under this path.

---

## 4. Set environment variables

In the same service, open **Variables** and add:

| Variable         | Value              | Notes                                   |
|------------------|--------------------|----------------------------------------|
| `DATA_ROOT`      | `/data`            | Root for dataset, crops, and output    |
| `OUTPUT_DIR`     | `/data/output`     | Model checkpoints (optional override)  |
| `KAGGLE_USERNAME`| your Kaggle user   | **Required** for dataset download       |
| `KAGGLE_KEY`     | your Kaggle API key| **Required** for dataset download      |
| `EPOCHS`         | `100`              | Embedding training epochs              |
| `YOLO_EPOCHS`    | `50`               | YOLO detector training epochs          |
| `RUN_YOLO_ONLY`  | *(unset)*          | Set to `1` to skip download/embedding and only run YOLO training |

---

## 5. Set the start command

1. Service → **Settings** (or **Deploy**).
2. Set **Start Command** to:

```bash
bash scripts/run_railway.sh
```

---

## 6. Deploy and run

1. **Deploy** the service (or push to `main` if already connected).
2. Railway will build the app and run `bash scripts/run_railway.sh`.
3. The script will:
   - Install dependencies
   - Download Kaggle YOLO traffic sign dataset to `$DATA_ROOT/yolo_signs` (or `$YOLO_DATASET_DIR`)
   - Build class-per-folder crops for embedding to `$DATA_ROOT/kaggle_crops`
   - Train the embedding model → `$OUTPUT_DIR/best_model.pth`, `checkpoint.pt`
   - Train the YOLO detector on real Kaggle data → `$OUTPUT_DIR/traffic_sign_yolo.pt`

**First run:** Download and training can take several hours. Check **Logs** in the Railway dashboard.

---

## 7. Getting the trained model

- **Outputs** are under the volume at **`/data/output`** (or `$OUTPUT_DIR`), e.g.:
  - `best_model.pth`, `checkpoint.pt` (embedding model)
  - `traffic_sign_yolo.pt` (YOLO detector)
  - `gallery.json`, `gallery.npz` (built after full training or via `build_gallery.py`)
- To copy them out: use **Run Command** / one-off shell and e.g. `tar czf /tmp/out.tar.gz /data/output`, then serve or upload as needed.

---

## 8. Resuming and YOLO-only runs

- **Resume:** Keep the same volume and `DATA_ROOT`. If `$OUTPUT_DIR/checkpoint.pt` exists, embedding training resumes from it.
- **YOLO only:** If the Kaggle dataset and crops are already on the volume and you only want to (re)run YOLO training, set **`RUN_YOLO_ONLY=1`**. The script will skip download and embedding and only run `train_yolo_detector.py --real-dataset ...`.

---

## 9. Summary checklist

- [ ] Railway project created from **ruslan7403/model-uzcosmos**
- [ ] Volume mounted at **`/data`**
- [ ] Variables: **`DATA_ROOT=/data`**, **`KAGGLE_USERNAME`**, **`KAGGLE_KEY`**
- [ ] Start command: **`bash scripts/run_railway.sh`**
- [ ] Deploy and watch **Logs**
