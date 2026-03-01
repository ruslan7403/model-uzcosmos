# Run download + training on Railway

This guide gets [model-uzcosmos](https://github.com/ruslan7403/model-uzcosmos) running on [Railway](https://railway.com): **download dataset** (~110 GB) then **train** the embedding model. No GitHub Actions; everything runs on Railway.

---

## 1. Prerequisites

- GitHub account (repo: `ruslan7403/model-uzcosmos`)
- [Railway](https://railway.com) account (Pro recommended for ~1 TB storage)
- Railway CLI (optional): `npm i -g @railway/cli` and `railway login`

---

## 2. Create a new project and deploy from GitHub

1. Go to [railway.com](https://railway.com) → **Dashboard** → **New Project**.
2. Choose **Deploy from GitHub repo**.
3. Connect GitHub if needed, then select **ruslan7403/model-uzcosmos**.
4. Railway will create a service and start a build (Nixpacks will detect Python). You’ll configure the rest before the first run.

---

## 3. Add a persistent volume (for dataset + output)

You need **at least ~150 GB** so the downloaded dataset and training outputs persist.

1. Open your service → **Variables** tab (or **Settings**).
2. Go to **Volumes** (or **Storage**).
3. Click **Add Volume** (or **Mount Volume**).
4. Set size to **200 GB** or more (e.g. 250 GB to be safe).
5. Set **Mount Path** to: **`/data`**  
   (The script will put `mapillary` and `output` under `/data`.)

---

## 4. Set environment variables

In the same service, open **Variables** and add:

| Variable       | Value           | Notes                          |
|----------------|-----------------|---------------------------------|
| `DATA_ROOT`    | `/data`         | Root for dataset and output    |
| `DATA_DIR`     | `/data/mapillary` | Dataset directory (optional) |
| `OUTPUT_DIR`   | `/data/output`  | Model checkpoints and artifacts (optional) |
| `EPOCHS`       | `100`           | Embedding training epochs (set `1` for a quick run) |
| `YOLO_EPOCHS`  | `30`            | YOLO detector training epochs (default 30) |
| `RUN_YOLO_ONLY`| *(unset)*       | Set to `1` to skip download and embedding training and only run YOLO detector training |
| `BATCH_SIZE`   | `32`            | Default 32 if unset            |
| `TRIPLETS_PER_EPOCH` | `10000` | Default 10000 if unset   |

If you only set **`DATA_ROOT=/data`**, the script uses `/data/mapillary` and `/data/output` by default.

**Quick run (1 epoch + YOLO):** Set `EPOCHS=1` to train the embedding model for one epoch only, then the script still runs YOLO detector training. Your tar from `/data/output` will include `best_model.pth`, `checkpoint.pt`, and `traffic_sign_yolo.pt`.

**YOLO only (finish just the detector):** If embedding training and gallery are already done and you only need to run the YOLO step (e.g. after a crash during YOLO training), set `RUN_YOLO_ONLY=1`. The script will skip download and embedding training and only run `train_yolo_detector.py`, writing `traffic_sign_yolo.pt` to `/data/output`. After that, unset `RUN_YOLO_ONLY` or remove it so the next run does the full pipeline again if needed.

---

## 5. Set the start command

Tell Railway to run the download + train script instead of a web server:

1. Service → **Settings** (or **Deploy**).
2. Find **Build** / **Deploy** section.
3. Set **Start Command** (or **Custom Start Command**) to:

```bash
bash scripts/run_railway.sh
```

If your repo has no `scripts/run_railway.sh` yet, add it (see repo) and push to `main`, then redeploy.

---

## 6. Deploy and run

1. **Deploy** the service (or push to `main` if you already connected the repo).
2. Railway will:
   - Build the app (install Python deps).
   - Run `bash scripts/run_railway.sh`.
3. The script will:
   - `pip install -r requirements.txt` and `pip install -e .`
   - Run `download_dataset.py` → writes to `/data/mapillary` (~110 GB).
   - Run `train.py` → reads `/data/mapillary`, writes to `/data/output` (embedding model: `best_model.pth`, `checkpoint.pt`).
   - Run `train_yolo_detector.py` → builds synthetic sign images, fine-tunes YOLOv8 to detect only traffic signs, writes `traffic_sign_yolo.pt` to `/data/output`.

**First run:** Download can take **several hours** (e.g. 3–12 h depending on connection). Training after that can take **~12–24 hours** on CPU. Check **Logs** in the Railway dashboard to follow progress.

---

## 7. Getting the trained model

- **Outputs** are under the volume at **`/data/output`**, e.g.:
  - `best_model.pth`, `checkpoint.pt` (embedding model)
  - `traffic_sign_yolo.pt` (custom YOLO detector — use this so only signs are detected, not people/cars)
  - `gallery.json`, `gallery.npz` (built after full training or via `build_gallery.py`)
- Railway doesn’t expose the volume as a direct download. Options:
  - **One-off shell:** Use **Run Command** (or a temporary shell) and run e.g. `tar czf /tmp/out.tar.gz /data/output` then use a file server or `railway run` to pull it.
  - **Persist in repo:** Add a small script that uploads `output/*.pth` and `output/gallery.*` to S3/GCS or pushes to GitHub Releases from inside the container (using a token in env).
  - **Copy via CLI:** If you use Railway CLI and can open a shell, `railway run` and `scp`/`rsync` are possible if you run a one-off service that serves the files.

---

## 8. Resuming training

If the run stops (e.g. you stop the service) and you want to resume:

- Leave **`DATA_ROOT=/data`** and **same volume**.
- Start the service again with the same start command. The script will see `/data/output/checkpoint.pt` and pass `--checkpoint /data/output/checkpoint.pt` to `train.py`.

No need to re-download the dataset if `/data/mapillary` is still on the volume.

---

## 9. Optional: run only download or only train

- **Download only** (e.g. to fill the volume once):

  Start command:
  ```bash
  pip install -r requirements.txt && pip install -e . && python scripts/download_dataset.py --output-dir /data/mapillary
  ```

- **Train only** (dataset already in `/data/mapillary`):

  Start command:
  ```bash
  pip install -r requirements.txt && pip install -e . && python scripts/train.py --data-dir /data/mapillary --output-dir /data/output --epochs 100 --batch-size 32 --triplets-per-epoch 10000 --embedding-dim 128 --learning-rate 1e-4 --margin 0.3 --image-size 224 --gallery-threshold 0.6 --checkpoint-interval 1
  ```

---

## 10. Summary checklist

- [ ] Railway project created from **ruslan7403/model-uzcosmos**.
- [ ] Volume **≥ 200 GB** mounted at **`/data`**.
- [ ] Variables: **`DATA_ROOT=/data`** (and optional overrides).
- [ ] Start command: **`bash scripts/run_railway.sh`**.
- [ ] Deploy and watch **Logs** for download then training.
- [ ] Plan how to copy **`/data/output`** (e.g. Run Command + tar, or upload script).
