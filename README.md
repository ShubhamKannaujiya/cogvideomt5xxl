---

# CogVideoX + mT5-XXL Fine-Tuning

This repository contains a training script to fine-tune **THUDM/CogVideoX-2B** with a **google/mt5-xxl** text encoder on your own video–text dataset.

The script:

* Trains **only**:

  * cross-attention / attention layers of the CogVideoX transformer
  * top-2 layers of mT5-XXL
* Uses **bfloat16** mixed precision
* Saves:

  * rolling **latest** checkpoint
  * multiple **best_step_XXXXX** checkpoints (by validation loss)
  * a **FULL** export of the entire pipeline for inference: `CogVideoX-mT5XXL-full/`

---

## 1. Hardware & Requirements

* **GPU**: At least **1 high-VRAM GPU** (e.g. A100 80GB or similar)
* **OS**: Linux recommended (works on Vast.ai / CUDA docker images)
* **Python**: 3.10+ recommended

### Key Python packages

* `torch` (with CUDA)
* `transformers`
* `diffusers`
* `accelerate` (optional but often pulled with diffusers)
* `decord`
* `pandas`
* `tqdm`

---

## 2. Environment Setup

Example using `conda`:

```bash
# 1. Create and activate env
conda create -n cogx python=3.10 -y
conda activate cogx

# 2. Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
pip install transformers diffusers accelerate
pip install decord pandas tqdm
```

If you use a venv instead of conda:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # if you create one
```

---

## 3. Repository Layout

Typical layout:

```text
.

├── Dataset
├── train.py                 # the script you pasted
├── train.csv
├── val.csv
├── checkpoints_mt5xxl/      # will be created automatically
│   ├── latest/
│   ├── best_step_02500/
│   ├── best_step_05000/
│   └── ...
└── CogVideoX-mT5XXL-full/   # full final pipeline for inference
```




## 4. Dataset Format

The script expects **two CSV files** in the working directory:

* `train.csv`
* `val.csv`

Each CSV must have at least these columns:

* **`text`** – the text prompt / caption for the video
* **`video_path`** – absolute or relative path to the video file on disk

Example `train.csv`:

```csv
text,video_path
"एक बिल्ली खिड़की से बाहर देख रही है","/data/videos/cat1.mp4"
"A dog running on the beach","/data/videos/dog_beach.mp4"
"एक आदमी बारिश में चलते हुए","/data/videos/rain_man.mp4"
```

**Important:**

* `video_path` must be valid paths that your training machine can read.
* The script will automatically clean missing files:

  * `clean_csv()` removes rows where `video_path` does not exist on disk and rewrites `train.csv` / `val.csv`.

Supported video formats depend on **decord** (e.g. `.mp4` is fine).

---

## 5. What the Script Does (High-Level)

1. **Configuration**
   At the top:

   ```python
   MODEL_ID   = "THUDM/CogVideoX-2b"
   ENCODER_ID = "google/mt5-xxl"

   TRAIN_CSV  = "train.csv"
   VAL_CSV    = "val.csv"

   BATCH_SIZE = 32
   NUM_FRAMES = 16
   TARGET_H   = 480
   TARGET_W   = 720

   LR_TRANSFORMER = 1e-5
   LR_MT5         = 1e-6

   MAX_STEPS      = 17500
   EVAL_INTERVAL  = 200
   MAX_VAL_BATCHES = 64
   KEEP_BEST_K    = 5
   ```

2. **CSV cleanup**

   * Removes rows with missing video files and overwrites `train.csv` / `val.csv`.

3. **Model loading**

   * Loads `MT5Tokenizer` + `MT5EncoderModel` (`google/mt5-xxl`)
   * Freezes all mT5 layers except top-2 (blocks 22 and 23)
   * Loads `CogVideoXPipeline` from `THUDM/CogVideoX-2b` with:

     * `text_encoder` set to the mT5 encoder
     * `tokenizer` set to mT5 tokenizer
   * Freezes VAE
   * Marks only **attention / cross-attention** parameters in `pipe.transformer` as trainable.

4. **Dataset & dataloaders**

   * Reads CSVs with `VideoCSVDataset`
   * Uses `decord.VideoReader` to grab **16 frames per video** evenly spaced.
   * Resizes frames to `(C=3, T=16, H=480, W=720)` (default)
   * Tokenizes text to max length 226.
   * `train_loader`: batch size = `BATCH_SIZE`, `num_workers=2`
   * `val_loader`: batch size = 4

5. **Loss**

   * Encodes video to VAE latents
   * Adds noise using the scheduler
   * Passes through `pipe.transformer` with `encoder_hidden_states = mT5 embeddings`
   * Computes MSE between predicted noise and ground-truth noise.

6. **Training loop**

   * Uses `AdamW` with different learning rates for transformer vs mT5.
   * Cosine LR scheduler with warmup.
   * Mixed precision with `torch.cuda.amp` and optional `GradScaler`.
   * Gradient clipping on both transformer and mT5 trainable params.
   * Measures timing for:

     * data load
     * text encoder
     * diffusion/vae
     * backward

7. **Validation + checkpoints**

   * Every `EVAL_INTERVAL` steps:

     * Runs validation over `MAX_VAL_BATCHES` mini-batches.
     * Tracks running **best validation loss**.
     * If new best: saves `checkpoints_mt5xxl/best_step_XXXXX/` and updates rolling **latest**.
     * Keeps only `KEEP_BEST_K` best checkpoints; deletes older worst ones.

8. **Final full save**

   * At the end, saves:

     ```text
     CogVideoX-mT5XXL-full/
       ├── transformer/
       ├── vae/
       ├── text_encoder/
       ├── tokenizer/
       ├── scheduler/
       ├── feature_extractor/  (if exists)
       └── config files
     ```

   * This folder can be used as a **standalone pipeline** for inference.

---

## 6. Running Training From Scratch

Assuming:

* You’re in the repo root.
* `train.py` is this script.
* `train.csv` and `val.csv` are prepared.

```bash
conda activate cogx   # or your venv
python3 train.py
```

The script will:

1. Clean the CSVs (removing missing video rows).
2. Download `THUDM/CogVideoX-2b` and `google/mt5-xxl` on first run.
3. Start training from **step 0**.
4. Periodically run validation and write:

   * `checkpoints_mt5xxl/latest/`
   * `checkpoints_mt5xxl/best_step_XXXXX/`
5. At the end, save the full pipeline in `CogVideoX-mT5XXL-full/`.

If you want a **fresh run** and you previously had checkpoints, you can:

```bash
rm -rf checkpoints_mt5xxl CogVideoX-mT5XXL-full
python3 train.py
```

*(Be careful: this deletes previous checkpoints.)*

---

## 7. Resuming Training

The script already has **resume logic**.

By default, it tries to resume from:

```python
latest_dir = OUT_DIR / "latest"
resume_dir = latest_dir
```

So if `checkpoints_mt5xxl/latest/` exists with `training_state.pt`, it will:

* Load:

  * `global_step`
  * `best_loss`
  * optimizer / scheduler / scaler state
* Reload pipeline weights from `resume_dir`
* Rebuild optimizer & param lists properly
* Continue training from `global_step + 1`

### Resume from a specific best step

If you want to resume from a particular `best_step_XXXXX` instead of `latest`:

In the script, change:

```python
#resume_step = 2500 
#resume_dir  = OUT_DIR / f"best_step_{resume_step:05d}"
latest_dir = OUT_DIR / "latest" 
resume_dir = latest_dir                         #commennt to resume from a specific step
```

to something like:

```python
resume_step = 2500
resume_dir  = OUT_DIR / f"best_step_{resume_step:05d}"
# latest_dir = OUT_DIR / "latest"
```

Then run:

```bash
python3 train.py
```

It will resume from `checkpoints_mt5xxl/best_step_02500/`.

---

## 8. Checkpoint Structure

Each checkpoint directory (e.g. `best_step_02500/`, `latest/`) contains:

* `training_state.pt`

  * `global_step`
  * `epoch`
  * `best_loss`
  * `current_val_loss`
  * optimizer state
  * scheduler state
  * scaler state (if used)

* Full pipeline weights saved via `pipe.save_pretrained(...)`:

  * `model_index.json`
  * subdirectories for components as per diffusers format.

This makes it easy to:

* Use **diffusers** to reload the pipeline.
* Resume training or run inference from any checkpoint.

---

## 9. Using the Full Saved Model for Inference (Short Hint)

Once training finishes, you will have:

```text
CogVideoX-mT5XXL-full/
```

You can later load it like:

```python
from diffusers import CogVideoXPipeline
import torch

pipe = CogVideoXPipeline.from_pretrained(
    "CogVideoX-mT5XXL-full",
    torch_dtype=torch.bfloat16
).to("cuda")

prompt = "एक शांत गाँव में बारिश हो रही है"
video = pipe(prompt, num_frames=16).videos  # example, depends on diffusers version
```

(Your actual `inference.py` can be more sophisticated, but this is the basic idea.)

---

## 10. Notes & Tips

* **VRAM**: `google/mt5-xxl` + `CogVideoX-2b` is heavy. bfloat16 (`DTYPE = torch.bfloat16`) helps, but you still need a large GPU.
* **NUM_WORKERS_T / NUM_WORKERS_V**:
  If you get dataloader crashes or dataloader OOM, you can lower them:

* **Batch size**:
  Default is `BATCH_SIZE = 32` for training. If you get CUDA OOM, reduce it (e.g. 8 or 4).

  
---
