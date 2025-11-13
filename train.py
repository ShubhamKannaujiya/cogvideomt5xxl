#!/usr/bin/env python3
# Debug + light optimization wrapper around your original script.
# - Adds timing breakdowns (data load, encoder, diffusion/vae, backward)
# - Caches VideoReader objects per-worker to avoid re-opening per sample
# - Keeps all numeric hyperparameters unchanged.

import os, math, gc, random, time
from pathlib import Path
import shutil 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

import pandas as pd
from tqdm import tqdm

from transformers import MT5Tokenizer, MT5EncoderModel, get_cosine_schedule_with_warmup
from diffusers import CogVideoXPipeline

from decord import VideoReader, cpu

# =========================
# CONFIG 
# =========================
MODEL_ID        = "THUDM/CogVideoX-2b"
ENCODER_ID      = "google/mt5-xxl"

DEVICE          = "cuda"
DTYPE           = torch.bfloat16
SEED            = 1234

TRAIN_CSV       = "train.csv"
VAL_CSV         = "val.csv"

BATCH_SIZE      = 32
NUM_FRAMES      = 16   #fixed value for cogvideox
TARGET_H        = 480
TARGET_W        = 720

LR_TRANSFORMER  = 1e-5
LR_MT5          = 1e-6
WARMUP_STEPS    = 500
MAX_STEPS       = 17500
GRAD_CLIP       = 1.0
EVAL_INTERVAL   = 200
MAX_VAL_BATCHES = 64
KEEP_BEST_K     = 5

OUT_DIR         = Path("checkpoints_mt5xxl")
LATEST_DIR      = OUT_DIR / "latest" 
FULL_SAVE_DIR   = Path("CogVideoX-mT5XXL-full")

NUM_WORKERS_T   = 2
NUM_WORKERS_V   = 2
NORMALIZE_TO_MINUS1_1 = True
MISSING_FILE_STRATEGY = "fallback"


os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LATEST_DIR, exist_ok=True)
os.makedirs(FULL_SAVE_DIR, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import pandas as pd, os

def clean_csv(path, out_path=None):
    df = pd.read_csv(path)
    ok = df["video_path"].apply(os.path.exists)
    df2 = df[ok].copy()
    print(f"CSV {path}: {len(df)-len(df2)} missing files removed; {len(df2)} remain.")
    if out_path:
        df2.to_csv(out_path, index=False)
        return out_path
    else:
        df2.to_csv(path, index=False)
        return path

TRAIN_CSV  = clean_csv(TRAIN_CSV)
VAL_CSV    = clean_csv(VAL_CSV)

# =========================
# REPRO
# =========================
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False, warn_only=True)
set_seed(SEED)

def log(msg: str): print(msg, flush=True)

def resize_video_CT_HW(frames_CT_HW: torch.Tensor, H: int, W: int) -> torch.Tensor:
    x = frames_CT_HW.permute(1, 0, 2, 3).float()  # [T,C,H,W]
    x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return x.permute(1, 0, 2, 3)                # [C,T,H,W]

def summarize_missing(csv_path: str, limit: int = 5):
    if not os.path.exists(csv_path):
        log(f"CSV {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    paths = df["video_path"].tolist()
    miss = [p for p in paths if not os.path.exists(p)]
    if miss:
        log(f"  {len(miss)} missing videos in {csv_path} (showing up to {limit}):")
        for m in miss[:limit]: log(f"    - {m}")

# TOKENIZER + ENCODER
tokenizer = MT5Tokenizer.from_pretrained(ENCODER_ID)
text_encoder = MT5EncoderModel.from_pretrained(
    ENCODER_ID,
    torch_dtype=DTYPE,
    device_map="auto"
)

# Freeze all but top-2 encoder blocks
for n, p in text_encoder.named_parameters():
    if not any(f"block.{i}." in n for i in [22, 23]):
        p.requires_grad = False

# PIPELINE
pipe = CogVideoXPipeline.from_pretrained(
    MODEL_ID,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    torch_dtype=DTYPE
).to(DEVICE)

if hasattr(pipe, "enable_gradient_checkpointing"):
    pipe.enable_gradient_checkpointing()
elif hasattr(pipe.transformer, "enable_gradient_checkpointing"):
    pipe.transformer.enable_gradient_checkpointing()

for p in pipe.vae.parameters():
    p.requires_grad = False

def mark_xattn_trainable(m):
    n_train = 0
    for n, p in m.named_parameters():
        if (".attn" in n) or ("attention" in n) or (".to_q" in n) or (".to_k" in n) or (".to_v" in n) or (".to_out" in n):
            p.requires_grad = True
            n_train += p.numel()
        else:
            p.requires_grad = False
    return n_train
# Only train attention layers in transformerS
trained_xattn = mark_xattn_trainable(pipe.transformer)

log(f" Trainable transformer x-attn params: {trained_xattn/1e6:.2f}M")
log(f" Trainable mT5 params (top-2 layers): {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)/1e6:.2f}M")

# =========================
# CHECKPOINT MANAGEMENT HELPERS
# =========================
def save_checkpoint(step, epoch, current_loss, best_loss, is_best, optimizer, scheduler, pipe, tokenizer, scaler, out_dir):
    """Handles saving the model and training state."""
    save_path = out_dir / f"best_step_{step:05d}" if is_best else LATEST_DIR

    # 1. Save model weights
    save_path.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 2. Save training state
    state = {
        'global_step': step,
        'epoch': epoch,
        'best_loss': best_loss,
        'current_val_loss': current_loss, 
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state': scaler.state_dict(), # <-- OPTIONAL SCALER SAVE
    }
    torch.save(state, save_path / "training_state.pt")

    log(f" Checkpoint saved to: {save_path}")

def cleanup_checkpoints(k_best: int = KEEP_BEST_K):
    """Keeps only the top k 'best' checkpoints by validation loss."""
    best_dirs = []
    
    # 1. Find all 'best_step_XXXXX' directories and their loss
    for d in OUT_DIR.iterdir():
        if d.is_dir() and d.name.startswith("best_step_"):
            ckpt_file = d / "training_state.pt"
            if ckpt_file.exists():
                try:
                    state = torch.load(ckpt_file, map_location="cpu")
                    loss = state.get("current_val_loss")
                    if loss is not None:
                        best_dirs.append((loss, d))
                except Exception as e:
                    log(f" Could not load state from {ckpt_file}: {e}")
                    
    # 2. Sort by loss (ascending)
    best_dirs.sort(key=lambda x: x[0])
    
    # 3. Identify and delete the worst ones
    if len(best_dirs) > k_best:
        to_delete = best_dirs[k_best:]
        log(f"🗑 Cleaning up {len(to_delete)} old 'best' checkpoints (keeping top {k_best})...")
        for loss, d in to_delete:
            log(f"    - Deleting: {d.name} (Loss: {loss:.6f})")
            shutil.rmtree(d)
        torch.cuda.empty_cache(); gc.collect()


# =========================
# DATASET
# =========================
class VideoCSVDataset(Dataset):
    def __init__(self, csv_path, tokenizer, num_frames=NUM_FRAMES, target_h=TARGET_H, target_w=TARGET_W):
        df = pd.read_csv(csv_path)
        self.texts  = df["text"].tolist()
        self.videos = df["video_path"].tolist()
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.H, self.W = target_h, target_w
        self._reader_cache = {}

    def __len__(self): return len(self.texts)

    def _get_reader(self, path):
        if path not in self._reader_cache:
            try:
                self._reader_cache[path] = VideoReader(path, ctx=cpu())
            except Exception:
                self._reader_cache[path] = None
        return self._reader_cache[path]

    def __getitem__(self, idx):
        txt  = str(self.texts[idx])
        path = str(self.videos[idx])

        frames = None
        if os.path.exists(path):
            try:
                vr = self._get_reader(path)
                if vr is not None:
                    n = len(vr)
                    if n > 0:
                        idxs = torch.linspace(0, n - 1, self.num_frames).long()
                        batch = vr.get_batch(idxs)                      # [T,H,W,C], uint8
                        frames = (batch.permute(0, 3, 1, 2) / 255.).permute(1, 0, 2, 3)  # [C,T,H,W]
                        frames = resize_video_CT_HW(frames, self.H, self.W)
            except Exception:
                if MISSING_FILE_STRATEGY == "raise":
                    raise
                frames = None

        if frames is None:
            if MISSING_FILE_STRATEGY == "raise":
                raise FileNotFoundError(path)
            frames = torch.rand(3, self.num_frames, self.H, self.W)  # fallback

        toks = self.tokenizer(
            txt, return_tensors="pt",
            truncation=True, padding="max_length", max_length=226
        )
        return {"input_ids": toks["input_ids"].squeeze(0), "video": frames}

summarize_missing(TRAIN_CSV)
summarize_missing(VAL_CSV)

train_ds = VideoCSVDataset(TRAIN_CSV, tokenizer)
val_ds   = VideoCSVDataset(VAL_CSV,   tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS_T, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    num_workers=0,                     # <-- crucial
    pin_memory=False,                  
    persistent_workers=False,          
)

# =========================
# LOSS
# =========================
def diffusion_loss(pipe, video_frames, text_embeds):
    video_in = (video_frames * 2 - 1) if NORMALIZE_TO_MINUS1_1 else video_frames
    latents = pipe.vae.encode(video_in).latent_dist.sample() * 0.18215
    B, C, T, H, W = latents.shape
    latents_5d = latents.permute(0, 2, 1, 3, 4).contiguous()
    t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (B,), device=latents_5d.device).long()
    noise = torch.randn_like(latents_5d)
    noisy_latents = pipe.scheduler.add_noise(latents_5d, noise, t)
    out = pipe.transformer(
        encoder_hidden_states=text_embeds,
        hidden_states=noisy_latents,
        timestep=t
    )
    noise_pred = out.sample if hasattr(out, "sample") else out
    if noise_pred.shape != noise.shape:
        noise_pred = noise_pred.view_as(noise)
    return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

# =========================
# TRAIN STATE + OPTIMIZER INITIALIZATION (Pre-Resume)
# =========================
best_loss   = float("inf")
global_step = 0

trainable_transformer = [p for p in pipe.transformer.parameters() if p.requires_grad]
trainable_mt5         = [p for p in text_encoder.parameters()      if p.requires_grad]

params = [
    {"params": trainable_transformer, "lr": LR_TRANSFORMER},
    {"params": trainable_mt5,          "lr": LR_MT5},
]
optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, MAX_STEPS)
use_scaler = (DTYPE == torch.float16)
scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)


# =========================
# RESUME LOGIC (CRITICAL FIXES APPLIED)
# =========================

# Scenario:
#   • Fresh training run → comment the entire block out.
#   • Resume from an explicit checkpoint → set resume_step to the desired best_step number that already exists in OUT_DIR
#resume_step = 2500 
#resume_dir  = OUT_DIR / f"best_step_{resume_step:05d}"
latest_dir = OUT_DIR / "latest" 
resume_dir = latest_dir                         #commennt to resume from a specific step

# Fallback to 'latest'
if not resume_dir.exists() and latest_dir.exists():
    log(f" Specific resume step {resume_step} not found. Falling back to latest checkpoint: {latest_dir}")
    resume_dir = latest_dir
    ckpt_path = latest_dir / "training_state.pt"
    if ckpt_path.exists():
        try:
             ckpt = torch.load(ckpt_path, map_location="cpu")
             resume_step = ckpt.get("global_step", global_step)
             log(f"Found global step {resume_step} in 'latest' state.")
        except:
             log(" Could not read global step from 'latest' training state.")


if resume_dir.exists():
    log(f" Loading checkpoint from {resume_dir}")

    # ---- (1) Load training state file ----
    ckpt_path = resume_dir / "training_state.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Update variables from checkpoint
        best_loss   = ckpt.get("best_loss",   best_loss)
        global_step = ckpt.get("global_step", global_step)

    else:
        log(" training_state.pt not found — resuming weights only")
        ckpt = {} # Ensure ckpt exists for the load_state_dict calls below

    # ---- (2) Reload pipeline (Creates NEW parameter objects) ----
    pipe = CogVideoXPipeline.from_pretrained(
        resume_dir,
        text_encoder=text_encoder,    
        tokenizer=tokenizer,
        torch_dtype=DTYPE,            
    ).to(DEVICE)

    # ---- (3) Re-enable gradient checkpointing ----
    if hasattr(pipe, "enable_gradient_checkpointing"):
        pipe.enable_gradient_checkpointing()
    elif hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()

    # ---- (4) Freeze VAE again ----
    for p in pipe.vae.parameters():
        p.requires_grad = False

    # extra memory-safety
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    # ---- (5) CRITICAL FIX: Rebuild Optimizer and Param Lists ----
    # Re-mark xattn trainable (necessary as pipe was reloaded)
    trained_xattn = mark_xattn_trainable(pipe.transformer) 

    # Rebuild param lists (necessary as pipe was reloaded)
    trainable_transformer = [p for p in pipe.transformer.parameters() if p.requires_grad]
    trainable_mt5         = [p for p in text_encoder.parameters()      if p.requires_grad]

    # Rebuild optimizer/scheduler with NEW param objects (CRITICAL)
    params = [
        {"params": trainable_transformer, "lr": LR_TRANSFORMER},
        {"params": trainable_mt5,         "lr": LR_MT5},
    ]
    optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, MAX_STEPS)

    # Now load state dicts (connects old state to new optimizer instance)
    if 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            log(f" Optimizer state not loaded cleanly: {e}")

    if 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            log(f" Scheduler state not loaded cleanly: {e}")
            
    if 'scaler_state' in ckpt and use_scaler: # <-- SCALER LOAD
        try:
            scaler.load_state_dict(ckpt['scaler_state'])
        except Exception as e:
            log(f" Scaler state not loaded cleanly: {e}")

    # ---- (6) Back to train-mode ----
    pipe.transformer.train()
    pipe.text_encoder.train()

    log(f" (resumed) Trainable x-attn params: {trained_xattn/1e6:.2f}M")
    log(f" (resumed) Trainable mT5 params: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)/1e6:.2f}M")
    log(f" Resume OK → continuing from step {global_step + 1}")

else:
    log(" No resume directory found → starting fresh")

# =========================
# TRAIN LOOP WITH TIMING (FIXED: best_loss reset removed)
# =========================
# best_loss is NOT reset here, it keeps its value (either inf or loaded value)
pipe.transformer.train()
pipe.text_encoder.train()

steps_per_epoch = max(1, len(train_loader))
num_epochs = math.ceil(MAX_STEPS / steps_per_epoch)
val_ptr = 0 
timing = {"data":0.0, "text_enc":0.0, "diff":0.0, "back":0.0, "other":0.0}
timing_count = 0

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    for batch in pbar:
        if global_step >= MAX_STEPS:
            break

        t0 = time.time()
        videos    = batch["video"].to(DEVICE, dtype=DTYPE, non_blocking=True)
        input_ids = batch["input_ids"].to(DEVICE,               non_blocking=True)
        t_data_done = time.time()

        with torch.cuda.amp.autocast(dtype=DTYPE):
            t_text0 = time.time()
            text_out = text_encoder(input_ids=input_ids, return_dict=True)
            embeds   = text_out.last_hidden_state
            t_text1 = time.time()
            loss     = diffusion_loss(pipe, videos, embeds)
            t_diff_done = time.time()

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_transformer, GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(trainable_mt5,GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_transformer, GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(trainable_mt5,         GRAD_CLIP)
            optimizer.step()
        t_back_done = time.time()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        t_done = time.time()

        # accumulate timings
        timing["data"] += (t_data_done - t0)
        timing["text_enc"] += (t_text1 - t_text0)
        timing["diff"] += (t_diff_done - t_text1)
        timing["back"] += (t_back_done - t_diff_done)
        timing["other"] += (t_done - t_back_done)
        timing_count += 1

        global_step += 1

        if global_step % 10 == 0:
            avg = {k: timing[k]/timing_count for k in timing}
            pbar.set_postfix(step=global_step, loss=f"{loss.item():.6f}", avg_data=f"{avg['data']:.3f}s", avg_diff=f"{avg['diff']:.3f}s")

        # VALIDATION + SAVE
        if global_step % EVAL_INTERVAL == 0:
            log(f"⏱ Running validation at step {global_step}...")
            
            pipe.transformer.eval()
            pipe.text_encoder.eval()
            
            # ===== Validation Loop =====
            val_pbar = tqdm(val_loader, desc="Validation", dynamic_ncols=True, leave=False)
            
            start = val_ptr
            end   = val_ptr + MAX_VAL_BATCHES
            total, count = 0.0, 0
            
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
                for i, vb in enumerate(val_pbar):
                    if i < start:
                        continue
                    if i >= end:
                        break
            
                    v = vb["video"].to(DEVICE, dtype=DTYPE, non_blocking=True)
                    e = text_encoder(
                        input_ids=vb["input_ids"].to(DEVICE, non_blocking=True)
                    ).last_hidden_state
            
                    l = diffusion_loss(pipe, v, e)
                    total += l.item()
                    count += 1
                    val_pbar.set_postfix(val_batch_loss=f"{l.item():.6f}")
            
            # Advance pointer & wrap
            val_ptr = end
            if val_ptr >= len(val_loader):
                val_ptr = 0
            
            val_loss = total / max(count, 1)
            
            log(f" Validation loss @ {global_step}: {val_loss:.6f} (samples={count})")
            
            pipe.transformer.train()
            pipe.text_encoder.train()


            # ====== 2. Save new best OR just latest ======
            if val_loss < best_loss:
                best_loss = val_loss   #  update before saving
            
                # save BEST
                save_checkpoint(
                    global_step, epoch, val_loss, best_loss,
                    is_best=True,
                    optimizer=optimizer, scheduler=scheduler,
                    pipe=pipe, tokenizer=tokenizer, scaler=scaler, out_dir=OUT_DIR
                )
                log(f"New best → {OUT_DIR / f'best_step_{global_step:05d}'}")
            
                cleanup_checkpoints(KEEP_BEST_K)
                torch.cuda.empty_cache(); gc.collect()
            
            # always save latest after best check
            save_checkpoint(
                global_step, epoch, val_loss, best_loss,
                is_best=False,  # latest dir
                optimizer=optimizer, scheduler=scheduler,
                pipe=pipe, tokenizer=tokenizer, scaler=scaler, out_dir=OUT_DIR
            )   
    if global_step >= MAX_STEPS:
        break

log(f"\n Training finished. Best validation loss: {best_loss:.6f}")
log("Timing averages (s): " + ", ".join([f"{k}={timing[k]/max(1,timing_count):.3f}" for k in timing]))

#Save full model parts (same as original)
# log("\n Saving FULL standalone model...")
# parts = {
    
#     "transformer":         pipe.transformer,
#     "vae":                 pipe.vae,
#     "text_encoder":        pipe.text_encoder,
#     "tokenizer":           getattr(pipe, "tokenizer", None),
#     "scheduler":           getattr(pipe, "scheduler", None),
#     "feature_extractor":   getattr(pipe, "feature_extractor", None),
# }
# for name, module in parts.items():
#     if module is not None and hasattr(module, "save_pretrained"):
#         save_dir = FULL_SAVE_DIR / name
#         save_dir.mkdir(parents=True, exist_ok=True)
#         module.save_pretrained(str(save_dir))

# pipe.save_config(str(FULL_SAVE_DIR))
# log(f" Full standalone model saved → {FULL_SAVE_DIR}")
# =========================================================
# FINAL SAVE: COMPLETE CogVideoX PIPELINE FOR INFERENCE
# This saves ALL fine-tuned weights and the full architecture.
# =========================================================

save_dir = "CogVideoX-mT5XXL-full"
os.makedirs(save_dir, exist_ok=True)

pipe.transformer.save_pretrained(f"{save_dir}/transformer")
pipe.vae.save_pretrained(f"{save_dir}/vae")
pipe.text_encoder.save_pretrained(f"{save_dir}/text_encoder")
pipe.tokenizer.save_pretrained(f"{save_dir}/tokenizer")
pipe.scheduler.save_pretrained(f"{save_dir}/scheduler")
if getattr(pipe, "feature_extractor", None) is not None:
    pipe.feature_extractor.save_pretrained(f"{save_dir}/feature_extractor")
pipe.save_config(save_dir)
print("FULL MODEL SAVED:", save_dir)


gc.collect()
torch.cuda.empty_cache()