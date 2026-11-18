"""Training loop: game-level split, mixed precision, resumable checkpoints, top-1/top-3 logging.

Set up for one RTX 3060 (12 GB). Runs unchanged on Colab or on a CPU / Mac, just slower.

What makes it fast on the 3060:
  * The whole pre-encoded dataset goes onto the GPU once (the 7.5M Lichess positions are
    about 2 GB as bitboards), so there's no DataLoader and the GPU never waits for data.
    If a dataset is ever too big for the card it stays in RAM instead, automatically.
  * bf16 mixed precision (Ampere cards like the 3060 support bf16, so no GradScaler).
  * TF32 for any float32 math, cuDNN autotuning, optional torch.compile.

What makes it safe to leave running:
  * A checkpoint every epoch (last.pt) plus the best one by validation top-1 (best.pt) in
    models/runs/<config name>/, along with log.csv and a copy of the config.
  * --resume continues a stopped run (power cut, Colab disconnect) from last.pt.
  * Trains only on split "train" and validates on "val". The "test" games are never touched
    here; evaluate.py scores them once a model is final.

Usage (repo root, after `pip install -e .` and pre-encoding with `python -m chesspred.encoding`):

    # 0. sanity check: a correct model must memorise 64 positions (~100% top-1)
    python -m chesspred.train configs/resnet_6x64_lichess.yaml --overfit 64

    # 1. pre-train the generic model on Lichess (row E)
    python -m chesspred.train configs/resnet_6x64_lichess.yaml

    # 2. fine-tune it on your games (row F; the config's init_from points at step 1's best.pt)
    python -m chesspred.train configs/resnet_6x64_finetune_whopizza1.yaml

    # continue a stopped run, or override any config value from the command line
    python -m chesspred.train configs/resnet_6x64_lichess.yaml --resume
    python -m chesspred.train configs/resnet_6x64_lichess.yaml --set name=r10x128 model.blocks=10 model.channels=128
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from chesspred.data.parse_pgn import REPO_ROOT
from chesspred.encoding import SPLIT_CODES, bitboards_to_planes, legal_mask_from_padded
from chesspred.models import build_model, count_params

RUNS_DIR = REPO_ROOT / "models" / "runs"

MODEL_DEFAULTS = {"type": "resnet", "blocks": 6, "channels": 64, "se": False, "elo": False}
DEFAULTS = {
    "name": "run",
    "model": {},  # filled in by resolve_model_config (MODEL_DEFAULTS < init_from < your config)
    "data": {"train": None, "train_positions": None, "keep_on_gpu": "auto"},
    "train": {
        "epochs": 10, "batch_size": 1024, "lr": 1e-3, "weight_decay": 1e-4,
        "warmup_pct": 0.1, "grad_clip": 1.0, "amp": True, "compile": False, "seed": 42,
        "log_every": 500, "init_from": None, "freeze_blocks": 0,
    },
}


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def load_config(path: Path, overrides: list[str] | None = None) -> dict:
    """Read a YAML config on top of DEFAULTS, then apply `section.key=value` overrides."""
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for key, value in user.items():
        if isinstance(value, dict):
            cfg.setdefault(key, {}).update(value)
        else:
            cfg[key] = value
    for item in overrides or []:
        dotted, _, raw = item.partition("=")
        *parents, leaf = dotted.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)  # "10" -> 10, "true" -> True, "null" -> None
    return cfg


def repo_path(p: str | Path) -> Path:
    """Config paths are relative to the repo root, so runs work from any directory."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # the RTX 3060 / Colab
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon
    return torch.device("cpu")


def setup_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # any float32 matmul uses tensor cores
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True          # pick the fastest conv kernels once
        print(f"GPU: {torch.cuda.get_device_name()}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
class PositionData:
    """A pre-encoded .npz as tensors, kept on the GPU when they fit.

    batch(idx) returns ready-to-use (planes, labels, legal mask, ratings) on the GPU.
    Unpacking bitboards and building masks happens on the GPU, which is far faster than
    doing it per position in Python.
    """

    FIELDS = ("boards", "labels", "legal", "elo")

    def __init__(self, path: Path, device: torch.device, keep_on_gpu="auto"):
        t0 = time.time()
        with np.load(path) as z:
            self.meta = json.loads(str(z["meta"]))
            self.split = torch.from_numpy(z["split"])
            tensors = {k: torch.from_numpy(z[k]) for k in self.FIELDS}
        self.device = device
        nbytes = sum(t.nbytes for t in tensors.values())

        if device.type == "cuda" and keep_on_gpu == "auto":
            free, _ = torch.cuda.mem_get_info()
            keep_on_gpu = nbytes < 0.5 * free  # leave at least half the card for the model
        self.on_device = device.type != "cpu" and bool(keep_on_gpu)
        self.storage = device if self.on_device else torch.device("cpu")
        self.t = {k: v.to(self.storage) for k, v in tensors.items()}
        where = "on the GPU" if self.on_device else "in RAM"
        print(f"Loaded {path.name}: {len(self.split):,} positions, {nbytes / 1e9:.2f} GB, "
              f"kept {where} ({time.time() - t0:.0f}s)")

    def indices(self, split_name: str) -> torch.Tensor:
        idx = torch.nonzero(self.split == SPLIT_CODES[split_name]).squeeze(1)
        return idx.to(self.storage)

    def batch(self, idx: torch.Tensor):
        get = lambda k: self.t[k][idx].to(self.device, non_blocking=True)  # noqa: E731
        x = bitboards_to_planes(get("boards"))        # (B, 18, 8, 8) float
        y = get("labels").long()                       # (B,) move index played
        mask = legal_mask_from_padded(get("legal"))    # (B, 4096) bool
        return x, y, mask, get("elo")


def masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Illegal moves -> -inf, so softmax gives them exactly 0 probability."""
    return logits.float().masked_fill(~mask, float("-inf"))


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data: PositionData, idx: torch.Tensor, batch_size: int, amp: bool) -> dict:
    """Validation loss, top-1 and top-3 accuracy over the positions in idx."""
    model.eval()
    loss_sum = torch.zeros((), device=data.device)
    top1 = torch.zeros((), device=data.device)
    top3 = torch.zeros((), device=data.device)
    for i in range(0, len(idx), batch_size):
        x, y, mask, elo = data.batch(idx[i:i + batch_size])
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x, elo)
        logits = masked(logits, mask)
        loss_sum += F.cross_entropy(logits, y, reduction="sum")
        best3 = logits.topk(3, dim=1).indices
        top1 += (best3[:, 0] == y).sum()
        top3 += (best3 == y[:, None]).any(dim=1).sum()
    n = max(len(idx), 1)
    return {"loss": loss_sum.item() / n, "top1": top1.item() / n, "top3": top3.item() / n}


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def save_checkpoint(path: Path, **state) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never corrupts the previous checkpoint


def load_pretrained(model, path: Path) -> None:
    """Load weights from another run (for fine-tuning). Optimizer state is not carried over."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt["model"], strict=False)
    print(f"Initialised from {path} (epoch {ckpt.get('epoch')}, "
          f"val top-1 {ckpt.get('val', {}).get('top1', float('nan')):.1%})")
    if result.missing_keys:
        print(f"  new layers, starting fresh: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  ignored from checkpoint: {result.unexpected_keys}")


def resolve_model_config(cfg: dict) -> dict:
    """The full model config: MODEL_DEFAULTS, then (when fine-tuning) the pre-trained model's
    architecture so its weights fit, then whatever this config's `model:` section sets
    (e.g. `elo: true` to add rating conditioning while fine-tuning)."""
    resolved = dict(MODEL_DEFAULTS)
    init_from = cfg["train"].get("init_from")
    if init_from:
        ckpt = torch.load(repo_path(init_from), map_location="cpu", weights_only=False)
        resolved.update(ckpt["config"]["model"])
    resolved.update(cfg.get("model") or {})
    return resolved


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train(cfg: dict, resume: bool = False) -> Path:
    tc = cfg["train"]
    seed_everything(tc["seed"])
    device = get_device()
    setup_device(device)
    amp = bool(tc["amp"]) and device.type == "cuda"

    run_dir = RUNS_DIR / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "log.csv"
    if last_path.exists() and not resume:
        print(f"Note: {run_dir} already has a run. Starting over and overwriting it "
              f"(use --resume to continue it, or --set name=... for a new run).")
        log_path.unlink(missing_ok=True)

    # ---- data ----
    data = PositionData(repo_path(cfg["data"]["train"]), device, cfg["data"]["keep_on_gpu"])
    train_idx, val_idx = data.indices("train"), data.indices("val")
    cap = cfg["data"].get("train_positions")
    if cap and cap < len(train_idx):  # a fixed random subset, for data-size experiments
        g = torch.Generator().manual_seed(tc["seed"])
        train_idx = train_idx[torch.randperm(len(train_idx), generator=g)[:cap].to(train_idx.device)]
    print(f"train {len(train_idx):,} positions | val {len(val_idx):,} positions")

    # ---- model ----
    cfg["model"] = resolve_model_config(cfg)
    model = build_model(cfg["model"]).to(device)
    if tc["init_from"] and not (resume and last_path.exists()):
        load_pretrained(model, repo_path(tc["init_from"]))
    if tc["freeze_blocks"]:
        model.freeze_early_layers(tc["freeze_blocks"])
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"model {cfg['model']}: {count_params(model):,} params "
          f"({sum(p.numel() for p in params):,} trainable)")

    # ---- optimiser: AdamW + one-cycle (warm up to lr, then anneal towards 0) ----
    bs, epochs = tc["batch_size"], tc["epochs"]
    steps_per_epoch = math.ceil(len(train_idx) / bs)
    optimizer = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tc["lr"], total_steps=epochs * steps_per_epoch,
        pct_start=tc["warmup_pct"])

    start_epoch, best_top1 = 1, -1.0
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt["scheduler"]["total_steps"] != scheduler.total_steps:
            raise SystemExit(
                "--resume continues the SAME run, but epochs / batch_size / training data "
                "changed since it started.\nTo train an existing model further, start a new "
                f"run from it instead:  --set name=<new name> train.init_from={last_path}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_top1 = ckpt["epoch"] + 1, ckpt["best_top1"]
        print(f"Resuming {cfg['name']} from epoch {ckpt['epoch']} (best val top-1 {best_top1:.1%})")
    elif resume:
        print(f"No {last_path} yet, starting fresh")
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    step_model = torch.compile(model) if tc["compile"] else model

    if start_epoch == 1 and tc["init_from"]:
        # Score the pre-trained model before fine-tuning: "does personalisation help?"
        before = evaluate(step_model, data, val_idx, bs, amp)
        print(f"before fine-tuning: val loss {before['loss']:.3f}  "
              f"top-1 {before['top1']:.1%}  top-3 {before['top3']:.1%}")
        append_log(log_path, 0, float("nan"), before, 0.0, 0.0, 0.0)

    # ---- the loop ----
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        perm = train_idx[torch.randperm(len(train_idx), device=train_idx.device)]
        running = torch.zeros((), device=device)  # summed on the GPU: no sync every step
        n_seen = 0
        for step, i in enumerate(range(0, len(perm), bs), 1):
            x, y, mask, elo = data.batch(perm[i:i + bs])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = step_model(x, elo)
            loss = F.cross_entropy(masked(logits, mask), y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            optimizer.step()
            scheduler.step()

            running += loss.detach() * len(y)
            n_seen += len(y)
            if step % tc["log_every"] == 0:
                rate = n_seen / (time.time() - t0)
                print(f"  epoch {epoch} step {step:,}/{steps_per_epoch:,}  "
                      f"loss {running.item() / n_seen:.3f}  {rate:,.0f} pos/s  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

        train_loss = running.item() / max(n_seen, 1)
        train_secs = time.time() - t0
        val = evaluate(step_model, data, val_idx, bs, amp)
        is_best = val["top1"] > best_top1
        best_top1 = max(best_top1, val["top1"])
        print(f"epoch {epoch:2d}/{epochs}  train loss {train_loss:.3f}  |  val loss {val['loss']:.3f}  "
              f"top-1 {val['top1']:.1%}  top-3 {val['top3']:.1%}"
              f"{'  *best*' if is_best else ''}  ({train_secs / 60:.1f} min, "
              f"{n_seen / train_secs:,.0f} pos/s)", flush=True)

        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), epoch=epoch, best_top1=best_top1,
                     val=val, config=cfg)
        save_checkpoint(last_path, **state)
        if is_best:
            shutil.copyfile(last_path, best_path)
        append_log(log_path, epoch, train_loss, val, scheduler.get_last_lr()[0],
                   n_seen / train_secs, train_secs / 60)

    print(f"\nDone. Best val top-1 {best_top1:.1%}  ->  {best_path}")
    return best_path


def append_log(path: Path, epoch, train_loss, val, lr, pos_per_sec, minutes) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "train_loss", "val_loss", "val_top1", "val_top3", "lr",
                        "pos_per_sec", "minutes", "time"])
        w.writerow([epoch, f"{train_loss:.4f}", f"{val['loss']:.4f}", f"{val['top1']:.4f}",
                    f"{val['top3']:.4f}", f"{lr:.2e}", f"{pos_per_sec:.0f}", f"{minutes:.2f}",
                    time.strftime("%Y-%m-%d %H:%M")])


# --------------------------------------------------------------------------------------
# Sanity check: can the model memorise a handful of positions?
# --------------------------------------------------------------------------------------
def overfit_check(cfg: dict, n: int = 64, steps: int = 300) -> None:
    """A correct model + loss + labels memorises n positions. If this doesn't reach ~100%
    top-1, there's a bug (see the debugging checklist in docs/cnn_resnet_tutorial.md)."""
    seed_everything(cfg["train"]["seed"])
    device = get_device()
    setup_device(device)
    data = PositionData(repo_path(cfg["data"]["train"]), device, cfg["data"]["keep_on_gpu"])
    x, y, mask, elo = data.batch(data.indices("train")[:n])
    model = build_model(resolve_model_config(cfg)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        model.train()
        loss = F.cross_entropy(masked(model(x, elo), mask), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                acc = (masked(model(x, elo), mask).argmax(1) == y).float().mean().item()
            print(f"  step {step:3d}  loss {loss.item():.4f}  top-1 on the {n} positions {acc:.1%}")
    print("PASS: the model can learn" if acc > 0.95 else "FAIL: look for a bug before training")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment YAML, e.g. configs/resnet_6x64_lichess.yaml")
    parser.add_argument("--resume", action="store_true", help="continue from models/runs/<name>/last.pt")
    parser.add_argument("--overfit", type=int, metavar="N", default=None,
                        help="sanity check only: try to memorise N training positions")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set train.lr=0.002 model.blocks=10")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.overfit:
        overfit_check(cfg, args.overfit)
    else:
        train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
