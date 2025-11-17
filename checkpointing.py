import datetime
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

MY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = MY_DIR / "big-train-checkpoints"


def load_checkpoint(checkpoint, model, optimizer=None, scaler=None):
    checkpoint_dir = CHECKPOINTS_DIR / checkpoint
    model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"))

    if optimizer:
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt"))

    if scaler:
        scaler.load_state_dict(torch.load(checkpoint_dir / "scaler.pt"))

    with open(checkpoint_dir / "meta.json", "r") as f:
        meta = json.load(f)
        restart_ds_offset = meta["train_ds_offset"] + 1

    with open(CHECKPOINTS_DIR / "best" / "meta.json") as f:
        best_loss = json.load(f)["val_loss"]

    return restart_ds_offset, best_loss


def save_checkpoint(
    name,
    model, optimizer, scaler,
    train_loss, val_loss,
    train_ds_offset, is_best
):
    if not CHECKPOINTS_DIR.exists():
        CHECKPOINTS_DIR.mkdir()

    now = datetime.datetime.now(datetime.UTC)
    checkpoint_name = f"{now:%Y%m%dZ%H%M%S}-{name}"
    checkpoint_dir = CHECKPOINTS_DIR / checkpoint_name
    checkpoint_dir.mkdir()

    save_file(model.state_dict(), checkpoint_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scaler.state_dict(), checkpoint_dir / "scaler.pt")

    with open(checkpoint_dir / "meta.json", "w") as f:
        json.dump(
            dict(
                train_loss=train_loss,
                val_loss=val_loss,
                train_ds_offset=train_ds_offset,
                is_best=is_best,
            ),
            f
        )

    symlink_target = Path(".") / checkpoint_dir.name
    if is_best:
        best_path = CHECKPOINTS_DIR / "best"
        best_path.unlink(missing_ok=True)
        best_path.symlink_to(symlink_target, target_is_directory=True)

    latest_path = CHECKPOINTS_DIR / "latest"
    latest_path.unlink(missing_ok=True)
    latest_path.symlink_to(symlink_target, target_is_directory=True)
