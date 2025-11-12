import datetime
import json
from pathlib import Path

import click
from tqdm import tqdm

from matplotlib.ticker import MaxNLocator
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch

from gpt import GPTModel


MY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = MY_DIR / "big-train-checkpoints"
DATASETS_DIR = MY_DIR / "big-train-datasets"


def load_checkpoint(checkpoint, model, optimizer, scaler):
    checkpoint_dir = CHECKPOINTS_DIR / checkpoint
    model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"))
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt"))
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


BATCH_SIZE = 6
SEQ_LENGTH = 1024


class BigTrainDataset(Dataset):

    def __init__(self, all_tokens):
        self.xs = all_tokens[:-1].reshape(-1, BATCH_SIZE, SEQ_LENGTH)
        self.ys = all_tokens[1:].reshape(-1, BATCH_SIZE, SEQ_LENGTH)

    def __getitem__(self, ix):
        return (self.xs[ix], self.ys[ix])

    def __len__(self):
        return self.xs.shape[0]


def load_dataset(split):
    return BigTrainDataset(
        load_file(DATASETS_DIR / f"{split}.safetensors")["tokens"]
    )


def calculate_loss(logits, targets):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten()
    )


MINS_BETWEEN_VAL_AND_CHECKPOINT = 30
EXPECTED_ITERATIONS_PER_SEC = 3.29
VAL_AND_CHECKPOINT_INTERVAL = int(
    MINS_BETWEEN_VAL_AND_CHECKPOINT * 60 * EXPECTED_ITERATIONS_PER_SEC
)


def get_training_data():
    train_losses = []
    val_losses = []
    best_train_ds_offset = None
    for item in CHECKPOINTS_DIR.iterdir():
        if item.name == "latest":
            continue

        meta = json.loads((item / "meta.json").read_text())
        if item.name == "best":
            best_train_ds_offset = meta["train_ds_offset"]
            continue

        train_losses.append((meta["train_ds_offset"], meta["train_loss"]))
        val_losses.append((meta["train_ds_offset"], meta["val_loss"]))

    train_losses.sort(key=lambda x: x[0])
    val_losses.sort(key=lambda x: x[0])

    return train_losses, val_losses, best_train_ds_offset


def generate_training_chart():
    train_points, val_points, best_train_ds_offset = get_training_data()

    plt.title("TRAINING RUN LOSS")
    plt.xkcd()
    plt.rcParams['font.family'] = "xkcd"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    train_epochs, train_losses = zip(*train_points)
    val_epochs, val_losses = zip(*val_points)
    ax.plot(train_epochs, train_losses, label="TRAINING LOSS", marker="o")
    ax.plot(val_epochs, val_losses, label="VALIDATION LOSS", marker="s")

    ax.axvline(
        best_train_ds_offset, color="red", linestyle="--", linewidth=1.5,
        label="BEST ITERATION"
    )

    ax.set_title("TRAINING RUN LOSS")
    ax.set_xlabel("ITERATION")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("LOSS")
    ax.legend()

    fig.tight_layout()
    image_file = MY_DIR / "big-training-run-chart.png"
    fig.savefig(image_file, bbox_inches="tight")
    plt.close(fig)


def train(
    model, optimizer, scaler,
    train_ds, val_ds,
    train_ds_offset, best_loss
):
    device = next(model.parameters()).device

    torch.set_float32_matmul_precision("high")

    print(f"Starting training at dataset offset {train_ds_offset}")
    for ix in tqdm(range(train_ds_offset, len(train_ds))):
        model.train()
        inputs, targets = train_ds[ix]
        inputs = inputs.to(device).to(torch.long)
        targets = targets.to(device).to(torch.long)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(inputs)

            train_loss = calculate_loss(logits, targets)

        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ix % VAL_AND_CHECKPOINT_INTERVAL == 0:
            print("Validation/checkpoint")
            model.eval()
            with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                val_losses = []
                for val_inputs, val_targets in tqdm(val_ds):
                    val_inputs = val_inputs.to(device).to(torch.long)
                    val_targets = val_targets.to(device).to(torch.long)
                    val_logits = model(val_inputs)
                    val_losses.append(
                        calculate_loss(val_logits, val_targets).item()
                    )
                val_loss = sum(val_losses) / len(val_losses)

            if best_loss is None or val_loss < best_loss:
                is_best = True
                best_loss = val_loss
            else:
                is_best = False

            save_checkpoint(
                f"iteration-{ix}",
                model, optimizer, scaler,
                train_loss.item(), val_loss,
                ix,
                is_best
            )
            generate_training_chart()

            model.train()
            print("Continuing training")


@click.command()
@click.argument("checkpoint", default=None)
def main(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    big_train_params = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    model = GPTModel(big_train_params)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )

    scaler = torch.amp.GradScaler()

    train_ds = load_dataset("train")
    val_ds = load_dataset("validation")

    if checkpoint:
        train_ds_offset, best_loss = load_checkpoint(
            checkpoint, model, optimizer, scaler
        )
    else:
        train_ds_offset = 0
        best_loss = None

    train(
        model, optimizer, scaler,
        train_ds, val_ds,
        train_ds_offset, best_loss
    )


if __name__ == "__main__":
    main()
