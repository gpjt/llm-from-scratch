import click
from tqdm import tqdm

import torch

from big_train import calculate_loss, load_dataset
from checkpointing import load_checkpoint
from gpt import GPTModel


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
    load_checkpoint("best", model)
    model.to(device)

    val_ds = load_dataset("validation")

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

    print(f"Loss against our validation dataset: {val_loss}")


if __name__ == "__main__":
    main()
