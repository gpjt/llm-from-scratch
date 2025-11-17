import click
from tqdm import tqdm

import torch

from big_train import calculate_loss, load_dataset
from download_and_use_gpt2 import load_weights_into_gpt
from gpt import GPTModel
from gpt_download import download_and_load_gpt2
from model_config import model_configs


@click.command()
@click.argument("checkpoint", default=None)
def main(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
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
