import click

import torch

from gpt import GPTModel


def load_checkpoint():
    ...


def save_checkpoint():
    ...


def train(model):
    ...


@click.command
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

    if checkpoint:
        load_checkpoint(checkpoint, model, optimizer, scaler)

    train(model)


if __name__ == "__main__":
    main()
