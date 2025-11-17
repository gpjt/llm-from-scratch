import click

import tiktoken
import torch

from checkpointing import load_checkpoint
from generate_text import generate
from gpt import GPTModel
from model_config import GPT_CONFIG_124M
from second_generation_test import text_to_token_ids, token_ids_to_text


@click.command()
@click.argument("checkpoint", default=None)
def main(checkpoint):
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
    model.eval()

    if checkpoint:
        _, __ = load_checkpoint(checkpoint, model)

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
