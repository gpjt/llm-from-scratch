import json

import click

from safetensors.torch import save_file

from download_and_use_gpt2 import load_weights_into_gpt
from gpt import GPTModel
from gpt_download import download_and_load_gpt2


@click.command()
@click.argument("model_size")
@click.argument("model_config_path")
@click.argument("safetensors_path")
def main(model_size, model_config_path, safetensors_path):
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    model = GPTModel(model_config)

    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )
    load_weights_into_gpt(model, params)

    save_file(model.state_dict(), safetensors_path)


if __name__ == "__main__":
    main()
