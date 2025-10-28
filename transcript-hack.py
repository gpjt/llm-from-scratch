import tiktoken

from download_and_use_gpt2 import load_weights_into_gpt
from generate_text_simple import generate_text_simple
from gpt import GPTModel
from gpt_download import download_and_load_gpt2
from model_config import model_configs
from second_generation_test import text_to_token_ids, token_ids_to_text


def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )

    print(f"Creating model with drop_rate {BASE_CONFIG['drop_rate']}")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    input_text = (
        "This is a transcript of a conversation between a helpful bot, 'Bot', "
        "and a human, 'User'.  The bot is very intelligent and always answers "
        "the human's questions with a useful reply.\n\n"
        "Human: Provide a synonym for 'bright'\n\n"
        "Bot: "
    )
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
