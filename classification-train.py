import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from download_and_use_gpt2 import load_weights_into_gpt
from generate_text_simple import generate_text_simple
from gpt import GPTModel
from gpt_download import download_and_load_gpt2
from model_config import model_configs
from second_generation_test import text_to_token_ids, token_ids_to_text


class SpamDataset(Dataset):

    def __init__(
        self, csv_file, tokenizer, max_length=None, pad_token_id=50256
    ):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]


    def __getitem__(self, ix):
        encoded = self.encoded_texts[ix]
        label = self.data.iloc[ix]["Label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


    def __len__(self):
        return len(self.data)


    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length




def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
        csv_file="classification-train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    print(train_dataset.max_length)
    val_dataset = SpamDataset(
        csv_file="classification-validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file="classification-test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    for input_batch, target_batch in train_loader:
        pass
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions:", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"
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

    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'?  Answer with 'yes' or 'no':"
        " 'You are a winner you have been specifically"
        " selected to receive $1000 cash or a $2000 reward.'"
    )
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))

    text_3 = (
        "This is a transcript of a conversation between a helpful bot, 'Bot', "
        "and a human, 'User'.  The bot is very intelligent and always answers the "
        "human's questions with a useful reply.\n\n"
        "Human: Is the following text 'spam'?  Answer with 'yes' or 'no':"
        " 'You are a winner you have been specifically"
        " selected to receive $1000 cash or a $2000 reward.'\n\n"
        "Bot: "
    )
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_3, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))







if __name__ == "__main__":
    main()
