import math
import shutil
from pathlib import Path

import tiktoken
import torch
from datasets import load_dataset
from safetensors.torch import save_file


BATCH_TOKENS = 6 * 1024

VAL_BATCHES = 3200
VAL_TOKEN_COUNT = VAL_BATCHES * BATCH_TOKENS + 1

MODEL_PARAMS = 163_009_536
CHINCHILLA_OPTIMUM_TOKENS = MODEL_PARAMS * 20
TRAIN_TOKEN_COUNT = math.ceil(CHINCHILLA_OPTIMUM_TOKENS / BATCH_TOKENS) * BATCH_TOKENS + 1


def build_and_save_dataset_tensor(ds, tokenizer, token_count, path):
    results = []
    num_tokens = 0
    batch_size = 1000
    for ix in range(0, len(ds), batch_size):
        print(f"{num_tokens:,}/{token_count:,}")
        texts = ds[ix:ix + batch_size]["text"]
        text_tokens = tokenizer.encode_batch(texts)
        all_tokens = []
        for toks in text_tokens:
            all_tokens.extend(toks)
            all_tokens.append(tokenizer.eot_token)
        results.append(torch.tensor(all_tokens, dtype=torch.int32))
        num_tokens += len(all_tokens)
        if num_tokens > token_count:
            break
    result = torch.cat(results)
    result = result[:token_count]
    print(f"Saving {result.shape[0]} tokens to {path}")
    save_file({"tokens": result}, path)


def main():
    splits = load_dataset(
        "parquet",
        data_files="./fineweb/sample/10BT/*.parquet",
        split={"train": "train[:99%]", "validation": "train[99%:]"}
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset_dir = Path(__file__).resolve().parent / "big-train-datasets"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir()

    build_and_save_dataset_tensor(
        splits["validation"], tokenizer, VAL_TOKEN_COUNT,
        dataset_dir / "validation.safetensors"
    )



if __name__ == "__main__":
    main()
