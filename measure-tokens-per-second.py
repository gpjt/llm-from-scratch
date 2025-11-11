import time

import tiktoken
import torch
from datasets import load_dataset
from tqdm import tqdm

from gpt import GPTModel

NUM_BATCHES = 100
MAX_BATCH_SIZE = 10
SEQ_LENGTH = 1024


def main():
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

    ds = load_dataset(
        "parquet",
        data_files="./fineweb/sample/10BT/*.parquet",
        split="train"
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    all_tokens = []
    train_tokens = (NUM_BATCHES * MAX_BATCH_SIZE * SEQ_LENGTH) + 1
    for element in ds:
        text = element["text"]
        tokens = tokenizer.encode(text)
        all_tokens += tokens
        if len(all_tokens) >= train_tokens:
            break
    all_tokens = all_tokens[:train_tokens]

    torch.set_float32_matmul_precision("high")
    scaler = torch.amp.GradScaler()

    for batch_size in range(1, MAX_BATCH_SIZE + 1):
        print(f"Testing with batch size {batch_size}")
        torch.manual_seed(42)
        model = GPTModel(big_train_params)
        model.to(device)
        model.train()

        batches = []
        input_token_count = 0
        for batch in range(NUM_BATCHES):
            start = batch * batch_size * SEQ_LENGTH
            end = start + batch_size * SEQ_LENGTH
            inputs = all_tokens[start:end]
            outputs = all_tokens[start + 1:end + 1]
            input_tensor = torch.tensor(inputs).reshape(batch_size, SEQ_LENGTH)
            output_tensor = torch.tensor(outputs).reshape(batch_size, SEQ_LENGTH)
            batches.append((input_tensor, output_tensor))
            input_token_count += batch_size * SEQ_LENGTH

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0004, weight_decay=0.1
        )

        start = time.time()
        for inputs, outputs in tqdm(batches):
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), outputs.flatten()
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        end = time.time()

        seconds = end - start

        print(f"Done, trained on {input_token_count:,} tokens in {seconds:.4f}s.")
        print(f"Tokens per second: {int(input_token_count / seconds):,}\n")


if __name__ == "__main__":
    main()
