import time

import tiktoken
import torch
from datasets import load_dataset
from tqdm import tqdm

from gpt import GPTModel

TRAIN_BATCHES = 100
MIN_VALIDATION_BATCHES = 2900
MAX_VALIDATION_BATCHES = 4000

BATCH_SIZE = 6
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
    token_count = ((TRAIN_BATCHES + MAX_VALIDATION_BATCHES) * BATCH_SIZE * SEQ_LENGTH) + 2
    for element in ds:
        text = element["text"]
        tokens = tokenizer.encode(text)
        all_tokens += tokens
        if len(all_tokens) >= token_count:
            break
    all_tokens = all_tokens[:token_count]

    torch.set_float32_matmul_precision("high")
    scaler = torch.amp.GradScaler()

    print("Doing initial train")
    torch.manual_seed(42)
    model = GPTModel(big_train_params)
    model.to(device)
    model.train()

    batches = []
    for batch in range(TRAIN_BATCHES + MAX_VALIDATION_BATCHES):
        start = batch * BATCH_SIZE * SEQ_LENGTH
        end = start + BATCH_SIZE * SEQ_LENGTH
        inputs = all_tokens[start:end]
        outputs = all_tokens[start + 1:end + 1]
        input_tensor = torch.tensor(inputs).reshape(BATCH_SIZE, SEQ_LENGTH)
        output_tensor = torch.tensor(outputs).reshape(BATCH_SIZE, SEQ_LENGTH)
        batches.append((input_tensor, output_tensor))

    train_batches = batches[:TRAIN_BATCHES]
    validation_batches = batches[TRAIN_BATCHES:]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )

    for inputs, outputs in tqdm(train_batches):
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

    model.eval()
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
        for num_validation_batches in range(MIN_VALIDATION_BATCHES, MAX_VALIDATION_BATCHES + 1, 100):
            print(f"Timing validation batch size {num_validation_batches}")
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            losses = []
            for inputs, outputs in tqdm(validation_batches[:num_validation_batches]):
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), outputs.flatten()
                )
                losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            print(f"Got loss {avg_loss:.4f} in {end - start:.4f}s")


if __name__ == "__main__":
    main()
