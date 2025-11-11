import time
from pathlib import Path
from shutil import rmtree

import tiktoken
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm

from gpt import GPTModel

NUM_BATCHES = 100
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
    train_tokens = (NUM_BATCHES * 2 * BATCH_SIZE * SEQ_LENGTH) + 1
    for element in ds:
        text = element["text"]
        tokens = tokenizer.encode(text)
        all_tokens += tokens
        if len(all_tokens) >= train_tokens:
            break
    all_tokens = all_tokens[:train_tokens]

    scaler = torch.amp.GradScaler()

    torch.manual_seed(42)
    model = GPTModel(big_train_params)
    model.to(device)
    model.train()

    batches = []
    for batch in range(2 * NUM_BATCHES):
        start = batch * BATCH_SIZE * SEQ_LENGTH
        end = start + BATCH_SIZE * SEQ_LENGTH
        inputs = all_tokens[start:end]
        outputs = all_tokens[start + 1:end + 1]
        input_tensor = torch.tensor(inputs).reshape(BATCH_SIZE, SEQ_LENGTH)
        output_tensor = torch.tensor(outputs).reshape(BATCH_SIZE, SEQ_LENGTH)
        batches.append((input_tensor, output_tensor))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )

    for inputs, outputs in tqdm(batches[:NUM_BATCHES]):
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

    start = time.time()
    checkpoint_dir = Path(__file__).resolve().parent / "tmp-test-checkpoint"
    if checkpoint_dir.exists():
        rmtree(checkpoint_dir)
    checkpoint_dir.mkdir()
    save_file(model.state_dict(), checkpoint_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scaler.state_dict(), checkpoint_dir / "scaler.pt")
    end = time.time()

    print(f"Checkpoint saved in {end - start:.2f}s")




if __name__ == "__main__":
    main()
