import json
import os
import urllib.request

import torch
from torch.utils.data import Dataset


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    data = json.loads(text_data)
    return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that described a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, ix):
        return self.encoded_texts[ix]

    def __len__(self):
        return len(self.encoded_texts)


def custom_collate_draft_1(
    batch, pad_token_id=50256, device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


def custom_collate_draft_2(
    batch, pad_token_id=50256, device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def main():
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    print("Example entry:\n", data[50])

    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)

    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"
    print(model_input + desired_response)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Test set length:", len(test_data))
    print("Validation set length:", len(val_data))

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (
        inputs_1,
        inputs_2,
        inputs_3,
    )
    print(custom_collate_draft_1(batch))

    inputs, targets = custom_collate_draft_2(batch)
    print(inputs)
    print(targets)







if __name__ == "__main__":
    main()
