import torch

from gpt import GPTModel
from model_config import GPT_CONFIG_124M


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

batch = torch.tensor(
    [
        [6109, 3626, 6100, 345],
        [6109, 1110, 6622, 257],
    ],
    dtype=torch.long
)
print("Input batch:\n", batch)

out = model(batch)
print("\nOutput shape:\n", out.shape)
print(out)


total_params = sum(p.numel() for p in model.parameters())
print(f"Total nuumber of parameters: {total_params:,}")

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Parameters if we were using weight tying: {total_params_gpt2:,}")
