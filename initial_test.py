import torch

from gpt import GPTModel


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


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
