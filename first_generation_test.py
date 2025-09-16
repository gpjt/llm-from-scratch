import tiktoken
import torch

from generate_text_simple import generate_text_simple
from gpt import GPTModel
from model_config import GPT_CONFIG_124M


torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor shape:", encoded_tensor.shape)

model.eval()
out = generate_text_simple(model, encoded_tensor, 6, GPT_CONFIG_124M["context_length"])
print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print("Decoded text:", decoded_text)
