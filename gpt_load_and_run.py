import torch
import tiktoken
from gpt import GPTModel, generate_text_simple

# 1. Setup configuration (MUST match what you used in gpt_train.py)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize model and load weights
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# 3. Test it!
tokenizer = tiktoken.get_encoding("gpt2")
prompt = "Who is Victor Grumble?" # Change this to anything!
encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

out_ids = generate_text_simple(
    model=model, 
    idx=encoded, 
    max_new_tokens=30, 
    context_size=GPT_CONFIG_124M["context_length"]
)

print("\n--- Model Output ---")
print(tokenizer.decode(out_ids.squeeze(0).tolist()))