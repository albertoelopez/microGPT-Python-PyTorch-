import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

checkpoint = torch.load('model_pytorch.pt', map_location=device)

vocab_size = checkpoint['vocab_size']
n_layer = checkpoint['n_layer']
n_embd = checkpoint['n_embd']
n_head = checkpoint['n_head']
block_size = checkpoint['block_size']
uchars = checkpoint['uchars']
BOS = checkpoint['BOS']

print(f"\nLoaded model:")
print(f"  Layers: {n_layer}")
print(f"  Embedding dim: {n_embd}")
print(f"  Attention heads: {n_head}")
print(f"  Context length: {block_size}")
print(f"  Vocab size: {vocab_size}")
print(f"  Final training loss: {checkpoint.get('final_loss', 'N/A')}")

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, block_size):
        super().__init__()
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.wpe(pos)

        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = GPT(vocab_size, n_layer, n_embd, n_head, block_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\n{'='*50}")
print("Model loaded successfully! Ready to generate.")
print(f"{'='*50}\n")

def generate_names(count=20, temperature=0.7):
    print(f"Generating {count} names (temperature={temperature}):\n")
    for i in range(count):
        idx = torch.tensor([[BOS]], dtype=torch.long, device=device)
        generated = model.generate(idx, max_new_tokens=block_size, temperature=temperature)
        sample = generated[0].tolist()[1:]

        if BOS in sample:
            sample = sample[:sample.index(BOS)]

        name = ''.join([uchars[i] for i in sample if i < len(uchars)])
        print(f"{i+1:2d}. {name}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            print("Interactive mode! Type 'q' to quit.\n")
            while True:
                try:
                    cmd = input("\nEnter command (generate/temp/help/q): ").strip().lower()

                    if cmd == 'q' or cmd == 'quit':
                        print("Goodbye!")
                        break
                    elif cmd == 'help':
                        print("\nCommands:")
                        print("  generate [count] - Generate N names (default: 20)")
                        print("  temp [value] - Set temperature (0.1-2.0)")
                        print("  q - Quit")
                    elif cmd.startswith('generate'):
                        parts = cmd.split()
                        count = int(parts[1]) if len(parts) > 1 else 20
                        generate_names(count)
                    elif cmd.startswith('temp'):
                        parts = cmd.split()
                        if len(parts) > 1:
                            temp = float(parts[1])
                            print(f"Temperature set to {temp}")
                            generate_names(10, temp)
                        else:
                            print("Usage: temp [value]")
                    else:
                        generate_names(20)
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        else:
            count = int(sys.argv[1])
            temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
            generate_names(count, temp)
    else:
        generate_names(20, 0.7)
