import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

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
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cache=None):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)
            cache['k'] = k
            cache['v'] = v

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if cache is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.wo(out)

        return out, attn

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

    def forward(self, x, cache=None):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, cache)
        x = x + attn_out

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x, attn_weights

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.08)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.wpe(pos)

        x = tok_emb + pos_emb

        attn_weights_list = []
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache else None
            x, attn_weights = block(x, block_cache)
            attn_weights_list.append(attn_weights)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, attn_weights_list

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        self.train()
        return idx

n_layer = 2
n_embd = 32
block_size = 16
n_head = 4

model = GPT(vocab_size, n_layer, n_embd, n_head, block_size).to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

optimizer = AdamW(model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-8)

num_steps = 5000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

    n = min(block_size, len(tokens) - 1)
    x = torch.tensor(tokens[:n], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(tokens[1:n+1], dtype=torch.long, device=device).unsqueeze(0)

    logits, loss, attn_weights = model(x, targets=y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step:4d} / {num_steps:4d} | loss {loss.item():.4f}")

    if step % 500 == 0 and step > 0:
        model.eval()
        idx = torch.tensor([[BOS]], dtype=torch.long, device=device)
        generated = model.generate(idx, max_new_tokens=block_size, temperature=0.7)
        sample = generated[0].tolist()[1:]
        if BOS in sample:
            sample = sample[:sample.index(BOS)]
        sample_text = ''.join([uchars[i] for i in sample if i < len(uchars)])
        print(f"  sample: {sample_text}")
        model.train()

print("\n--- inference (new, hallucinated names) ---")
model.eval()
for sample_idx in range(20):
    idx = torch.tensor([[BOS]], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=block_size, temperature=0.7)
    sample = generated[0].tolist()[1:]
    if BOS in sample:
        sample = sample[:sample.index(BOS)]
    sample_text = ''.join([uchars[i] for i in sample if i < len(uchars)])
    print(f"sample {sample_idx+1:2d}: {sample_text}")

torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'n_layer': n_layer,
    'n_embd': n_embd,
    'n_head': n_head,
    'block_size': block_size,
    'uchars': uchars,
    'BOS': BOS
}, 'model_pytorch.pt')
print("\nModel saved to model_pytorch.pt")
