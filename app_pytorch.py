from flask import Flask, render_template, Response, jsonify, request
import json
import time
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import math

random.seed(42)
torch.manual_seed(42)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index_pytorch.html')

@app.route('/start_training')
def start_training():
    return jsonify({'status': 'started'})

@app.route('/check_model')
def check_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return jsonify({
        'exists': os.path.exists('model_pytorch.pt'),
        'cuda_available': torch.cuda.is_available(),
        'device': str(device)
    })

@app.route('/stream')
def stream():
    num_steps = int(request.args.get('num_steps', 5000))
    n_layer = int(request.args.get('n_layer', 2))
    n_embd = int(request.args.get('n_embd', 32))
    n_head = int(request.args.get('n_head', 4))
    block_size = int(request.args.get('block_size', 16))
    learning_rate = float(request.args.get('learning_rate', 0.01))
    temperature = float(request.args.get('temperature', 0.7))

    def generate():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists('input.txt'):
            import urllib.request
            names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
            urllib.request.urlretrieve(names_url, 'input.txt')

        docs = [line.strip() for line in open('input.txt') if line.strip()]
        random.shuffle(docs)

        uchars = sorted(set(''.join(docs)))
        BOS = len(uchars)
        vocab_size = len(uchars) + 1

        yield f"data: {json.dumps({'type': 'init', 'vocab_size': vocab_size, 'num_docs': len(docs), 'device': str(device), 'cuda_available': torch.cuda.is_available()})}\n\n"

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

            def forward(self, x, cache=None):
                B, T, C = x.shape

                q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

                scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

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

            def forward(self, x):
                x_norm = self.norm1(x)
                attn_out, attn_weights = self.attn(x_norm)
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

            def forward(self, idx, targets=None):
                B, T = idx.shape

                tok_emb = self.wte(idx)
                pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
                pos_emb = self.wpe(pos)

                x = tok_emb + pos_emb

                attn_weights_list = []
                for block in self.blocks:
                    x, attn_weights = block(x)
                    attn_weights_list.append(attn_weights)

                x = self.norm_f(x)
                logits = self.lm_head(x)

                loss = None
                if targets is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

                return logits, loss, attn_weights_list

            @torch.no_grad()
            def generate(self, idx, max_new_tokens, temperature=1.0):
                for _ in range(max_new_tokens):
                    idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
                    logits, _, _ = self(idx_cond)
                    logits = logits[:, -1, :] / temperature
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, idx_next], dim=1)
                return idx

        model = GPT(vocab_size, n_layer, n_embd, n_head, block_size).to(device)
        num_params = sum(p.numel() for p in model.parameters())

        optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.85, 0.99), eps=1e-8)

        for step in range(num_steps):
            start_time = time.time()

            doc = docs[step % len(docs)]
            tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

            n = min(block_size, len(tokens) - 1)
            x = torch.tensor(tokens[:n], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(tokens[1:n+1], dtype=torch.long, device=device).unsqueeze(0)

            logits, loss, attn_weights = model(x, targets=y)

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)).item()

            optimizer.step()

            step_time = time.time() - start_time

            metrics = {
                'type': 'step',
                'step': step,
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'grad_norm': grad_norm,
                'doc': doc,
                'step_time': step_time,
                'num_params': num_params
            }

            if step % 50 == 0:
                model.eval()
                idx = torch.tensor([[BOS]], dtype=torch.long, device=device)
                generated = model.generate(idx, max_new_tokens=block_size, temperature=temperature)
                sample = generated[0].tolist()[1:]
                if BOS in sample:
                    sample = sample[:sample.index(BOS)]
                sample_text = ''.join([uchars[i] for i in sample if i < len(uchars)])
                metrics['sample'] = sample_text
                model.train()

            if step % 100 == 0 and attn_weights:
                attn_viz = attn_weights[0][0, 0].detach().cpu().numpy().tolist()
                metrics['attention'] = attn_viz[:min(8, len(attn_viz))]

            yield f"data: {json.dumps(metrics)}\n\n"

        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': vocab_size,
            'n_layer': n_layer,
            'n_embd': n_embd,
            'n_head': n_head,
            'block_size': block_size,
            'uchars': uchars,
            'BOS': BOS,
            'temperature': temperature,
            'final_loss': loss.item()
        }, 'model_pytorch.pt')

        yield f"data: {json.dumps({'type': 'complete', 'model_saved': True, 'final_loss': loss.item()})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/generate')
def generate_names():
    try:
        count = int(request.args.get('count', 20))
        temperature = float(request.args.get('temperature', 0.7))

        if not os.path.exists('model_pytorch.pt'):
            return jsonify({'error': 'No trained model found. Please train a model first!'})

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('model_pytorch.pt', map_location=device)

        vocab_size = checkpoint['vocab_size']
        n_layer = checkpoint['n_layer']
        n_embd = checkpoint['n_embd']
        n_head = checkpoint['n_head']
        block_size = checkpoint['block_size']
        uchars = checkpoint['uchars']
        BOS = checkpoint['BOS']

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

        names = []
        for _ in range(count):
            idx = torch.tensor([[BOS]], dtype=torch.long, device=device)
            generated = model.generate(idx, max_new_tokens=block_size, temperature=temperature)
            sample = generated[0].tolist()[1:]

            if BOS in sample:
                sample = sample[:sample.index(BOS)]

            name = ''.join([uchars[i] for i in sample if i < len(uchars)])
            names.append(name)

        return jsonify({'names': names})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)
