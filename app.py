from flask import Flask, render_template, Response, jsonify, request
import json
import time
import os
import math
import random
random.seed(42)

app = Flask(__name__)

training_active = False
training_metrics = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training')
def start_training():
    global training_active, training_metrics
    training_active = True
    training_metrics = []
    return jsonify({'status': 'started'})

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
        global training_active, training_metrics

        if not os.path.exists('input.txt'):
            import urllib.request
            names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
            urllib.request.urlretrieve(names_url, 'input.txt')

        docs = [line.strip() for line in open('input.txt') if line.strip()]
        random.shuffle(docs)

        uchars = sorted(set(''.join(docs)))
        BOS = len(uchars)
        vocab_size = len(uchars) + 1

        yield f"data: {json.dumps({'type': 'init', 'vocab_size': vocab_size, 'num_docs': len(docs), 'chars': uchars})}\n\n"

        class Value:
            __slots__ = ('data', 'grad', '_children', '_local_grads')

            def __init__(self, data, children=(), local_grads=()):
                self.data = data
                self.grad = 0
                self._children = children
                self._local_grads = local_grads

            def __add__(self, other):
                other = other if isinstance(other, Value) else Value(other)
                return Value(self.data + other.data, (self, other), (1, 1))

            def __mul__(self, other):
                other = other if isinstance(other, Value) else Value(other)
                return Value(self.data * other.data, (self, other), (other.data, self.data))

            def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
            def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
            def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
            def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
            def __neg__(self): return self * -1
            def __radd__(self, other): return self + other
            def __sub__(self, other): return self + (-other)
            def __rsub__(self, other): return other + (-self)
            def __rmul__(self, other): return self * other
            def __truediv__(self, other): return self * other**-1
            def __rtruediv__(self, other): return other * self**-1

            def backward(self):
                topo = []
                visited = set()
                def build_topo(v):
                    if v not in visited:
                        visited.add(v)
                        for child in v._children:
                            build_topo(child)
                        topo.append(v)
                build_topo(self)
                self.grad = 1
                for v in reversed(topo):
                    for child, local_grad in zip(v._children, v._local_grads):
                        child.grad += local_grad * v.grad

        def linear(x, w):
            return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

        def softmax(logits):
            max_val = max(val.data for val in logits)
            exps = [(val - max_val).exp() for val in logits]
            total = sum(exps)
            return [e / total for e in exps]

        def rmsnorm(x):
            ms = sum(xi * xi for xi in x) / len(x)
            scale = (ms + 1e-5) ** -0.5
            return [xi * scale for xi in x]

        head_dim = n_embd // n_head

        matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
        state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
        for i in range(n_layer):
            state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
            state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
        params = [p for mat in state_dict.values() for row in mat for p in row]

        def gpt(token_id, pos_id, keys, values, return_attn=False):
            tok_emb = state_dict['wte'][token_id]
            pos_emb = state_dict['wpe'][pos_id]
            x = [t + p for t, p in zip(tok_emb, pos_emb)]
            x = rmsnorm(x)

            attn_weights_all = []
            for li in range(n_layer):
                x_residual = x
                x = rmsnorm(x)
                q = linear(x, state_dict[f'layer{li}.attn_wq'])
                k = linear(x, state_dict[f'layer{li}.attn_wk'])
                v = linear(x, state_dict[f'layer{li}.attn_wv'])
                keys[li].append(k)
                values[li].append(v)
                x_attn = []

                for h in range(n_head):
                    hs = h * head_dim
                    q_h = q[hs:hs+head_dim]
                    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
                    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
                    attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
                    attn_weights = softmax(attn_logits)
                    if return_attn and h == 0:
                        attn_weights_all.append([w.data for w in attn_weights])
                    head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
                    x_attn.extend(head_out)

                x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
                x = [a + b for a, b in zip(x, x_residual)]
                x_residual = x
                x = rmsnorm(x)
                x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
                x = [xi.relu() for xi in x]
                x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
                x = [a + b for a, b in zip(x, x_residual)]

            logits = linear(x, state_dict['lm_head'])
            if return_attn:
                return logits, attn_weights_all
            return logits

        beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
        m = [0.0] * len(params)
        v = [0.0] * len(params)

        for step in range(num_steps):
            doc = docs[step % len(docs)]
            tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
            n = min(block_size, len(tokens) - 1)

            keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
            losses = []
            attn_weights_step = None

            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                if pos_id == n - 1 and step % 100 == 0:
                    logits, attn_weights_step = gpt(token_id, pos_id, keys_cache, values_cache, return_attn=True)
                else:
                    logits = gpt(token_id, pos_id, keys_cache, values_cache, return_attn=False)
                probs = softmax(logits)
                loss_t = -probs[target_id].log()
                losses.append(loss_t)

            loss = (1 / n) * sum(losses)
            loss.backward()

            lr_t = learning_rate * (1 - step / num_steps)
            grad_norm = sum(p.grad ** 2 for p in params) ** 0.5

            for i, p in enumerate(params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
                m_hat = m[i] / (1 - beta1 ** (step + 1))
                v_hat = v[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
                p.grad = 0

            metrics = {
                'type': 'step',
                'step': step,
                'loss': loss.data,
                'lr': lr_t,
                'grad_norm': grad_norm,
                'doc': doc
            }

            if step % 50 == 0:
                keys_gen, values_gen = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
                token_id = BOS
                sample = []
                for pos_id in range(block_size):
                    logits = gpt(token_id, pos_id, keys_gen, values_gen, return_attn=False)
                    probs = softmax([l / temperature for l in logits])
                    token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
                    if token_id == BOS:
                        break
                    sample.append(uchars[token_id])
                metrics['sample'] = ''.join(sample)

            if attn_weights_step and step % 100 == 0:
                metrics['attention'] = attn_weights_step[:min(8, len(attn_weights_step))]

            yield f"data: {json.dumps(metrics)}\n\n"
            time.sleep(0.01)

        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
