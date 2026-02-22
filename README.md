<div align="center">
  <img src="docs/images/hero-banner.svg" alt="microGPT - Educational GPT Implementation" width="100%"/>
</div>

<br/>

# microGPT - Educational GPT Implementation

Two complete implementations of GPT for learning and comparison, organized in separate folders:

## 📁 Repository Structure

```
microGPT/
├── python-cpu/          # Pure Python implementation (CPU-only)
│   ├── microgpt.py      # Core implementation
│   ├── app.py           # Flask dashboard
│   └── templates/       # Web UI
│       └── index.html
│
├── pytorch-gpu/         # PyTorch implementation (GPU-accelerated)
│   ├── microgpt_pytorch.py  # Core implementation
│   ├── app_pytorch.py       # Flask dashboard
│   ├── inference.py         # Model inference script
│   ├── INFERENCE_GUIDE.md   # Inference documentation
│   └── templates/           # Web UI
│       └── index_pytorch.html
│
├── input.txt            # Shared training data
├── requirements.txt     # Dependencies
└── README.md           # This file
```

---

## 🐍 Pure Python Version (CPU-Only)
**Educational focus:** Understand backpropagation and autograd from scratch

### Features:
- ✅ Zero dependencies (only Python stdlib)
- ✅ Custom autograd engine (`Value` class)
- ✅ Every operation visible and understandable
- ✅ Perfect for learning fundamentals

### Run:
```bash
cd python-cpu
python microgpt.py
```

### Web Dashboard:
```bash
cd python-cpu
python app.py
# Visit: http://localhost:5000
```

---

## ⚡ PyTorch Version (GPU-Accelerated)
**Practical focus:** Industry-standard transformer training

### Features:
- ✅ 100-1000x faster (GPU support)
- ✅ Professional PyTorch code patterns
- ✅ Automatic differentiation via PyTorch
- ✅ Efficient matrix operations
- ✅ Model saving/loading
- ✅ Built-in inference UI

### Run:
```bash
cd pytorch-gpu
python microgpt_pytorch.py
```

### Web Dashboard:
```bash
cd pytorch-gpu
python app_pytorch.py
# Visit: http://localhost:5001
```

---

## 📊 Side-by-Side Comparison

| Feature | Pure Python | PyTorch |
|---------|-------------|---------|
| **Speed** | Slow (CPU only) | Fast (GPU/CPU) |
| **Dependencies** | None | PyTorch |
| **Learning Value** | Fundamentals | Best Practices |
| **Code Length** | ~200 lines | ~200 lines |
| **Autograd** | Custom built | PyTorch autograd |
| **Training Steps** | 1000-5000 | 5000-50000+ |
| **Model Size** | Tiny (16-32 dim) | Flexible (32-256 dim) |

---

## 🎯 Which Should You Use?

### Use **Pure Python** if you want to:
- Understand how backpropagation works
- Learn autograd from first principles
- See every gradient computation step
- Build intuition for neural networks

### Use **PyTorch** if you want to:
- Train models quickly on GPU
- Learn industry-standard practices
- Build production-ready models
- Experiment with larger architectures

### Use **Both** if you want to:
- Complete understanding (recommended!)
- Compare implementations side-by-side
- See what PyTorch abstracts away

---

## 🚀 Installation

### Pure Python Version:
```bash
pip install Flask==3.0.0
```

### PyTorch Version:
```bash
pip install Flask==3.0.0 torch>=2.0.0
```

Or install everything:
```bash
pip install -r requirements.txt
```

---

## 📖 Training Configuration

Both dashboards support adjustable hyperparameters:

- **Training Steps**: How many optimization steps
- **Number of Layers**: Transformer depth
- **Embedding Dimension**: Model width
- **Attention Heads**: Multi-head attention count
- **Context Length**: Maximum sequence length
- **Learning Rate**: Optimization step size
- **Temperature**: Generation creativity

---

## 🎨 Visualizations

Both dashboards include:
- Real-time loss curves
- Generated samples during training
- Attention heatmaps
- Optimizer statistics
- Training speed metrics (PyTorch only)
- GPU detection (PyTorch only)

---

## 💾 Model Inference

### Pure Python:
Models are not saved by default. You can add saving logic to `microgpt.py`.

### PyTorch:
Models are automatically saved to `model_pytorch.pt` after training.

Load and use:
```python
import torch

checkpoint = torch.load('model_pytorch.pt')
# Model ready for inference!
```

---

## 🎓 Learning Path

1. **Start with Pure Python** - Read and understand `microgpt.py`
2. **Run the dashboard** - Watch training in real-time
3. **Study PyTorch version** - Compare to pure Python
4. **Experiment** - Adjust hyperparameters, see what happens
5. **Build your own** - Create variations and improvements

---

## 📝 Credits

Based on Andrej Karpathy's microGPT (@karpathy)

PyTorch version adds:
- GPU acceleration
- Modern transformer architecture
- Web-based training visualization
- Professional code patterns

---

## 🐛 Troubleshooting

### Pure Python is too slow
- It's supposed to be! Use PyTorch for speed
- Keep models tiny (n_embd=16-32, n_layer=1-2)
- Reduce training steps to 1000-2000

### PyTorch GPU not working
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/

### Dashboards not loading
- Check ports 5000 and 5001 are available
- Try: `lsof -i :5000` to see what's using the port
- Kill old Flask processes if needed

---

Enjoy learning! 🚀
