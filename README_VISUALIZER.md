# microGPT Training Visualizer

A real-time web dashboard to visualize the training process of microGPT with beautiful charts and metrics.

## Features

- **Real-time Loss Curve**: Watch the training loss decrease over time
- **Generated Samples**: See the model generate new names as it learns
- **Optimizer Statistics**: Monitor learning rate, gradient norms, and more
- **Attention Heatmaps**: Visualize what the model pays attention to

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Click the "Start Training" button to begin training and watch the visualizations update in real-time!

## What You'll See

- **Loss Curve**: Real-time graph showing training progress
- **Optimizer Stats**: Current learning rate, gradient norm, vocabulary size, and document count
- **Generated Samples**: Every 50 steps, the model generates a new sample to show learning progress
- **Attention Heatmap**: Every 100 steps, visualize attention patterns (lighter = more attention)
- **Current Training Example**: See what document the model is currently learning from

## Architecture

- **Backend**: Flask with Server-Sent Events (SSE) for real-time streaming
- **Frontend**: Chart.js for beautiful, responsive visualizations
- **Training**: Pure Python GPT implementation (no PyTorch/TensorFlow)

## Customization

Edit `app.py` to modify training parameters:
- `num_steps`: Number of training steps (default: 1000)
- `n_layer`: Number of transformer layers (default: 1)
- `n_embd`: Embedding dimension (default: 16)
- `block_size`: Maximum context length (default: 16)
- `learning_rate`: Initial learning rate (default: 0.01)

Enjoy watching your GPT learn!
