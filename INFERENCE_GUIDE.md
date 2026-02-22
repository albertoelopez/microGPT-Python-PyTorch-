# Model Inference Guide

## ✅ **What's New:**

The PyTorch dashboard now **automatically saves** your trained model when training completes!

---

## 💾 **Model Saving**

After training finishes, the model is saved to: **`model_pytorch.pt`**

This file contains:
- Model weights
- Architecture configuration (layers, embedding dim, etc.)
- Vocabulary (character mappings)
- Final training loss

---

## 🚀 **How to Use the Trained Model**

### **Option 1: Quick Generation** (Default)

Generate 20 names with default settings:
```bash
python inference.py
```

### **Option 2: Custom Count & Temperature**

Generate N names with custom temperature:
```bash
python inference.py 50 0.8
```
- First number: how many names (e.g., 50)
- Second number: temperature 0.1-2.0 (e.g., 0.8)

### **Option 3: Interactive Mode** (Recommended!)

```bash
python inference.py --interactive
```

Then use these commands:
```
generate       - Generate 20 names
generate 50    - Generate 50 names
temp 0.8       - Set temperature to 0.8 and generate 10 samples
help           - Show commands
q              - Quit
```

---

## 🌡️ **Temperature Guide**

Try different temperatures to see what works best:

```bash
python inference.py 10 0.3    # Conservative (safe, common names)
python inference.py 10 0.5    # Balanced
python inference.py 10 0.7    # Creative (default)
python inference.py 10 1.0    # Very creative
python inference.py 10 1.2    # Experimental (unique/weird)
```

---

## 📊 **Example Workflow**

1. **Train on dashboard**:
   - Visit: http://localhost:5001
   - Set config (50k steps, n_embd=256, etc.)
   - Click "Start Training"
   - Wait for completion
   - Model auto-saves to `model_pytorch.pt`

2. **Test the model**:
   ```bash
   python inference.py --interactive
   ```

3. **Generate names**:
   ```
   generate 20
   temp 0.8
   generate 50
   ```

4. **Try different temperatures** to find your favorite style!

---

## 🔧 **Troubleshooting**

### **"FileNotFoundError: model_pytorch.pt"**
- You need to train a model first!
- Go to http://localhost:5001 and complete a training run
- The model will be saved automatically

### **"CUDA out of memory"**
- The inference uses way less memory than training
- Should work fine even on CPU
- If needed, inference will fall back to CPU automatically

### **"Generated names are weird"**
- Try lower temperature (0.5-0.6)
- Train longer (more steps = better quality)
- Check your final training loss (should be < 1.5)

---

## 📈 **Expected Quality by Loss**

| Final Loss | Quality | Examples |
|-----------|---------|----------|
| > 2.5 | Poor | Random letters |
| 2.0 - 2.5 | Learning | Some name-like patterns |
| 1.5 - 2.0 | Decent | Recognizable as names |
| 1.0 - 1.5 | Good | Realistic names |
| 0.5 - 1.0 | Great | High-quality names |
| < 0.5 | Excellent | Professional quality |

---

## 💡 **Pro Tips**

1. **Train with RTX 4090 settings** (50k+ steps, n_embd=256)
2. **Wait for loss < 1.0** before stopping
3. **Try temp=0.7** first, adjust from there
4. **Save multiple models** by renaming:
   ```bash
   mv model_pytorch.pt model_50k_loss0.8.pt
   ```
5. **Compare models** by training with different configs

---

Enjoy generating names! 🎉
