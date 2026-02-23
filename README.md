# Training Large Models Without Going Broke

So I kept running into "CUDA out of memory" errors trying to train a 7B model on my GPU. Spent a weekend figuring out how to actually make it work without paying for expensive cloud GPUs. This repo has all the code and benchmarks.

## The Problem

My GPU has 16GB. Training a large model needs like 100GB+. Most tutorials assume you have access to fancy hardware or unlimited cloud credits. As a grad student... yeah, no.

## What Actually Worked

I tested a bunch of memory optimization techniques and combined the ones that worked best:

- **Mixed Precision (FP16)**: Cut memory in half, actually made training faster
- **Gradient Checkpointing**: Another 30-50% memory savings
- **8-bit Optimizers**: Reduced optimizer memory by 75%

Combined these to go from **8.2GB → 2.6GB** (3.2× reduction) with basically no accuracy loss.

## Results

Tested on ResNet-50 with CIFAR-100:

| What I Used | Memory | Speed | Accuracy |
|------------|--------|-------|----------|
| Baseline (FP32) | 8.2 GB | 100% | 78.5% |
| + FP16 | 4.5 GB | 122% (faster!) | 78.4% |
| + Gradient Checkpointing | 3.1 GB | 114% | 78.4% |
| + 8-bit Optimizer | 2.6 GB | 109% | 78.3% |

Lost only 0.2% accuracy. Totally worth it.

## Running the Code

### What You Need

- Python 3.8 or newer
- A GPU with CUDA (for meaningful results)
- About 10GB disk space for the dataset

### Setup

```bash
git clone https://github.com/Shellsss29/memory-efficient-training.git
cd memory-efficient-training
pip install -r requirements.txt
```

### Installing bitsandbytes (for 8-bit optimizer)

**Linux/WSL**:
```bash
pip install bitsandbytes
```

**Windows**:
```bash
pip install bitsandbytes-windows
```

**Mac**: Not supported, the benchmark will skip this part

### Run the Benchmark

```bash
python benchmark.py
```

This will:
1. Download CIFAR-100 (~170MB)
2. Run 4 different training configurations
3. Train for 3 epochs each
4. Save a comparison chart as `benchmark_results.png`

Takes about 15-30 minutes depending on your GPU.

## How It Works

### Mixed Precision Training

Instead of 32-bit floats everywhere, use 16-bit for most operations:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():  # This runs in FP16
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Checkpointing

Trade computation for memory - recompute activations during backprop instead of storing them:

```python
import torch.utils.checkpoint as checkpoint

# In your model's forward pass:
x = checkpoint.checkpoint(self.expensive_layer, x)
```

### 8-bit Optimizer

Quantize Adam optimizer states to 8-bit instead of 32-bit:

```python
import bitsandbytes as bnb

# Instead of: optimizer = torch.optim.Adam(...)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
```

## Common Issues

**Still getting OOM errors?**
- Try reducing batch size in `benchmark.py` (change `batch_size=128` to `64`)
- Close other stuff using your GPU

**bitsandbytes not installing?**
- Benchmark will skip 8-bit tests automatically
- You'll still get FP16 + checkpointing results

**Running slow on CPU?**
- This needs a GPU to be useful
- Memory measurements won't make sense on CPU

## Using This in Your Projects

Here's the minimal code to add to your training loop:

```python
from torch.cuda.amp import autocast, GradScaler
import bitsandbytes as bnb

# Setup
scaler = GradScaler()
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Check `blog_post.md` for the full write-up with more details and explanations.

## Project Files

```
├── blog_post.md          - Full explanation of everything
├── benchmark.py          - Script that runs all the tests
├── requirements.txt      - Python packages you need
└── benchmark_results.png - Generated charts (after running)
```

## Useful Resources

- [PyTorch AMP Docs](https://pytorch.org/docs/stable/amp.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

## Questions?

Read the full blog post in `blog_post.md` or feel free to open an issue.

---

**Shailly Bhati** | [Portfolio](https://shellsss29.github.io) | [LinkedIn](https://linkedin.com/in/shailly-bhati) | [GitHub](https://github.com/Shellsss29)
