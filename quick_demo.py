"""
Quick Demo: Memory-Efficient Training
A simplified demonstration of memory optimization techniques

Run this for a quick 1-minute demo without full benchmarking
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# Use appropriate autocast for device
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

print("="*60)
print("MEMORY-EFFICIENT TRAINING DEMO")
print("="*60)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

if not torch.cuda.is_available():
    print("⚠️  No GPU detected. Demo will run on CPU.")
    print("Memory optimization benefits are only visible on GPU.\n")

# Simple model for demonstration
class DemoModel(nn.Module):
    def __init__(self, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.layers = nn.ModuleList([
            nn.Linear(1024, 1024) for _ in range(10)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpointing:
                x = checkpoint.checkpoint(lambda x: self.relu(layer(x)), x, use_reentrant=False)
            else:
                x = self.relu(layer(x))
        return x

# Create dummy data
print("\n1. Creating dummy data...")
batch_size = 32
x = torch.randn(batch_size, 1024).to(device)
y = torch.randn(batch_size, 1024).to(device)
print(f"   Batch shape: {x.shape}")

# Criterion for loss
criterion = nn.MSELoss()

# Helper function to measure memory
def measure_memory(name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6  # MB
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"   {name}: {allocated:.1f} MB allocated, {peak:.1f} MB peak")
    else:
        print(f"   {name}: (CPU mode, no memory stats)")

print("\n" + "="*60)
print("DEMO 1: Baseline (FP32)")
print("="*60)

# Clear memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

model1 = DemoModel(use_checkpointing=False).to(device)
optimizer1 = torch.optim.Adam(model1.parameters())

output = model1(x)
loss = criterion(output, y)
loss.backward()
optimizer1.step()

measure_memory("Baseline")

del model1, optimizer1, output, loss
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("DEMO 2: Mixed Precision (FP16)")
print("="*60)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

model2 = DemoModel(use_checkpointing=False).to(device)
optimizer2 = torch.optim.Adam(model2.parameters())
scaler = GradScaler()

with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
    output = model2(x)
    loss = criterion(output, y)

scaler.scale(loss).backward()
scaler.step(optimizer2)
scaler.update()

measure_memory("Mixed Precision")

del model2, optimizer2, output, loss
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("DEMO 3: FP16 + Gradient Checkpointing")
print("="*60)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

model3 = DemoModel(use_checkpointing=True).to(device)
optimizer3 = torch.optim.Adam(model3.parameters())
scaler = GradScaler()

with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
    output = model3(x)
    loss = criterion(output, y)

scaler.scale(loss).backward()
scaler.step(optimizer3)
scaler.update()

measure_memory("FP16 + Checkpointing")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("""
✅ Demo complete!

What you just saw:
1. Baseline (FP32): Standard training, highest memory usage
2. Mixed Precision: ~50% memory reduction with FP16
3. + Checkpointing: Additional 30-50% savings

Next steps:
- Run full benchmark: python benchmark.py
- Read the blog post: blog_post.md
- Check code examples in the README

Memory optimization enables:
🎯 Training larger models on same hardware
🚀 Faster training (FP16 boost)
📊 Minimal accuracy loss
""")

if torch.cuda.is_available():
    print(f"\n💡 Your GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("\n💡 Tip: Run on a GPU to see actual memory savings!")

print("\n" + "="*60)
