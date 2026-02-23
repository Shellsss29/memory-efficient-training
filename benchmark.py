"""
Memory-Efficient Training Benchmark
Compares different memory optimization techniques for deep learning

Author: Shailly Bhati
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Check if bitsandbytes is available
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not installed. 8-bit optimizer tests will be skipped.")
    print("Install with: pip install bitsandbytes")

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# ========== MODEL DEFINITIONS ==========

class ResNetBlock(nn.Module):
    """Basic ResNet block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SimpleResNet(nn.Module):
    """Simplified ResNet for benchmarking"""
    def __init__(self, num_classes=100, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Create residual blocks
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.ModuleList(layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        # Layer 1
        for block in self.layer1:
            if self.use_checkpointing:
                out = checkpoint.checkpoint(block, out)
            else:
                out = block(out)

        # Layer 2
        for block in self.layer2:
            if self.use_checkpointing:
                out = checkpoint.checkpoint(block, out)
            else:
                out = block(out)

        # Layer 3
        for block in self.layer3:
            if self.use_checkpointing:
                out = checkpoint.checkpoint(block, out)
            else:
                out = block(out)

        # Layer 4
        for block in self.layer4:
            if self.use_checkpointing:
                out = checkpoint.checkpoint(block, out)
            else:
                out = block(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ========== UTILITY FUNCTIONS ==========

def get_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0, 0


def clear_memory():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_cifar100_loaders(batch_size=128):
    """Get CIFAR-100 data loaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return trainloader, testloader


# ========== TRAINING FUNCTIONS ==========

def train_epoch_baseline(model, trainloader, optimizer, criterion, device):
    """Baseline training (FP32, no optimizations)"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(trainloader), 100. * correct / total


def train_epoch_mixed_precision(model, trainloader, optimizer, criterion, device, scaler):
    """Training with mixed precision (FP16)"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(trainloader), 100. * correct / total


@torch.no_grad()
def evaluate(model, testloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(testloader), 100. * correct / total


# ========== BENCHMARK EXPERIMENTS ==========

def run_benchmark(config_name, model, trainloader, testloader, optimizer,
                 criterion, device, num_epochs=3, use_amp=False):
    """Run training benchmark and collect metrics"""
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"{'='*60}")

    clear_memory()

    metrics = {
        'memory_allocated': [],
        'memory_peak': [],
        'train_time': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    scaler = GradScaler() if use_amp else None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Record memory before training
        clear_memory()
        start_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # Train
        start_time = time.time()

        if use_amp:
            train_loss, train_acc = train_epoch_mixed_precision(
                model, trainloader, optimizer, criterion, device, scaler
            )
        else:
            train_loss, train_acc = train_epoch_baseline(
                model, trainloader, optimizer, criterion, device
            )

        train_time = time.time() - start_time

        # Record peak memory
        peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        # Store metrics
        metrics['memory_allocated'].append(start_mem)
        metrics['memory_peak'].append(peak_mem)
        metrics['train_time'].append(train_time)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)

        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")
        print(f"Memory Peak: {peak_mem:.2f} GB | Time: {train_time:.1f}s")

    return metrics


def main():
    """Main benchmark function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Benchmarks will run on CPU.")
        print("Memory measurements will not be meaningful.\n")

    # Load data
    print("\nLoading CIFAR-100 dataset...")
    trainloader, testloader = get_cifar100_loaders(batch_size=128)
    print("Dataset loaded!")

    # Store results
    all_results = {}

    # ========== EXPERIMENT 1: Baseline (FP32) ==========
    print("\n" + "="*60)
    print("EXPERIMENT 1: Baseline (FP32 + Regular Adam)")
    print("="*60)

    model = SimpleResNet(num_classes=100, use_checkpointing=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    all_results['Baseline (FP32)'] = run_benchmark(
        'Baseline (FP32)', model, trainloader, testloader,
        optimizer, criterion, device, num_epochs=3, use_amp=False
    )

    del model, optimizer
    clear_memory()

    # ========== EXPERIMENT 2: Mixed Precision (FP16) ==========
    print("\n" + "="*60)
    print("EXPERIMENT 2: Mixed Precision (FP16)")
    print("="*60)

    model = SimpleResNet(num_classes=100, use_checkpointing=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    all_results['FP16'] = run_benchmark(
        'Mixed Precision (FP16)', model, trainloader, testloader,
        optimizer, criterion, device, num_epochs=3, use_amp=True
    )

    del model, optimizer
    clear_memory()

    # ========== EXPERIMENT 3: FP16 + Gradient Checkpointing ==========
    print("\n" + "="*60)
    print("EXPERIMENT 3: FP16 + Gradient Checkpointing")
    print("="*60)

    model = SimpleResNet(num_classes=100, use_checkpointing=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    all_results['FP16 + Checkpointing'] = run_benchmark(
        'FP16 + Checkpointing', model, trainloader, testloader,
        optimizer, criterion, device, num_epochs=3, use_amp=True
    )

    del model, optimizer
    clear_memory()

    # ========== EXPERIMENT 4: FP16 + Checkpointing + 8-bit Optimizer ==========
    if HAS_BITSANDBYTES and torch.cuda.is_available():
        print("\n" + "="*60)
        print("EXPERIMENT 4: FP16 + Checkpointing + 8-bit Adam")
        print("="*60)

        model = SimpleResNet(num_classes=100, use_checkpointing=True).to(device)
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        all_results['FP16 + Checkpointing + 8bit'] = run_benchmark(
            'All Optimizations', model, trainloader, testloader,
            optimizer, criterion, device, num_epochs=3, use_amp=True
        )

        del model, optimizer
        clear_memory()

    # ========== PLOT RESULTS ==========
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)

    plot_results(all_results)
    print("\nBenchmark complete! Results saved to benchmark_results.png")


def plot_results(all_results):
    """Plot comparison of different configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Memory-Efficient Training: Benchmark Results', fontsize=16, fontweight='bold')

    configs = list(all_results.keys())

    # Extract final epoch metrics
    peak_memory = [all_results[c]['memory_peak'][-1] for c in configs]
    train_time = [sum(all_results[c]['train_time']) / len(all_results[c]['train_time']) for c in configs]
    final_acc = [all_results[c]['test_acc'][-1] for c in configs]

    # Plot 1: Peak Memory Usage
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(configs)), peak_memory, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(configs)])
    ax1.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax1.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([c.replace(' + ', '\n+\n') for c in configs], rotation=0, ha='center', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} GB', ha='center', va='bottom', fontsize=10)

    # Plot 2: Training Time
    ax2 = axes[0, 1]
    baseline_time = train_time[0] if train_time[0] > 0 else 1
    relative_time = [t / baseline_time * 100 for t in train_time]
    bars2 = ax2.bar(range(len(configs)), relative_time, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(configs)])
    ax2.set_ylabel('Relative Time (%)', fontsize=12)
    ax2.set_title('Training Speed (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([c.replace(' + ', '\n+\n') for c in configs], rotation=0, ha='center', fontsize=9)
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=10)

    # Plot 3: Test Accuracy
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(configs)), final_acc, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(configs)])
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Final Test Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels([c.replace(' + ', '\n+\n') for c in configs], rotation=0, ha='center', fontsize=9)
    ax3.set_ylim([min(final_acc) - 5, max(final_acc) + 2])
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    # Create table data
    table_data = []
    table_data.append(['Configuration', 'Memory', 'Speed', 'Accuracy'])
    for i, config in enumerate(configs):
        memory_reduction = (1 - peak_memory[i] / peak_memory[0]) * 100 if peak_memory[0] > 0 else 0
        table_data.append([
            config,
            f"{peak_memory[i]:.1f}GB\n(-{memory_reduction:.0f}%)" if i > 0 else f"{peak_memory[i]:.1f}GB",
            f"{relative_time[i]:.0f}%",
            f"{final_acc[i]:.1f}%"
        ])

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code rows
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_facecolor(colors[i-1] if i-1 < len(colors) else 'white')
            table[(i, j)].set_alpha(0.3)

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("Saved: benchmark_results.png")


if __name__ == '__main__':
    main()
