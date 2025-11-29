#!/usr/bin/env python3
"""
Quick start script for flow-based generative models.

This script provides a simple way to get started with the project.
It will train a small model on MNIST for demonstration purposes.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from models.flows import RealNVPFlow
from models.trainer import FlowTrainer
from data.datasets import get_mnist_loaders
from utils.utils import set_seed, get_device, save_checkpoint


def quick_start():
    """Run a quick demonstration of flow-based generative models."""
    print("🌊 Flow-based Generative Models - Quick Start")
    print("=" * 50)
    
    # Set up
    device = get_device()
    set_seed(42)
    
    print(f"Using device: {device}")
    
    # Create a small model for quick training
    model = RealNVPFlow(
        in_channels=1,
        num_coupling_layers=4,  # Smaller for quick training
        mid_channels=128,       # Smaller for quick training
        image_size=28
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create trainer
    trainer = FlowTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        log_interval=100
    )
    
    # Get MNIST data (smaller batch size for quick training)
    train_loader, val_loader = get_mnist_loaders(
        data_dir="./data",
        batch_size=128,
        num_workers=2
    )
    
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
    # Quick training (just 5 epochs for demonstration)
    print("\nStarting quick training (5 epochs)...")
    
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
        print(f"Bits per Dimension: {val_metrics['bits_per_dim']:.4f}")
        
        # Generate samples
        if epoch == 4:  # Generate samples on last epoch
            samples = trainer.generate_samples(num_samples=16)
            
            # Save samples
            import matplotlib.pyplot as plt
            
            samples = (samples + 1) / 2  # Denormalize
            samples = torch.clamp(samples, 0, 1)
            
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            fig.suptitle('Generated MNIST Digits', fontsize=16)
            
            for i in range(16):
                row = i // 4
                col = i % 4
                img = samples[i].squeeze().cpu().numpy()
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig('samples/quick_start_samples.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("Generated samples saved to samples/quick_start_samples.png")
    
    # Save final model
    save_checkpoint(
        model, optimizer, 4, val_metrics["val_loss"],
        "checkpoints/quick_start_model.pth",
        **val_metrics
    )
    
    print("\n✅ Quick start completed!")
    print("Next steps:")
    print("1. Check out the generated samples in samples/quick_start_samples.png")
    print("2. Run the full training: python 0369.py --model realnvp --dataset mnist --epochs 50")
    print("3. Try the interactive demo: streamlit run demo/app.py")
    print("4. Explore the Jupyter notebook: jupyter notebook notebooks/demo.ipynb")


if __name__ == "__main__":
    quick_start()
