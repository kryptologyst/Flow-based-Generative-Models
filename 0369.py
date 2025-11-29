#!/usr/bin/env python3
"""
Project 369: Flow-based Generative Models

A modern implementation of flow-based generative models including RealNVP and Glow.
This replaces the previous CNN classifier with proper flow-based architectures.

Usage:
    python 0369.py --model realnvp --dataset cifar10 --epochs 50
    python 0369.py --model glow --dataset mnist --epochs 30
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from models.flows import RealNVPFlow, GlowFlow
from models.trainer import FlowTrainer
from data.datasets import get_cifar10_loaders, get_mnist_loaders, get_fashion_mnist_loaders
from utils.utils import set_seed, get_device, save_checkpoint


def create_model(model_name: str, dataset: str) -> torch.nn.Module:
    """Create flow-based model based on name and dataset.
    
    Args:
        model_name: Name of the model ('realnvp' or 'glow')
        dataset: Name of the dataset ('cifar10', 'mnist', 'fashion_mnist')
        
    Returns:
        Flow-based model
    """
    # Determine input channels and image size based on dataset
    if dataset == "cifar10":
        in_channels = 3
        image_size = 32
    else:  # mnist or fashion_mnist
        in_channels = 1
        image_size = 28
    
    if model_name == "realnvp":
        model = RealNVPFlow(
            in_channels=in_channels,
            num_coupling_layers=8,
            mid_channels=512,
            image_size=image_size
        )
    elif model_name == "glow":
        model = GlowFlow(
            in_channels=in_channels,
            num_scales=2,
            num_steps_per_scale=4,
            mid_channels=512,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def get_data_loaders(dataset: str, batch_size: int = 64):
    """Get data loaders for the specified dataset.
    
    Args:
        dataset: Name of the dataset
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if dataset == "cifar10":
        return get_cifar10_loaders(
            data_dir="./data",
            batch_size=batch_size,
            num_workers=2
        )
    elif dataset == "mnist":
        return get_mnist_loaders(
            data_dir="./data",
            batch_size=batch_size,
            num_workers=2
        )
    elif dataset == "fashion_mnist":
        return get_fashion_mnist_loaders(
            data_dir="./data",
            batch_size=batch_size,
            num_workers=2
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def save_samples(samples: torch.Tensor, epoch: int, model_name: str, dataset: str):
    """Save generated samples as images.
    
    Args:
        samples: Generated samples
        epoch: Current epoch
        model_name: Name of the model
        dataset: Name of the dataset
    """
    os.makedirs("samples", exist_ok=True)
    
    # Denormalize samples from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    nrow = min(8, samples.shape[0])
    fig, axes = plt.subplots(nrow, nrow, figsize=(10, 10))
    fig.suptitle(f'{model_name.upper()} Generated Samples - Epoch {epoch}', fontsize=16)
    
    for i in range(nrow * nrow):
        row = i // nrow
        col = i % nrow
        
        if i < samples.shape[0]:
            img = samples[i].permute(1, 2, 0).cpu().numpy()
            axes[row, col].imshow(img)
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'samples/{model_name}_{dataset}_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train flow-based generative models')
    parser.add_argument('--model', type=str, default='realnvp', choices=['realnvp', 'glow'],
                       help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'mnist', 'fashion_mnist'],
                       help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    print(f"Using device: {device}")
    print(f"Model: {args.model.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
    # Create model
    model = create_model(args.model, args.dataset)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Create trainer
    trainer = FlowTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        log_interval=50
    )
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(args.dataset, args.batch_size)
    print(f"Data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Evaluate
        if (epoch + 1) % 5 == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
            print(f"Bits per Dimension: {val_metrics['bits_per_dim']:.4f}")
            
            # Save best model
            if val_metrics["val_loss"] < best_loss:
                best_loss = val_metrics["val_loss"]
                save_checkpoint(
                    model, optimizer, epoch, val_metrics["val_loss"],
                    f"checkpoints/best_{args.model}_{args.dataset}.pth",
                    **val_metrics
                )
        
        # Generate and save samples
        if (epoch + 1) % 10 == 0:
            samples = trainer.generate_samples(num_samples=64)
            save_samples(samples, epoch + 1, args.model, args.dataset)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics["train_loss"],
                f"checkpoints/checkpoint_{args.model}_{args.dataset}_epoch_{epoch + 1}.pth"
            )
    
    print("Training completed!")
    print(f"Best model saved as: checkpoints/best_{args.model}_{args.dataset}.pth")


if __name__ == "__main__":
    main()