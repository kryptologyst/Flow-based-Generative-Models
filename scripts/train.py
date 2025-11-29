"""Main training script for flow-based generative models."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from src.models.flows import RealNVPFlow, GlowFlow
from src.models.trainer import FlowTrainer
from src.data.datasets import get_cifar10_loaders, get_mnist_loaders, get_fashion_mnist_loaders
from src.utils.utils import (
    set_seed, get_device, save_checkpoint, load_checkpoint,
    compute_bits_per_dimension, create_sample_grid
)


def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Flow-based model
    """
    model_config = config["model"]
    
    if model_config["name"] == "realnvp":
        model = RealNVPFlow(
            in_channels=model_config["in_channels"],
            num_coupling_layers=model_config["num_coupling_layers"],
            mid_channels=model_config["mid_channels"],
            image_size=model_config["image_size"]
        )
    elif model_config["name"] == "glow":
        model = GlowFlow(
            in_channels=model_config["in_channels"],
            num_scales=2,
            num_steps_per_scale=model_config["num_coupling_layers"] // 2,
            mid_channels=model_config["mid_channels"],
            image_size=model_config["image_size"]
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")
    
    return model


def get_data_loaders(config: Dict[str, Any]) -> tuple:
    """Get data loaders based on configuration.
    
    Args:
        config: Data configuration
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config["data"]
    
    if data_config["dataset"] == "cifar10":
        return get_cifar10_loaders(
            data_dir=data_config["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=data_config["num_workers"],
            image_size=data_config["image_size"]
        )
    elif data_config["dataset"] == "mnist":
        return get_mnist_loaders(
            data_dir=data_config["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=data_config["num_workers"],
            image_size=data_config["image_size"]
        )
    elif data_config["dataset"] == "fashion_mnist":
        return get_fashion_mnist_loaders(
            data_dir=data_config["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=data_config["num_workers"],
            image_size=data_config["image_size"]
        )
    else:
        raise ValueError(f"Unknown dataset: {data_config['dataset']}")


def save_samples(
    samples: torch.Tensor,
    epoch: int,
    sample_dir: str,
    nrow: int = 8
) -> None:
    """Save generated samples as images.
    
    Args:
        samples: Generated samples
        epoch: Current epoch
        sample_dir: Directory to save samples
        nrow: Number of images per row
    """
    os.makedirs(sample_dir, exist_ok=True)
    
    # Denormalize samples from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    grid = torch.nn.functional.grid_sample(
        samples.unsqueeze(0),
        torch.nn.functional.affine_grid(
            torch.eye(2, 3).unsqueeze(0),
            samples.shape[1:]
        ),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    # Save as image
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title(f'Generated Samples - Epoch {epoch}')
    plt.savefig(f'{sample_dir}/samples_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train flow-based generative models')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    if config["device"]["deterministic"]:
        set_seed(config["device"]["seed"])
    
    # Create directories
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["sample_dir"], exist_ok=True)
    
    # Create model
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Create trainer
    trainer = FlowTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        log_interval=config["training"]["log_interval"]
    )
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(config)
    print(f"Data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Evaluate
        if (epoch + 1) % config["training"]["eval_interval"] == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
            print(f"Bits per Dimension: {val_metrics['bits_per_dim']:.4f}")
            
            # Save best model
            if val_metrics["val_loss"] < best_loss:
                best_loss = val_metrics["val_loss"]
                save_checkpoint(
                    model, optimizer, epoch, val_metrics["val_loss"],
                    f"{config['paths']['checkpoint_dir']}/best_model.pth",
                    **val_metrics
                )
        
        # Generate and save samples
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            samples = trainer.generate_samples(num_samples=64)
            save_samples(samples, epoch + 1, config["paths"]["sample_dir"])
        
        # Save checkpoint
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics["train_loss"],
                f"{config['paths']['checkpoint_dir']}/checkpoint_epoch_{epoch + 1}.pth"
            )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
