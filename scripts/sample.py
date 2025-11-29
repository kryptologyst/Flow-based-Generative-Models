"""Sampling script for flow-based generative models."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from src.models.flows import RealNVPFlow, GlowFlow
from src.utils.utils import set_seed, get_device, load_checkpoint, interpolate_latent


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


def generate_samples(
    model: torch.nn.Module,
    num_samples: int,
    device: torch.device,
    seed: int = None
) -> torch.Tensor:
    """Generate samples from the model.
    
    Args:
        model: Trained flow model
        num_samples: Number of samples to generate
        device: Device to generate on
        seed: Random seed for reproducibility
        
    Returns:
        Generated samples
    """
    if seed is not None:
        set_seed(seed)
    
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device)
    
    return samples


def create_sample_grid(samples: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Create a grid of samples for visualization.
    
    Args:
        samples: Sample tensors
        nrow: Number of images per row
        
    Returns:
        Grid tensor
    """
    # Denormalize from [-1, 1] to [0, 1]
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
    
    return grid.squeeze(0)


def save_samples_as_images(
    samples: torch.Tensor,
    output_path: str,
    nrow: int = 8,
    title: str = "Generated Samples"
) -> None:
    """Save samples as image files.
    
    Args:
        samples: Generated samples
        output_path: Path to save images
        nrow: Number of images per row
        title: Title for the plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create grid
    grid = create_sample_grid(samples, nrow)
    
    # Save as image
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title(title)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Samples saved to {output_path}")


def interpolate_samples(
    model: torch.nn.Module,
    device: torch.device,
    num_steps: int = 10
) -> torch.Tensor:
    """Create interpolation between random samples.
    
    Args:
        model: Trained flow model
        device: Device to use
        num_steps: Number of interpolation steps
        
    Returns:
        Interpolated samples
    """
    model.eval()
    
    # Generate two random latent vectors
    z1 = torch.randn(1, model.in_channels, model.image_size, model.image_size, device=device)
    z2 = torch.randn(1, model.in_channels, model.image_size, model.image_size, device=device)
    
    # Interpolate
    interpolated = interpolate_latent(model, z1, z2, num_steps)
    
    return interpolated


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description='Generate samples from flow-based models')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./samples',
                       help='Directory to save samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--interpolate', action='store_true',
                       help='Generate interpolation samples')
    parser.add_argument('--interpolation_steps', type=int, default=10,
                       help='Number of interpolation steps')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    print(f"Model loaded from {args.checkpoint}")
    print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    
    # Generate samples
    if args.interpolate:
        print("Generating interpolation samples...")
        samples = interpolate_samples(model, device, args.interpolation_steps)
        output_path = f"{args.output_dir}/interpolation_samples.png"
        title = f"Latent Space Interpolation ({args.interpolation_steps} steps)"
    else:
        print(f"Generating {args.num_samples} samples...")
        samples = generate_samples(model, args.num_samples, device, args.seed)
        output_path = f"{args.output_dir}/generated_samples.png"
        title = f"Generated Samples (n={args.num_samples})"
    
    # Save samples
    save_samples_as_images(samples, output_path, title=title)
    
    # Also save individual samples if requested
    if args.num_samples <= 16:  # Only for small numbers
        individual_dir = f"{args.output_dir}/individual"
        os.makedirs(individual_dir, exist_ok=True)
        
        for i, sample in enumerate(samples):
            sample_path = f"{individual_dir}/sample_{i:03d}.png"
            save_samples_as_images(
                sample.unsqueeze(0), 
                sample_path, 
                nrow=1, 
                title=f"Sample {i}"
            )
    
    print("Sampling completed!")


if __name__ == "__main__":
    main()
