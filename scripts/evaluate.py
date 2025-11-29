"""Evaluation script for flow-based generative models."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.flows import RealNVPFlow, GlowFlow
from src.data.datasets import get_cifar10_loaders, get_mnist_loaders, get_fashion_mnist_loaders
from src.utils.utils import set_seed, get_device, load_checkpoint, compute_bits_per_dimension


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


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 1000
) -> Dict[str, float]:
    """Evaluate model on test data.
    
    Args:
        model: Trained flow model
        data_loader: Test data loader
        device: Device to evaluate on
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    total_log_prob = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Evaluating"):
            data = data.to(device)
            
            # Compute log probability
            log_prob = model.log_prob(data)
            total_log_prob += log_prob.sum().item()
            total_samples += data.shape[0]
            
            if total_samples >= num_samples:
                break
    
    # Compute average log probability
    avg_log_prob = total_log_prob / total_samples
    
    # Compute bits per dimension
    data_shape = (data.shape[1], data.shape[2], data.shape[3])  # (C, H, W)
    bits_per_dim = compute_bits_per_dimension(torch.tensor(avg_log_prob), data_shape)
    
    return {
        "avg_log_prob": avg_log_prob,
        "bits_per_dim": bits_per_dim,
        "num_samples": total_samples
    }


def compute_likelihood_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Compute likelihood-based metrics.
    
    Args:
        model: Trained flow model
        data_loader: Test data loader
        device: Device to compute on
        
    Returns:
        Dictionary containing likelihood metrics
    """
    model.eval()
    
    log_probs = []
    
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Computing likelihood metrics"):
            data = data.to(device)
            log_prob = model.log_prob(data)
            log_probs.extend(log_prob.cpu().numpy())
    
    log_probs = np.array(log_probs)
    
    return {
        "mean_log_prob": np.mean(log_probs),
        "std_log_prob": np.std(log_probs),
        "min_log_prob": np.min(log_probs),
        "max_log_prob": np.max(log_probs),
        "median_log_prob": np.median(log_probs)
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate flow-based generative models')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to evaluate')
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt',
                       help='File to save evaluation results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    print(f"Model loaded from {args.checkpoint}")
    
    # Get test data loader
    if config["data"]["dataset"] == "cifar10":
        _, test_loader = get_cifar10_loaders(
            data_dir=config["data"]["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"]
        )
    elif config["data"]["dataset"] == "mnist":
        _, test_loader = get_mnist_loaders(
            data_dir=config["data"]["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"]
        )
    elif config["data"]["dataset"] == "fashion_mnist":
        _, test_loader = get_fashion_mnist_loaders(
            data_dir=config["data"]["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"]
        )
    else:
        raise ValueError(f"Unknown dataset: {config['data']['dataset']}")
    
    print(f"Test dataset: {config['data']['dataset']}")
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate model
    print("Evaluating model...")
    eval_metrics = evaluate_model(model, test_loader, device, args.num_samples)
    
    # Compute likelihood metrics
    print("Computing likelihood metrics...")
    likelihood_metrics = compute_likelihood_metrics(model, test_loader, device)
    
    # Combine all metrics
    all_metrics = {
        **eval_metrics,
        **likelihood_metrics,
        "checkpoint": args.checkpoint,
        "config": str(config)
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average Log Probability: {eval_metrics['avg_log_prob']:.4f}")
    print(f"Bits per Dimension: {eval_metrics['bits_per_dim']:.4f}")
    print(f"Number of Samples: {eval_metrics['num_samples']}")
    print(f"Mean Log Prob: {likelihood_metrics['mean_log_prob']:.4f}")
    print(f"Std Log Prob: {likelihood_metrics['std_log_prob']:.4f}")
    print(f"Min Log Prob: {likelihood_metrics['min_log_prob']:.4f}")
    print(f"Max Log Prob: {likelihood_metrics['max_log_prob']:.4f}")
    print(f"Median Log Prob: {likelihood_metrics['median_log_prob']:.4f}")
    print("="*50)
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        f.write("Flow-based Generative Model Evaluation Results\n")
        f.write("="*50 + "\n")
        for key, value in all_metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
