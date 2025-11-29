"""Utility functions for flow-based generative models."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device.
    
    Returns:
        Available device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_bits_per_dimension(
    log_prob: Tensor, 
    data_shape: Tuple[int, ...]
) -> float:
    """Compute bits per dimension from log probability.
    
    Args:
        log_prob: Log probability tensor
        data_shape: Shape of the data (excluding batch dimension)
        
    Returns:
        Bits per dimension
    """
    num_pixels = np.prod(data_shape)
    bits_per_dim = -log_prob.mean().item() / (num_pixels * np.log(2.0))
    return bits_per_dim


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs: Any
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def interpolate_latent(
    model: nn.Module,
    z1: Tensor,
    z2: Tensor,
    num_steps: int = 10
) -> Tensor:
    """Interpolate between two latent vectors.
    
    Args:
        model: Flow model
        z1: First latent vector
        z2: Second latent vector
        num_steps: Number of interpolation steps
        
    Returns:
        Interpolated samples
    """
    model.eval()
    
    # Create interpolation weights
    alphas = torch.linspace(0, 1, num_steps, device=z1.device)
    
    interpolated_samples = []
    
    with torch.no_grad():
        for alpha in alphas:
            # Linear interpolation in latent space
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Transform to data space
            x_interp, _ = model.inverse(z_interp)
            interpolated_samples.append(x_interp)
    
    return torch.cat(interpolated_samples, dim=0)


def compute_fid_score(
    real_features: Tensor,
    fake_features: Tensor
) -> float:
    """Compute Fréchet Inception Distance (FID) score.
    
    Note: This is a simplified version. For production use, consider
    using clean-fid or torch-fidelity libraries.
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        
    Returns:
        FID score
    """
    # Compute means and covariances
    mu1 = real_features.mean(dim=0)
    mu2 = fake_features.mean(dim=0)
    
    sigma1 = torch.cov(real_features.T)
    sigma2 = torch.cov(fake_features.T)
    
    # Compute FID
    diff = mu1 - mu2
    covmean = torch.sqrt(sigma1 @ sigma2)
    
    fid = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)
    
    return fid.item()


def compute_inception_score(
    logits: Tensor,
    splits: int = 10
) -> Tuple[float, float]:
    """Compute Inception Score (IS).
    
    Args:
        logits: Logits from Inception model
        splits: Number of splits for computation
        
    Returns:
        Tuple of (mean IS, std IS)
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)
    
    # Split into groups
    split_size = logits.shape[0] // splits
    scores = []
    
    for i in range(splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size
        
        if i == splits - 1:
            end_idx = logits.shape[0]
        
        split_probs = probs[start_idx:end_idx]
        
        # Compute marginal distribution
        marginal = split_probs.mean(dim=0)
        
        # Compute KL divergence
        kl_div = split_probs * (torch.log(split_probs + 1e-16) - torch.log(marginal + 1e-16))
        kl_div = kl_div.sum(dim=1)
        
        # Compute IS for this split
        is_score = torch.exp(kl_div.mean())
        scores.append(is_score.item())
    
    return np.mean(scores), np.std(scores)


def create_sample_grid(
    samples: Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None
) -> Tensor:
    """Create a grid of samples for visualization.
    
    Args:
        samples: Sample tensors
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize values
        value_range: Range for normalization
        
    Returns:
        Grid tensor
    """
    if normalize:
        if value_range is None:
            # Normalize to [0, 1]
            samples = (samples - samples.min()) / (samples.max() - samples.min())
        else:
            # Normalize to specified range
            samples = (samples - value_range[0]) / (value_range[1] - value_range[0])
    
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
