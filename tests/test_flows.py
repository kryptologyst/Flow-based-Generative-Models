"""Tests for flow-based generative models."""

import pytest
import torch
import numpy as np

from src.models.flows import RealNVPFlow, GlowFlow, CouplingLayer
from src.utils.utils import set_seed, get_device, compute_bits_per_dimension


class TestCouplingLayer:
    """Test cases for CouplingLayer."""
    
    def test_coupling_layer_forward_inverse(self):
        """Test that forward and inverse operations are consistent."""
        layer = CouplingLayer(in_channels=4, mid_channels=64)
        x = torch.randn(2, 4, 8, 8)
        
        # Forward pass
        y, log_det = layer(x)
        
        # Inverse pass
        x_reconstructed, log_det_inv = layer.inverse(y)
        
        # Check reconstruction
        assert torch.allclose(x, x_reconstructed, atol=1e-5)
        
        # Check log determinant
        assert torch.allclose(log_det + log_det_inv, torch.zeros_like(log_det), atol=1e-5)
    
    def test_coupling_layer_shape_preservation(self):
        """Test that coupling layer preserves input shape."""
        layer = CouplingLayer(in_channels=6, mid_channels=128)
        x = torch.randn(3, 6, 16, 16)
        
        y, _ = layer(x)
        
        assert y.shape == x.shape


class TestRealNVPFlow:
    """Test cases for RealNVPFlow."""
    
    def test_realnvp_forward_inverse(self):
        """Test that forward and inverse operations are consistent."""
        model = RealNVPFlow(in_channels=3, num_coupling_layers=4, image_size=32)
        x = torch.randn(2, 3, 32, 32)
        
        # Forward pass
        z, log_det = model.forward(x)
        
        # Inverse pass
        x_reconstructed, log_det_inv = model.inverse(z)
        
        # Check reconstruction
        assert torch.allclose(x, x_reconstructed, atol=1e-4)
        
        # Check log determinant
        assert torch.allclose(log_det + log_det_inv, torch.zeros_like(log_det), atol=1e-4)
    
    def test_realnvp_log_prob(self):
        """Test log probability computation."""
        model = RealNVPFlow(in_channels=1, num_coupling_layers=2, image_size=8)
        x = torch.randn(4, 1, 8, 8)
        
        log_prob = model.log_prob(x)
        
        assert log_prob.shape == (4,)
        assert torch.isfinite(log_prob).all()
    
    def test_realnvp_sampling(self):
        """Test sampling from the model."""
        model = RealNVPFlow(in_channels=2, num_coupling_layers=3, image_size=16)
        device = torch.device("cpu")
        
        samples = model.sample(num_samples=8, device=device)
        
        assert samples.shape == (8, 2, 16, 16)
        assert torch.isfinite(samples).all()


class TestGlowFlow:
    """Test cases for GlowFlow."""
    
    def test_glow_forward_inverse(self):
        """Test that forward and inverse operations are consistent."""
        model = GlowFlow(in_channels=3, num_scales=2, num_steps_per_scale=2, image_size=32)
        x = torch.randn(2, 3, 32, 32)
        
        # Forward pass
        z, log_det = model.forward(x)
        
        # Inverse pass
        x_reconstructed, log_det_inv = model.inverse(z)
        
        # Check reconstruction
        assert torch.allclose(x, x_reconstructed, atol=1e-4)
        
        # Check log determinant
        assert torch.allclose(log_det + log_det_inv, torch.zeros_like(log_det), atol=1e-4)
    
    def test_glow_log_prob(self):
        """Test log probability computation."""
        model = GlowFlow(in_channels=1, num_scales=1, num_steps_per_scale=2, image_size=8)
        x = torch.randn(4, 1, 8, 8)
        
        log_prob = model.log_prob(x)
        
        assert log_prob.shape == (4,)
        assert torch.isfinite(log_prob).all()
    
    def test_glow_sampling(self):
        """Test sampling from the model."""
        model = GlowFlow(in_channels=2, num_scales=2, num_steps_per_scale=2, image_size=16)
        device = torch.device("cpu")
        
        samples = model.sample(num_samples=8, device=device)
        
        assert samples.shape == (8, 2, 16, 16)
        assert torch.isfinite(samples).all()


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test that seeding works correctly."""
        set_seed(42)
        rand1 = torch.randn(10)
        
        set_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_compute_bits_per_dimension(self):
        """Test bits per dimension computation."""
        log_prob = torch.tensor([-100.0, -200.0, -300.0])
        data_shape = (3, 32, 32)
        
        bits_per_dim = compute_bits_per_dimension(log_prob, data_shape)
        
        assert isinstance(bits_per_dim, float)
        assert bits_per_dim > 0


if __name__ == "__main__":
    pytest.main([__file__])
