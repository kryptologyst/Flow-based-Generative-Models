"""Streamlit demo for flow-based generative models."""

import os
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

from src.models.flows import RealNVPFlow, GlowFlow
from src.utils.utils import set_seed, get_device, load_checkpoint, interpolate_latent


@st.cache_resource
def load_model(checkpoint_path: str, config: dict):
    """Load model with caching.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        
    Returns:
        Loaded model
    """
    device = get_device()
    
    # Create model
    if config["model"]["name"] == "realnvp":
        model = RealNVPFlow(
            in_channels=config["model"]["in_channels"],
            num_coupling_layers=config["model"]["num_coupling_layers"],
            mid_channels=config["model"]["mid_channels"],
            image_size=config["model"]["image_size"]
        )
    elif config["model"]["name"] == "glow":
        model = GlowFlow(
            in_channels=config["model"]["in_channels"],
            num_scales=2,
            num_steps_per_scale=config["model"]["num_coupling_layers"] // 2,
            mid_channels=config["model"]["mid_channels"],
            image_size=config["model"]["image_size"]
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, device


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.
    
    Args:
        tensor: Image tensor
        
    Returns:
        PIL Image
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and PIL
    if tensor.dim() == 4:  # Batch
        tensor = tensor[0]  # Take first image
    
    numpy_array = tensor.permute(1, 2, 0).cpu().numpy()
    numpy_array = (numpy_array * 255).astype(np.uint8)
    
    return Image.fromarray(numpy_array)


def create_sample_grid(samples: torch.Tensor, nrow: int = 8) -> Image.Image:
    """Create a grid of samples.
    
    Args:
        samples: Sample tensors
        nrow: Number of images per row
        
    Returns:
        PIL Image of the grid
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
    
    grid = grid.squeeze(0)
    numpy_array = grid.permute(1, 2, 0).cpu().numpy()
    numpy_array = (numpy_array * 255).astype(np.uint8)
    
    return Image.fromarray(numpy_array)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Flow-based Generative Models Demo",
        page_icon="🌊",
        layout="wide"
    )
    
    st.title("🌊 Flow-based Generative Models Demo")
    st.markdown("Generate images using RealNVP and Glow flow-based models")
    
    # Sidebar for controls
    st.sidebar.header("Model Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["realnvp", "glow"],
        help="Choose the flow-based model architecture"
    )
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["cifar10", "mnist", "fashion_mnist"],
        help="Choose the dataset the model was trained on"
    )
    
    # Checkpoint selection
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoint_files:
            checkpoint_file = st.sidebar.selectbox(
                "Checkpoint",
                checkpoint_files,
                help="Choose a trained model checkpoint"
            )
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        else:
            st.sidebar.error("No checkpoint files found in checkpoints/ directory")
            st.stop()
    else:
        st.sidebar.error("Checkpoints directory not found")
        st.stop()
    
    # Model configuration
    config = {
        "model": {
            "name": model_type,
            "in_channels": 3 if dataset == "cifar10" else 1,
            "num_coupling_layers": 8,
            "mid_channels": 512,
            "image_size": 32 if dataset == "cifar10" else 28
        }
    }
    
    # Load model
    try:
        model, device = load_model(checkpoint_path, config)
        st.sidebar.success(f"Model loaded successfully!")
        st.sidebar.info(f"Device: {device}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Generation controls
    st.sidebar.header("Generation Controls")
    
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=1,
        max_value=64,
        value=16,
        help="Number of images to generate"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000000,
        value=42,
        help="Seed for reproducible generation"
    )
    
    # Generate samples button
    if st.sidebar.button("Generate Samples", type="primary"):
        with st.spinner("Generating samples..."):
            set_seed(seed)
            
            with torch.no_grad():
                samples = model.sample(num_samples, device)
            
            # Create grid
            grid_image = create_sample_grid(samples)
            
            # Display
            st.image(grid_image, caption=f"Generated {num_samples} samples", use_column_width=True)
            
            # Download button
            img_buffer = io.BytesIO()
            grid_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="Download Samples",
                data=img_buffer.getvalue(),
                file_name=f"flow_samples_{model_type}_{dataset}_{num_samples}.png",
                mime="image/png"
            )
    
    # Interpolation section
    st.header("Latent Space Interpolation")
    st.markdown("Generate smooth transitions between two random points in latent space")
    
    col1, col2 = st.columns(2)
    
    with col1:
        interpolation_steps = st.slider(
            "Interpolation Steps",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of steps in the interpolation"
        )
    
    with col2:
        interpolation_seed = st.number_input(
            "Interpolation Seed",
            min_value=0,
            max_value=1000000,
            value=123,
            help="Seed for interpolation points"
        )
    
    if st.button("Generate Interpolation", type="secondary"):
        with st.spinner("Generating interpolation..."):
            set_seed(interpolation_seed)
            
            with torch.no_grad():
                # Generate two random latent vectors
                z1 = torch.randn(1, config["model"]["in_channels"], 
                               config["model"]["image_size"], 
                               config["model"]["image_size"], device=device)
                z2 = torch.randn(1, config["model"]["in_channels"], 
                               config["model"]["image_size"], 
                               config["model"]["image_size"], device=device)
                
                # Interpolate
                interpolated = interpolate_latent(model, z1, z2, interpolation_steps)
            
            # Create grid
            grid_image = create_sample_grid(interpolated, nrow=interpolation_steps)
            
            # Display
            st.image(grid_image, caption=f"Latent space interpolation ({interpolation_steps} steps)", 
                    use_column_width=True)
            
            # Download button
            img_buffer = io.BytesIO()
            grid_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="Download Interpolation",
                data=img_buffer.getvalue(),
                file_name=f"flow_interpolation_{model_type}_{dataset}_{interpolation_steps}.png",
                mime="image/png"
            )
    
    # Model information
    st.header("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", model_type.upper())
    
    with col2:
        st.metric("Dataset", dataset.upper())
    
    with col3:
        st.metric("Image Size", f"{config['model']['image_size']}x{config['model']['image_size']}")
    
    # Model architecture details
    st.subheader("Architecture Details")
    st.write(f"**Model**: {model_type.upper()}")
    st.write(f"**Input Channels**: {config['model']['in_channels']}")
    st.write(f"**Coupling Layers**: {config['model']['num_coupling_layers']}")
    st.write(f"**Hidden Channels**: {config['model']['mid_channels']}")
    st.write(f"**Image Size**: {config['model']['image_size']}x{config['model']['image_size']}")
    
    # Instructions
    st.header("Instructions")
    st.markdown("""
    1. **Select Model**: Choose between RealNVP and Glow architectures
    2. **Choose Dataset**: Select the dataset the model was trained on
    3. **Load Checkpoint**: Pick a trained model checkpoint
    4. **Generate**: Use the controls to generate samples or interpolations
    5. **Download**: Save generated images to your computer
    
    **Tips**:
    - Try different seeds for varied results
    - Use interpolation to see smooth transitions in latent space
    - Adjust the number of samples based on your needs
    """)


if __name__ == "__main__":
    main()
