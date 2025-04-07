# Restoration paper code by Minglong Xue, Jinhong He, Shivakumara Palaiahnakote, Mingliang Zhou
# This is a JAX implementation of the Restoration model, which is used in the diffusion model.
# Translated by Kershan A.
# 4/6/25

import jax
import jax.numpy as jnp
import numpy as np

import os
import torch  # Still needed for data loading and saving
from PIL import Image

# Data transformation function
def data_transform(X):
    return 2 * X - 1.0

# Inverse
def inverse_transform(X):
    return jnp.clip((X + 1.0) / 2.0, 0.0, 1.0)

# JAX-compatible image saving function
def save_jax_image(img_array, filepath):
    if img_array.min() < 0 or img_array.max() > 1.0:
        print(f"Warning: Image values outside range: min={img_array.min()}, max={img_array.max()}")
        img_array = inverse_transform(img_array)

    # Convert to numpy, ensure proper shape and range
    img_np = np.array(img_array)

    # Remove batch dim if present and transpose from NCHW to HWMC
    if len(img_np.shape) == 4:
        img_np = img_np[0]  # Remove batch dimension
    img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
    # Scale to 0-255 range for PIL
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    # Save with PIL
    Image.fromarray(img_np).save(filepath)
    print(f"Saved image to {filepath}")

# Diffusion Restoration model (most code from paper)
class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            print("Using EMA parameters for sampling")
        else:
            print('Pre-trained diffusion model path is missing!')

    # This function restores the image
    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        os.makedirs(image_folder, exist_ok=True)
        
        print("Starting image restoration with JAX...")

        for i, (x, y) in enumerate(val_loader):
            # Process one image at a time to avoid memory issues
            for b_idx in range(x.shape[0]):
                if isinstance(x, torch.Tensor):
                    # Extract single image and convert to JAX
                    x_single = jnp.array(x[b_idx:b_idx+1].cpu().numpy())
                else:
                    x_single = x[b_idx:b_idx+1]

                # Process single image with detailed diagnostics
                x_cond = x_single[:, :3, :, :]
                
                # Ensure proper normalization to [0,1] first
                if x_cond.max() > 1.0:
                    print(f"Normalizing input from range [{x_cond.min()}, {x_cond.max()}] to [0,1]")
                    x_cond = x_cond / 255.0
                
                # Transform to [-1,1] for model
                x_cond = data_transform(x_cond)
                print(f"Input after transform: min={x_cond.min()}, max={x_cond.max()}, mean={x_cond.mean()}")
                
                # Get dimensions and pad if needed
                _, c, h, w = x_cond.shape
                img_h_32 = int(32 * jnp.ceil(h / 32.0))
                img_w_32 = int(32 * jnp.ceil(w / 32.0))

                # Add padding if necessary
                if h != img_h_32 or w != img_w_32:
                    x_cond = jnp.pad(x_cond, 
                        ((0, 0), (0, 0), (0, img_h_32 - h), (0, img_w_32 - w)),
                        mode='reflect')
                    print(f"Padded input to shape {x_cond.shape}")

                # Clear GPU cache
                if hasattr(jax, 'clear_backends'):
                    jax.clear_backends()

                try:
                    # Create a fixed random key for deterministic results
                    key = jax.random.PRNGKey(42)
                    
                    # Process with deterministic sampling (eta=0)
                    x_output = self.diffusion.model.sample_training(
                        x_cond, 
                        self.diffusion.model.betas,
                        eta=0.0,  # Deterministic sampling
                        dm_num=True,
                    )[-1]
                    
                    # Crop to original size
                    x_output = x_output[:, :, :h, :w]
                    
                    # Apply inverse transform explicitly
                    print(f"Raw output stats: min={x_output.min()}, max={x_output.max()}, mean={x_output.mean()}")
                    
                    # Apply inverse transform to get values in [0,1] range
                    x_output = inverse_transform(x_output)
                    print(f"Transformed output: min={x_output.min()}, max={x_output.max()}, mean={x_output.mean()}")
                    
                    # Save using JAX image saving function
                    img_name = f"{y[b_idx]}.png" if isinstance(y, list) or isinstance(y, tuple) else f"{i}_{b_idx}.png"
                    save_jax_image(x_output, os.path.join(image_folder, img_name))
                    
                    print(f"Processed image {i*x.shape[0]+b_idx+1}")

                except (ValueError, RuntimeError) as e:
                    if "RESOURCE_EXHAUSTED" in str(e):
                        print(f"GPU memory exhausted for image {i*x.shape[0]+b_idx+1}, skipping...")
                    else:
                        print(f"Error processing image: {str(e)}")
                        raise e

    def restore_single_loader(self, val_loader):
        image_folder = self.args.image_folder
        os.makedirs(image_folder, exist_ok=True)
        
        for i, (x, y) in enumerate(val_loader):
            if isinstance(x, torch.Tensor):
                # Convert PyTorch tensor to JAX array
                x = jnp.array(x.cpu().numpy())
            
            # Process image using JAX operations
            x_cond = x[:, :3, :, :]
            b, c, h, w = x_cond.shape
            img_h_32 = int(32 * jnp.ceil(h / 32.0))
            img_w_32 = int(32 * jnp.ceil(w / 32.0))
            
            # Use JAX padding format
            x_cond = jnp.pad(x_cond, 
                ((0, 0), (0, 0), (0, img_h_32 - h), (0, img_w_32 - w)),
                mode='reflect')
            
            # Use consistent approach with restore method
            key = jax.random.PRNGKey(42)  # Fixed seed for deterministic results
            x_output = self.diffusion.model.sample_training(
                x_cond, 
                self.diffusion.model.betas,
                eta=0.0,
                dm_num=True
            )[-1]
            
            x_output = x_output[:, :, :h, :w]
            x_output = inverse_transform(x_output)
            
            # Save using our JAX-friendly function
            img_name = f"{y[0]}.png" if isinstance(y, list) or isinstance(y, tuple) else f"{y}.png"
            save_jax_image(x_output, os.path.join(image_folder, img_name))
            print(f"Processed image {i+1}")

    # Restoration function with JAX approach
    def diffusive_restoration(self, x_cond):
        key = jax.random.PRNGKey(42)
        x_output = self.diffusion.model.sample_training(
            x_cond, 
            self.diffusion.model.betas,
            eta=0.0, 
            dm_num=True
        )[-1]
        return x_output