# Restoration paper code by Minglong Xue, Jinhong He, Shivakumara Palaiahnakote, Mingliang Zhou
# This is a JAX implementation of the Restoration model, which is used in the diffusion model.
# Translated by Kershan A.
# 4/6/25

import jax
import jax.numpy as jnp
from flax import nnx

from . import utilsrdm
import os

import torch

# Data transformation function
def data_transform(X):
    return 2 * X - 1.0

#Inverse
def inverse_transform(X):
    return jnp.clamp((X + 1.0) / 2.0, 0.0, 1.0)

# Diffusion Restoration model (most code from paper)
class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.Module.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    # This function restores the image
    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        for i, (x, y) in enumerate(val_loader):
            if isinstance(x, torch.Tensor):
                x = jnp.array(x.cpu().numpy())
        
            x_cond = x[:, :3, :, :]
            b, c, h, w = x_cond.shape
            img_h_32 = int(32 * jnp.ceil(h / 32.0))
            img_w_32 = int(32 * jnp.ceil(w / 32.0))
            x_cond = jnp.pad(x_cond, 
                ((0, 0),           # Batch dimension: no padding
                 (0, 0),           # Channel dimension: no padding
                 (0, img_h_32 - h),  # Height dimension: pad after
                 (0, img_w_32 - w)),  # Width dimension: pad after
                mode='reflect')
            x_output = self.diffusion.model.sample_training(x_cond, self.diffusion.model.betas)[-1]
            x_output = x_output[:, :, :h, :w]

            utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))

    def restore_single_loader(self, val_loader):
        image_folder = self.args.image_folder
        for i, (x, y) in enumerate(val_loader):
            x_cond = x[:, :3, :, :].to(self.diffusion.device)
            b, c, h, w = x_cond.shape
            img_h_32 = int(32 * jnp.ceil(h / 32.0))
            img_w_32 = int(32 * jnp.ceil(w / 32.0))
            x_cond = jnp.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
            x_output = self.diffusion(x_cond)
            x_output = x_output[:, :, :h, :w]

            utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))

    # Restoration function
    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.Module(x_cond)
        return x_output
