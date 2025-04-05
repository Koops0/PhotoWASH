# DDM paper code by Minglong Xue, Jinhong He, Shivakumara Palaiahnakote, Mingliang Zhou
# This is a JAX implementation of the Diffusion model, which is used in the diffusion model.
# Translated by Kershan A.
# 4/6/25


#General imports
import os
import numpy as np
import math
import time

#JAX + Flax imports
import jax
import jax.numpy as jnp
from flax import nnx, traverse_util
from flax.training import train_state
import optax
import dm_pix as pix

import cv2 
import albumentations as A

import pyiqa

import utils
from models.fgm_jax import FGM
from models.unet_jax import DiffusionUNet

# Frequency and Wavelet transforms
class FrequencyTransform(nnx.Module):
    def __init__(self):
        super(FrequencyTransform, self).__init__()

    def __call__(self, dp):
        # Apply the frequency transform
        dp = jnp.fft.rfft2(dp, norm='backward')
        dp_amp = jnp.abs(dp)
        dp_phase = jnp.angle(dp)
        return dp_amp, dp_phase
    
class WaveletTransform(nnx.Module):
    def __init__(self):
        super(WaveletTransform, self).__init__()

    # Inverse Wavelet Transform (from paper)
    def iwt(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch = int(in_batch / (r**2))
        out_channel, out_height, out_width = in_channel, r * in_height, r * in_width
        x1 = x[0:out_batch, :, :, :] / 2
        x2 = x[out_batch:out_batch * 2, :, :, :] / 2
        x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
        x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

        h = jnp.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

    # Discrete Wavelet Transform (from paper)
    def dwt(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return jnp.concatenate([x_LL, x_HL, x_LH, x_HH], axis=0)
    
    def __call__(self, x, inverse=False):
        # Apply the wavelet transform
        if inverse:
            return self.iwt(x)
        else:
            return self.dwt(x)
        
#Normalize class
class Normalize:
    @staticmethod
    def apply(x):
        ymax = 255
        ymin = 0
        xmax = x.max()
        xmin = x.min()
        return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return jnp.clamp((X + 1.0) / 2.0, 0.0, 1.0)
        
#EMA Helper. The Exponential Moving Average (EMA) is used to smooth the model weights during training.
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        flat_params = traverse_util.flatten_dict(module.params)
        self.shadow = {
            "/".join(k): jnp.array(v) 
            for k, v in flat_params.items()
        }

    def update(self, module):
        flat_params = traverse_util.flatten_dict(module.params)
        
        # Update each parameter in the shadow dictionary
        for name, param in flat_params.items():
            key = "/".join(name)
            if key in self.shadow:
                self.shadow[key] = (1.0 - self.mu) * param + self.mu * self.shadow[key]

    def ema(self):
        flat_shadow = {
            tuple(k.split("/")): v 
            for k, v in self.shadow.items()
        }
        ema_params = traverse_util.unflatten_dict(flat_shadow)
        return {"params": ema_params}

    def ema_copy(self, module):
        ema_params = self.ema()
        return module, ema_params


    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
    
# This is a function to get the beta schedule for the diffusion model
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    #Sigmoid (CUSTOM)
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))
    
    if beta_schedule == "quad":
        betas = (jnp.linspace(beta_start ** 0.5, beta_end ** 0.5,
                 num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = jnp.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * jnp.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd": # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0/jnp.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = jnp.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise ValueError(f"Unknown beta schedule: {beta_schedule}")
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

#Main DDM Net Class
class Net(nnx.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()
        self.config = config
        self.args = args
        self.device = config.device

        # Initialize the diffusion model
        self.diffusion_model = DiffusionUNet(config)
        
        # Initialize the FGM model
        self.fgm_model = FGM(in_c=3,out_c=64)

        #Initialize the beta schedule
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = jnp.asarray(betas, dtype=jnp.float32)
        self.num_timesteps = self.betas.shape[0]

    #Compute alpha
    def compute_alpha(self, beta, t):
        beta = jnp.concat([jnp.zeros(1).to(beta.device)], axis=0)
        alpha = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return alpha
    
    # Sample training fn
    def sample_training(self, x_c, b, dm_num=True, eta=0.):
        # Skip and sequence
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)

        # Next timestep
        skip_1 = self.config.diffusion.num_diffusion_timesteps_1 // (self.args.sampling_timesteps + 1)
        seq_1 = range(0, self.config.diffusion.num_diffusion_timesteps_1, skip_1)

        n,c,h,w = x_c.shape

        seq_next = [-1] + list(seq[:-1])
        seq_next_1 = [-1] + list(seq[:-1])

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (n, c, h, w))
        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)) if dm_num else zip(reversed(seq_1), reversed(seq_next_1)):
            t = (jnp.ones(n) * i).to(x.device)
            next_t = (jnp.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.diffusion_model(jnp.concat([xt, x_c], axis=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            key, subkey = jax.random.split(key)  # Get a new random key for this step
            noise = jax.random.normal(subkey, x.shape)  # Generate noise with the same shape as x
            xt_next = at_next.sqrt() * x0_t + c1 * noise + c2 * et

            xs.append(xt_next.to(x.device))

        return xs


    def __call__(self, x):
        data_dict = {}
        dwt, idwt = WaveletTransform(), WaveletTransform()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img)

        input_LL, input_h0 = input_dwt[:n, :, :, :], input_dwt[n:, :, :, :]

        b = self.betas.to(input_img.device)

        b1 = self.betas.to(input_LL.device)
        key, subkey = jax.random.split(key)  # Get a new random key
        t1 = jax.random.randint(
            subkey, 
            shape=(input_LL.shape[0] // 2 + 1,), 
            minval=0, 
            maxval=self.num_timesteps
        )
        t1 = jnp.concat([t1, self.num_timesteps - t1 - 1],
                      dim=0)[: input_LL.shape[0]].to(x.device)
        a1 = (1-b1).cumprod(dim=0).index_select(0, t1 + 1).view(-1, 1, 1, 1)
        e1 = jax.random.normal(subkey, input_LL.shape)

        b2 = self.betas.to(input_LL.device)
        key, subkey = jax.random.split(key)
        t2 = jax.random.randint(
            subkey, 
            shape=(input_h0.shape[0] // 2 + 1,), 
            minval=0, 
            maxval=self.num_timesteps
        )
        t2 = jnp.concat([t2, self.num_timesteps - t2 - 1],
                      axis=0)[: input_h0.shape[0]].to(x.device)
        a2 = (1-b2).cumprod(dim=0).index_select(0, t2 + 1).view(-1, 1, 1, 1)
        e2 = jax.random.normal(subkey, input_h0.shape)

        #Sample training conditional
        if self.training==False:
            img_list = self.sample_training(input_img, b)
            pred_x = img_list[-1]

            pred_x_list_1 = self.sample_training(pred_x, b2)
            pred_x_1 = pred_x_list_1[-1]

            pred_x_dwt = dwt.dwt(pred_x_1)
            pred_x_LL, pred_x_h0 = pred_x_dwt[:n, :, :, :], pred_x_dwt[n:, :, :, :]
            pred_LL_list = self.sample_training(pred_x_LL, b1)
            pred_LL = pred_LL_list[-1]
            pred_x_h0 = self.fgm_model(pred_x_h0)
            pred_x_2 = idwt.iwt(jnp.concat([pred_LL, pred_x_h0], axis=0))

            # Append the results to the data dictionary
            data_dict["pred_x"] = pred_x
            data_dict["pred_x_1"] = pred_x_1
            data_dict['pred_x_2'] = pred_x_2

            return data_dict
        
# Class for denoising diffusion
class DenoisingDiffusion (object):
    def __init__(self, args, config):
        super(DenoisingDiffusion, self).__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='rgb')
        self.model = Net(args, config) 
        
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 3, 256, 256))  # Adjust shape as needed
        self.params = self.model.init(key, dummy_input)

        self.pmapped_model_apply = jax.pmap(
            lambda params, x: self.model.apply(params, x),
            axis_name='devices'
        )

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("Load checkpoint: ", os.path.exists(load_path))
        print("Current checkpoint: {}".format(load_path))
        