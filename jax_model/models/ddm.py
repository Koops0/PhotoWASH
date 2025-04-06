# DDM paper code by Minglong Xue, Jinhong He, Shivakumara Palaiahnakote, Mingliang Zhou
# This is a JAX implementation of the Denoising Diffusion model, which is used in the diffusion model.
# Translated by Kershan A.
# 4/6/25

#General imports
import os
import numpy as np

#JAX + Flax imports
import jax
import jax.numpy as jnp
from flax import nnx, traverse_util

import pyiqa

import jax_model.utils as utils
from jax_model.models.fgm_jax import FGM
from jax_model.models.unet_jax import DiffusionUNet

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
        # Add leading zero without device handling
        beta_with_zero = jnp.concatenate([jnp.zeros(1), beta], axis=0)
        # Cumulative product along axis 0
        alpha_cumprod = jnp.cumprod(1 - beta_with_zero, axis=0)
        # Index by t+1 and reshape
        alpha_selected = jnp.take(alpha_cumprod, t + 1)
        return jnp.reshape(alpha_selected, (-1, 1, 1, 1))
    
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
            t = jnp.ones(n, dtype=jnp.int32) * i
            next_t = jnp.ones(n, dtype=jnp.int32) * j
            at = self.compute_alpha(b, t)
            at_next = self.compute_alpha(b, next_t)
            xt = xs[-1]

            xt_nhwc = jnp.transpose(xt, (0, 2, 3, 1))  # NCHW -> NHWC
            x_c_nhwc = jnp.transpose(x_c, (0, 2, 3, 1))  # NCHW -> NHWC
            
            # Concatenate along the channel axis (last dimension in NHWC)
            model_input = jnp.concatenate([xt_nhwc, x_c_nhwc], axis=3)
            
            # Run model and convert output back to NCHW
            et_nhwc = self.diffusion_model(model_input, t)
            et = jnp.transpose(et_nhwc, (0, 3, 1, 2))  # NHWC -> NCHW
            
            # Continue with the rest of your code using et in NCHW format
            x0_t = (xt - et * jnp.sqrt(1 - at)) / jnp.sqrt(at)


            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            key, subkey = jax.random.split(key)  # Get a new random key for this step
            noise = jax.random.normal(subkey, x.shape)  # Generate noise with the same shape as x
            xt_next = at_next.sqrt() * x0_t + c1 * noise + c2 * et

            xs.append(xt_next)

        return xs


    def __call__(self, x, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)

        data_dict = {}
        dwt, idwt = WaveletTransform(), WaveletTransform()

        # Extract input image
        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img)

        input_LL, input_h0 = input_dwt[:n, :, :, :], input_dwt[n:, :, :, :]

        b = self.betas  # No device handling needed in JAX

        b1 = self.betas  # No device handling needed in JAX
        key, subkey = jax.random.split(key)
        t1 = jax.random.randint(
            subkey, 
            shape=(input_LL.shape[0] // 2 + 1,), 
            minval=0, 
            maxval=self.num_timesteps
        )
        # Concatenate and slice
        t1 = jnp.concatenate([t1, self.num_timesteps - t1 - 1], axis=0)[:input_LL.shape[0]]

        # Replace cumprod and index_select with JAX operations
        alpha_cumprod = jnp.cumprod(1 - b1, axis=0)
        a1 = jnp.reshape(alpha_cumprod[t1 + 1], (-1, 1, 1, 1))

        key, subkey = jax.random.split(key)
        e1 = jax.random.normal(subkey, input_LL.shape)

        b2 = self.betas
        key, subkey = jax.random.split(key)
        t2 = jax.random.randint(
            subkey, 
            shape=(input_h0.shape[0] // 2 + 1,), 
            minval=0, 
            maxval=self.num_timesteps
        )
        t2 = jnp.concatenate([t2, self.num_timesteps - t2 - 1], axis=0)[:input_h0.shape[0]]

        # Similar JAX-style operations for a2
        alpha_cumprod2 = jnp.cumprod(1 - b2, axis=0)
        a2 = jnp.reshape(alpha_cumprod2[t2 + 1], (-1, 1, 1, 1))

        key, subkey = jax.random.split(key)
        e2 = jax.random.normal(subkey, input_h0.shape)

        # Use is_training flag instead of .training attribute
        # Typically passed through apply() in JAX/Flax
        is_training = False  # You may want to pass this as an argument

        if not is_training:
            img_list = self.sample_training(input_img, b)
            pred_x = img_list[-1]

            key, subkey = jax.random.split(key)
            pred_x_list_1 = self.sample_training(pred_x, b2)
            pred_x_1 = pred_x_list_1[-1]

            pred_x_dwt = dwt.dwt(pred_x_1)
            pred_x_LL, pred_x_h0 = pred_x_dwt[:n, :, :, :], pred_x_dwt[n:, :, :, :]

            key, subkey = jax.random.split(key)
            pred_LL_list = self.sample_training(pred_x_LL, b1)
            pred_LL = pred_LL_list[-1]

            pred_x_h0 = self.fgm_model(pred_x_h0)
            pred_x_2 = idwt.iwt(jnp.concatenate([pred_LL, pred_x_h0], axis=0))

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
        self.params = self.model(dummy_input, key=key)

        self.ema_helper = EMAHelper(mu=0.9999)
        self.ema_helper.register(self.model)
        self.ema_helper.update(self.model)
        self.model.apply(self.params, dummy_input, key=key)
        self.model = self.model.apply(self.params, dummy_input, key=key)
        self.model = self.model.to(self.device)
        
        
        # Set up pmap for parallel processing
        self.pmapped_model_apply = jax.pmap(
            lambda model, x, key: model(x, key=key),
            axis_name='devices'
        )

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        
        # In JAX/nnx, parameters are part of the model object itself
        # We need a custom loading mechanism
        if 'state_dict' in checkpoint:
            # Convert checkpoint to nnx model parameters
            self._load_parameters_from_checkpoint(checkpoint['state_dict'])
        
        # Load EMA helper state
        if 'ema_helper' in checkpoint and hasattr(self, 'ema_helper'):
            self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        
        # Apply EMA if requested
        if ema and hasattr(self, 'ema_helper'):
            self.model, ema_params = self.ema_helper.ema_copy(self.model)
            
        print(f"Loaded checkpoint from: {load_path}")
        print(f"Checkpoint exists: {os.path.exists(load_path)}")
    
    def _load_parameters_from_checkpoint(self, state_dict):
        # Flatten the state_dict for easier access
        flat_params = traverse_util.flatten_dict(state_dict)
        
        # Unflatten and assign to model parameters
        for name, param in flat_params.items():
            key = "/".join(name)
            if key in self.params:
                self.params[key] = param
            else:
                print(f"Warning: Key {key} not found in model parameters.")