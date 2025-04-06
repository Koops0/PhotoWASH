import argparse
import os
import sys
# Use absolute path to ensure consistent imports regardless of where script is run
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import yaml
import jax
import jax.numpy as jnp  # Replace torch with jax
import numpy as np
from model import datasets
# Now use explicit import path
from jax_model.models import DenoisingDiffusion, DiffusiveRestoration
from tqdm import tqdm

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate Image restoration tasks')
    parser.add_argument("--config", required=True, type=str,
                    help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=20826, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    # Use script directory to find configs folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", args.config)
    
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: '{config_path}'")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    
    # Set up JAX devices instead of torch devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    if jax.default_backend() == 'gpu':
        print(f"Using GPU: {jax.devices('gpu')}")
    config.device = None  # JAX handles devices differently
    
    # Initialize random key from seed
    key = jax.random.PRNGKey(args.seed)
    # JAX uses a key-splitting approach for randomness
    key, subkey = jax.random.split(key)
    
    # Set numpy random seed for dataset operations
    np.random.seed(args.seed)
    
    # data loading - dataset code remains the same
    print(f"Current Task '{config.data.task}'")
    print(f"Current dataset '{config.data.val_dataset}'")
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # limit testing to 20 images from the validation set  
    val_loader.dataset.input_names = val_loader.dataset.input_names[:20]
    
    # Initialize model (JAX style)
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    
    # Create a progress bar for the validation loader
    val_loader_with_progress = tqdm(val_loader, desc='Loading Validation Data', leave=False)
    
    # Run restoration (pass the random key to the restore method)
    model.restore(val_loader_with_progress)

if __name__ == '__main__':
    main()