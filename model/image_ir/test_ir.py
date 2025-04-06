import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from jax_model.models import DiffusiveRestoration, DenoisingDiffusion

# Replace torch with jax imports
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import datasets
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

    # Use current script directory to find configs folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", args.config)
    
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: '{config_path}'")
    
    if os.path.isdir(config_path):
        raise ValueError(f"Config path '{config_path}' is a directory, not a file. Please specify a config file.")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config