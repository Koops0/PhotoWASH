import argparse
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from models import DenoisingDiffusion, DiffusiveRestoration
import cv2
import utils
from tqdm import tqdm

class SingleImageDataset(Dataset):
    """Dataset for a single image that mimics the format of the validation loader."""
    
    def __init__(self, image, filename="image.png"):
        """
        Args:
            image: numpy array (H, W, C) in RGB format 0-255
            filename: the filename to assign to this image
        """
        self.image = image
        self.filename = filename
        self.input_names = [filename]  # Match the structure in test_ir.py
    
    def __len__(self):
        return 1
    
    def __getitem__(self, _):
        # Convert to tensor and normalize to match the format in your dataset class
        img_tensor = torch.from_numpy(self.image).float()
        
        # Convert from (H, W, C) to (C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # Normalize to [0, 1]
        img_tensor = img_tensor / 255.0
        
        # Return in the same format as your validation loader
        # First tensor is the input image, second is the filename
        return img_tensor, self.filename

def create_single_image_loader(image, filename="image.png", batch_size=1):
    """
    Creates a DataLoader for a single image that matches the validation loader format.
    
    Args:
        image: numpy array (H, W, C) in RGB format 0-255
        filename: name to give the image
        batch_size: should typically be 1 for single image processing
        
    Returns:
        A DataLoader that can be passed to model.restore()
    """
    dataset = SingleImageDataset(image, filename)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

class ImageRestoration:
    def dict2namespace(self, config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = self.dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    def __init__(self, config_path, model_path, time_steps=10, seed=20826, output_dir="./output"):
        # Load config and initialize models
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = self.dict2namespace(config)

        # Set up output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.args = argparse.Namespace(
            config=config_path,
            resume=model_path,
            sampling_timesteps=time_steps,
            image_folder=self.output_dir,
            seed=seed
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config.device = self.device

        self.diffusion = DenoisingDiffusion(self.args, self.config)
        self.restorer = DiffusiveRestoration(self.diffusion, self.args, self.config)
        print(f"Model loaded successfully from {model_path}")

    def restore_image_data(self, lq_image, output_filename="restored_image.png"):
        """
        Process a single image through the same pipeline as validation images
        
        Args:
            lq_image: numpy array (H, W, C) in RGB format 0-255
            output_filename: name for the output file (without path)
            
        Returns:
            Restored image as numpy array (H, W, C) in RGB format 0-255
        """
        # Ensure image is in RGB format
        if lq_image.shape[2] == 3 and isinstance(lq_image, np.ndarray):
            # Check if image is in BGR format (OpenCV default)
            if cv2.cvtColor(cv2.cvtColor(lq_image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB).sum() != lq_image.sum():
                lq_image = cv2.cvtColor(lq_image, cv2.COLOR_BGR2RGB)
                print("Converted image from BGR to RGB")
        
        # Debug info
        print(f"Processing image with shape: {lq_image.shape}, dtype: {lq_image.dtype}")
         
        # Setup filename to match validation structure
        base_name = os.path.splitext(output_filename)[0]
        
        # Create loader for single image
        loader = create_single_image_loader(lq_image, filename=base_name)
        
        # Configure the image folder in args
        self.args.image_folder = self.output_dir
              
        # Process the image through the restore method - 
        # use tqdm to show progress, similar to the validation loop
        print(f"Running restoration on single image...")
        self.restorer.restore_single_loader(tqdm(loader, desc='Processing Image'))
        
        # Construct the path to the restored image as done in restoration.py
        # Note: The file name is constructed as {base_name}_restored.png
        restored_path = os.path.join(self.output_dir, f"{base_name}_restored.png")
        
        print(f"Looking for restored image at: {restored_path}")
        
        # Read the restored image
        if os.path.exists(restored_path):
            print(f"Found restored image at: {restored_path}")
            restored_image = cv2.imread(restored_path)
            restored_image = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                   
            return restored_image
        else:
            print(f"ERROR: Could not find restored image at {restored_path}")
            print(f"Available files in directory:")
            if os.path.exists(self.output_dir):
                print(os.listdir(self.output_dir))
            else:
                print(f"Directory {self.output_dir} does not exist")
                
            # Debug: List all files in output directory recursively
            print("Searching output directory for restored files:")
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    if "_restored" in file:
                        print(f"Found restored file: {os.path.join(root, file)}")
            
            raise FileNotFoundError(f"Restored image not found at {restored_path}")
    
    def restore_batch(self, image_paths, output_dir=None):
        """
        Restore multiple images and save them to output directory
        
        Args:
            image_paths: List of paths to images to restore
            output_dir: Directory to save restored images (defaults to self.output_dir)
            
        Returns:
            List of paths to restored images
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        restored_paths = []
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            restored = self.restore_image_data(img, filename)
            
            # Save restored image
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_restored.png")
            cv2.imwrite(output_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
            restored_paths.append(output_path)
        
        return restored_paths