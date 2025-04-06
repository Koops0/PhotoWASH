import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

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