import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

def create_denoising_dataset(coco_dir, output_dir, num_images=500, noise_level=50):
    """
    Creates a denoising dataset from COCO images with Gaussian noise.
    
    Args:
        coco_dir (str): Path to COCO images folder (e.g. 'train2017')
        output_dir (str): Where to save the new dataset
        num_images (int): Number of images to select (default: 500)
        noise_level (int): Standard deviation of Gaussian noise (default: 50)
    """
    # Create output directories
    gt_dir = os.path.join(output_dir, 'GT')
    lq_dir = os.path.join(output_dir, 'LQ')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)
    
    # Get all COCO image paths
    coco_images = list(Path(coco_dir).glob('*.jpg'))  # Adjust extension if needed
    random.shuffle(coco_images)
    selected_images = coco_images[:num_images]
    
    print(f"Processing {len(selected_images)} images...")
    
    # Process each image
    for img_path in tqdm(selected_images):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Convert to float32 for noise addition
        img_float = img.astype(np.float32)
        
        # Add Gaussian noise (σ=50)
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        noisy_img = np.clip(img_float + noise, 0, 255).astype(np.uint8)
        
        # Save with standardized names
        base_name = img_path.stem
        gt_name = f"normal_{base_name}.png"
        lq_name = f"low_{base_name}.png"
        
        cv2.imwrite(os.path.join(gt_dir, gt_name), img)
        cv2.imwrite(os.path.join(lq_dir, lq_name), noisy_img)
    
    # Create train.txt listing
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for img_path in selected_images:
            f.write(f"low_{img_path.stem}.png\n")
    
    print(f"\nDataset created at {output_dir}")
    print(f"- GT (clean) images: {len(os.listdir(gt_dir))}")
    print(f"- LQ (noisy) images: {len(os.listdir(lq_dir))}")

if __name__ == "__main__":
    # Example usage
    create_denoising_dataset(
        coco_dir="./data/denoising/DN_CBSCOCO/training_subset",  # Change to your COCO path
        output_dir="./data/denoising/DN_CBSCOCO/train",  # Will create GT/LQ folders here
        num_images=1000,
        noise_level=50
    )