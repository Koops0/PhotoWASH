import os
import shutil
from pathlib import Path

def organize_deraining_dataset(root_dir):
    """
    Organizes a deraining dataset with mixed files into GT/LQ subfolders.
    
    Args:
        root_dir (str): Path to the folder containing train/test subfolders
    """
    for split in ['train', 'test']:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue
     
        # Create GT and LQ subfolders if they don't exist
        gt_dir = os.path.join(split_dir, 'GT')
        lq_dir = os.path.join(split_dir, 'LQ')
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(lq_dir, exist_ok=True)
        
        # Process each file in the split directory
        for filename in os.listdir(split_dir):
            filepath = os.path.join(split_dir, filename)
            
            # Skip directories and non-image files
            if os.path.isdir(filepath) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            # Move GT files (nonrain-*)
            if filename.startswith('norain-'):
                shutil.move(filepath, os.path.join(gt_dir, filename))
                
            # Move LQ files (rain-*)
            elif filename.startswith('rain-'):
                shutil.move(filepath, os.path.join(lq_dir, filename))
                
            # Ignore rain graphics files (rainregion-*, rainstreak-*)


if __name__ == "__main__":
    # Example usage - point this to your dataset root folder
    dataset_root = "data/deraining/Rain100H"  # Change this to your path
    
    # Organize files into GT/LQ folders
    organize_deraining_dataset(dataset_root)
        
    print("Dataset organization complete!")