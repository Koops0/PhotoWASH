import os
import shutil
from pathlib import Path

def rename_denoising_dataset(test_dir):
    """
    Organizes a deraining dataset with mixed files into GT/LQ subfolders.
    
    Args:
        root_dir (str): Path to the folder containing train/test subfolders
    """
    
    gt_dir = os.path.join(test_dir, 'GT')
    lq_dir = os.path.join(test_dir, 'LQ')
  
    # Process each file in the split directory
    for subdir in [gt_dir, lq_dir]: 
        for filename in os.listdir(subdir):
            filepath = os.path.join(subdir, filename)
                
            # Skip directories and non-image files
            if os.path.isdir(filepath) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            if subdir == gt_dir:
                new_name = "normal_" + filename
                print(new_name)
                os.rename(filepath, os.path.join(gt_dir, new_name))
                print(f"Renamed GT: {filename} -> {new_name}")
        
            if subdir == lq_dir:
                new_name = "low_" + filename
                print(new_name)
                os.rename(filepath, os.path.join(lq_dir, new_name))
                print(f"Renamed LQ: {filename} -> {new_name}")


def update_list_files(test_dir):
    """
    Updates test.txt to list the new 'low_*' filenames.
    """
    lq_dir = os.path.join(test_dir, 'LQ')
    list_file = os.path.join(test_dir, 'test.txt')
                
    # Get all renamed LQ files (low_*.png)
    lq_files = [f for f in os.listdir(lq_dir) if  f.startswith('low_')]
    lq_files.sort()
        
    # Overwrite the list file
    with open(list_file, 'w') as f:
        for filename in lq_files:
            f.write(f"{filename}\n")
        
    print(f"Updated {list_file} with {len(lq_files)} entries")


if __name__ == "__main__":
    # Example usage - point this to your dataset root folder
    test_path = "./data/denoising/DN_CBSCOCO/test"  # Change this to your path
    
    # Organize files into GT/LQ folders
    # rename_denoising_dataset(test_path)

    update_list_files(test_path)
        
    print("Dataset renaming complete!")