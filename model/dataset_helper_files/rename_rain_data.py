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

        gt_dir = os.path.join(split_dir, 'GT')
        lq_dir = os.path.join(split_dir, 'LQ')
  
        # Process each file in the split directory
        for subdir in [gt_dir, lq_dir]: 
            for filename in os.listdir(subdir):
                filepath = os.path.join(subdir, filename)
                
                # Skip directories and non-image files
                if os.path.isdir(filepath) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                            
                # Process GT files (clean)
                if '_clean' in filename:
                    new_name = filename.replace('_clean', '_normal')
                    shutil.move(filepath, os.path.join(gt_dir, new_name))
                    print(f"Moved GT: {filename} -> {new_name}")
                    
                # Process LQ files (rain)
                elif '_rain' in filename:
                    new_name = filename.replace('_rain', '_low')
                    shutil.move(filepath, os.path.join(lq_dir, new_name))
                    print(f"Moved LQ: {filename} -> {new_name}")
                    
                # Ignore rain graphics files (rainregion-*, rainstreak-*)

def update_list_files(root_dir):
    """
    Updates train.txt/test.txt to list the new 'low_*' filenames.
    """
    for split in ['train', 'test']:
        lq_dir = os.path.join(root_dir, split, 'LQ')
        list_file = os.path.join(root_dir, split, f'{split}.txt')
        
        if not os.path.exists(lq_dir):
            continue
            
        # Get all renamed LQ files (low_*.png)
        lq_files = [f for f in os.listdir(lq_dir) if '_low' in f]
        lq_files.sort()
        
        # Overwrite the list file
        with open(list_file, 'w') as f:
            for filename in lq_files:
                f.write(f"{filename}\n")
        
        print(f"Updated {list_file} with {len(lq_files)} entries")


if __name__ == "__main__":
    # Example usage - point this to your dataset root folder
    dataset_root = "data/raindrop/RainDrop"  # Change this to your path
    
    # Organize files into GT/LQ folders
    # organize_deraining_dataset(dataset_root)

    update_list_files(dataset_root)
        
    print("Dataset renaming complete!")