import os
import glob
from pathlib import Path

def rename_to_low_normal_structure(root_dir):
    """
    Renames files to standardized 'low_' (LQ) and 'normal_' (GT) naming convention.
    Works for both train and test folders.
    
    Args:
        root_dir (str): Path to dataset root (e.g., './data/dehazing/RESIDE-6k')
    """
    for split in ['train', 'test']:
        # Paths to original LQ/GT folders
        lq_dir = os.path.join(root_dir, split, 'LQ')
        gt_dir = os.path.join(root_dir, split, 'GT')
        
        if not os.path.exists(lq_dir) or not os.path.exists(gt_dir):
            continue  # Skip if folders don't exist
        
        print(f"Processing {split}...")
        
        # Rename LQ files (e.g., '1.png' -> 'low_1.png')
        for lq_path in glob.glob(os.path.join(lq_dir, '*')):
            if os.path.isdir(lq_path):
                continue
            
            filename = os.path.basename(lq_path)
            new_name = f"low_{filename}"
            os.rename(lq_path, os.path.join(lq_dir, new_name))
            print(f"Renamed LQ: {filename} -> {new_name}")
        
        # Rename GT files (e.g., '1.png' -> 'normal_1.png')
        for gt_path in glob.glob(os.path.join(gt_dir, '*')):
            if os.path.isdir(gt_path):
                continue
            
            filename = os.path.basename(gt_path)
            new_name = f"normal_{filename}"
            os.rename(gt_path, os.path.join(gt_dir, new_name))
            print(f"Renamed GT: {filename} -> {new_name}")

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
        lq_files = [f for f in os.listdir(lq_dir) if f.startswith('low_')]
        lq_files.sort()
        
        # Overwrite the list file
        with open(list_file, 'w') as f:
            for filename in lq_files:
                f.write(f"{filename}\n")
        
        print(f"Updated {list_file} with {len(lq_files)} entries")

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
        lq_files = [f for f in os.listdir(lq_dir) if f.startswith('low_')]
        lq_files.sort()
        
        # Overwrite the list file
        with open(list_file, 'w') as f:
            for filename in lq_files:
                f.write(f"{filename}\n")
        
        print(f"Updated {list_file} with {len(lq_files)} entries")

if __name__ == "__main__":
    # Example usage - point to your dataset folder
    dataset_root = "./data/dehazing/RESIDE-6k"  # CHANGE THIS
    
    # Step 1: Rename files
    rename_to_low_normal_structure(dataset_root)
    
    # Step 2: Update train.txt/test.txt
    update_list_files(dataset_root)
    
    print("Standardization complete! All files now use low_/normal_ prefixes.")