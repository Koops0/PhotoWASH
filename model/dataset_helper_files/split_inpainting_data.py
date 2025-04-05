import os
import random
import shutil
from pathlib import Path

def rename_and_split_dataset(data_root, train_ratio=0.8):
    """
    1. Renames files with 'normal_' (GT) and 'low_' (LQ) prefixes
    2. Splits into 80% train / 20% test sets
    """
    # Path setup
    gt_dir = os.path.join(data_root, 'GT')
    lq_dir = os.path.join(data_root, 'LQ')
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')
    
    # Create output directories
    os.makedirs(os.path.join(train_dir, 'GT'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'LQ'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'GT'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'LQ'), exist_ok=True)

    # Get all base filenames (without extensions)
    gt_files = {Path(f).stem: f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))}
    lq_files = {Path(f).stem: f for f in os.listdir(lq_dir) if os.path.isfile(os.path.join(lq_dir, f))}
    
    # Find common pairs (files present in both GT and LQ)
    common_files = set(gt_files.keys()) & set(lq_files.keys())
    common_files = sorted(list(common_files))
    random.shuffle(common_files)  # Shuffle for random split
    
    # Calculate split index
    split_idx = int(len(common_files) * train_ratio)
    train_files = common_files[:split_idx]
    test_files = common_files[split_idx:]
    
    print(f"Found {len(common_files)} image pairs")
    print(f"Training set: {len(train_files)} pairs")
    print(f"Test set: {len(test_files)} pairs")

    # Process and copy files
    def process_files(file_list, dest_dir):
        for base_name in file_list:
            # Original paths
            gt_original = os.path.join(gt_dir, gt_files[base_name])
            lq_original = os.path.join(lq_dir, lq_files[base_name])
            
            # New names with prefixes
            gt_new = f"normal_{gt_files[base_name]}"
            lq_new = f"low_{lq_files[base_name]}"
            
            # Copy with new names
            shutil.copy2(gt_original, os.path.join(dest_dir, 'GT', gt_new))
            shutil.copy2(lq_original, os.path.join(dest_dir, 'LQ', lq_new))
            
        # Create list file
        list_file = os.path.join(dest_dir, 'train.txt' if dest_dir == train_dir else 'test.txt')
        with open(list_file, 'w') as f:
            for base_name in file_list:
                f.write(f"low_{lq_files[base_name]}\n")  # List LQ files

    # Process both sets
    process_files(train_files, train_dir)
    process_files(test_files, test_dir)
    
    print(f"\nRenaming complete with prefixes:")
    print("- GT files: 'normal_*'")
    print("- LQ files: 'low_*'")
    print(f"\nDataset split into:")
    print(f"- Train: {len(train_files)} pairs ({train_ratio:.0%})")
    print(f"- Test: {len(test_files)} pairs ({1-train_ratio:.0%})")

if __name__ == "__main__":
    # Example usage
    dataset_root = "./data/inpainting"  # CHANGE THIS to your path
    rename_and_split_dataset(dataset_root)