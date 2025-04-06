import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from image_ir.process_single import ImageRestoration

# Initialize dehazing restorer
dehazing_restorer = ImageRestoration(
    config_path="/mnt/c/Users/parsa/OneDrive/Desktop/CSCI 4220U/Project/Project_Repos/PhotoWASH/cycleRDM_fork/configs/inpainting.yml",
    model_path="/mnt/c/Users/parsa/OneDrive/Desktop/CSCI 4220U/Project/Project_Repos/PhotoWASH/cycleRDM_fork/pretrained-models/inpainting.pth.tar",
    output_dir="/mnt/c/Users/parsa/OneDrive/Desktop/CSCI 4220U/Project/Project_Repos/PhotoWASH/cycleRDM_fork/test_images_single/inpainting/output"
)

# Load test image
lq_image_path = "/mnt/c/Users/parsa/OneDrive/Desktop/CSCI 4220U/Project/Project_Repos/PhotoWASH/cycleRDM_fork/test_images_single/" \
"inpainting/input/low_08240.jpg"
print(f"Loading image from: {lq_image_path}")
lq_image = cv2.imread(lq_image_path)
if lq_image is None:
    raise FileNotFoundError(f"Could not load image from {lq_image_path}")
    
lq_image = cv2.cvtColor(lq_image, cv2.COLOR_BGR2RGB)

# Process low quality image
output_filename = "08240.jpg"  # Changed to .png to match restoration.py output
print(f"Processing image with shape {lq_image.shape}, output name: {output_filename}")

restored_image = dehazing_restorer.restore_image_data(
    lq_image=lq_image,
    output_filename=output_filename
)

#  Verify results
print(f"Input shape: {lq_image.shape}, Output shape: {restored_image.shape}")
print(f"Input dtype: {lq_image.dtype}, Output dtype: {restored_image.dtype}")
print(f"Processing complete!")