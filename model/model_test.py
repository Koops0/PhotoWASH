import torch
checkpoint = torch.load("pretrained-models/dehazing.pth.tar")  # Safer mode
print(checkpoint["config"])  # Compare with your YAML