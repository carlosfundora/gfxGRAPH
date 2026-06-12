import torch
print("Creating model...")
model = torch.nn.Linear(64, 64).cuda().eval()
print("Allocating 1x64...")
x = torch.randn(1, 64, device="cuda")
print("Memory allocated successfully")
