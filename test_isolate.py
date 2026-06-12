import torch
from hipgraph_bridge.shape_bucketing import ShapeBucketPool
model = torch.nn.Linear(64, 64).cuda().eval()
pool = ShapeBucketPool(lambda x: model(x), buckets=[1, 4, 8, 16, 32])
print("Allocating 1x64...")
x = torch.randn(1, 64, device="cuda")
print("Running pool...")
out = pool(x)
print("Allocating 3x64...")
x = torch.randn(3, 64, device="cuda")
print("Running pool...")
out = pool(x)
print("Done")
