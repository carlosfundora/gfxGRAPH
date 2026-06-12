import torch
print("Imported torch")
from hipgraph_bridge.shape_bucketing import ShapeBucketPool
print("Imported ShapeBucketPool")

model = torch.nn.Linear(64, 64).cuda().eval()
def model_fn(x):
    print("model_fn called")
    return model(x)

pool = ShapeBucketPool(model_fn, buckets=[1, 4, 8, 16, 32])
x = torch.randn(1, 64, device="cuda")
print("calling pool")
try:
    pool(x)
except Exception as e:
    print(f"Exception: {e}")
print("pool done")
