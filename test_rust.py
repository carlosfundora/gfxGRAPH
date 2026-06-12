import torch
import rs_gfxgraph
print("Instantiating router")
router = rs_gfxgraph.BucketRouter([1, 4, 8, 16, 32])
print("Router instantiated. Allocating memory...")
x = torch.randn(1, 64, device="cuda")
print("Memory allocated successfully")
