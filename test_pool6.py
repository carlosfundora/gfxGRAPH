import torch
print("Capturing first graph")
g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    torch.zeros(1, 64, device="cuda")
print("Getting pool from g1")
pool = g1.pool()
print("Capturing second graph with g1.pool()")
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2, pool=pool):
    torch.zeros(1, 64, device="cuda")
print("Allocating eager memory...")
x = torch.zeros(1, 64, device="cuda")
print("Memory allocated successfully")
