import torch
g = torch.cuda.CUDAGraph()
x = torch.zeros(1, device='cuda')
with torch.cuda.graph(g):
    y = x * 2

pool = torch.cuda.graph_pool_handle()
x = torch.randn(1, 64, device='cuda')
print('OK')
