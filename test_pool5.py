import torch
print("Calling pool handle")
pool = torch.cuda.graph_pool_handle()
print("Pool handle called, allocating memory...")
x = torch.zeros(1, 64, device="cuda")
print("Memory allocated successfully")
