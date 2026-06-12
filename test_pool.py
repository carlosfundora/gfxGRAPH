import torch

def test():
    model = torch.nn.Linear(64, 64).cuda()
    x = torch.randn(1, 64, device="cuda")

    pool = torch.cuda.graph_pool_handle()
    print("Pool:", pool)

    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool):
        out1 = model(x)

    print("Graph 1 captured")
    x2 = torch.randn(4, 64, device="cuda")
    print("x2 allocated")

test()
print("Done")
