import torch
def f(x):
    return x + 1


x = torch.randn(1, 3, 224, 224)
m = torch.compile(f)
print(m(x)) 