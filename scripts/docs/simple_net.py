# to showcase a simple torch.export
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(3, 6, 5)
        self.fc = nn.Linear(6 * 14 * 14, 10)
        self.register_buffer("mask", torch.randn(6, 14, 14) > 0.5)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = x * self.mask
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x) 
        return x
    

inputs = (torch.randn(1, 3, 32, 32), )
model = SimpleNet().eval()

ep = torch.export.export(model, inputs)
print(ep)
torch.export.save(ep, "simple_net.pt2")
