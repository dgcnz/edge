import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """
    Just a simple network
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.fc = nn.Linear(4704, 10)

    def forward(self, x: torch.Tensor):
        z = self.conv1(x)
        z = F.relu(z)
        y = self.conv2(x)
        y = F.relu(y)
        o = z + y
        o = torch.flatten(o, 1)
        o = self.fc(o)
        return o
    