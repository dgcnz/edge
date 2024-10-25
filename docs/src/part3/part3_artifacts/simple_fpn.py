import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """
    Just a simple network
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 5)
        self.fc = nn.Linear(6 * 14 * 14, 10)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
