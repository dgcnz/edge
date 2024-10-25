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
        self.conv2 = nn.Conv2d(6, 9, 5)
        self.fc = nn.Linear(5184, 10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
