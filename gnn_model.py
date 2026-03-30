
import torch
import torch.nn as nn

class DynamicGNN(nn.Module):
    def __init__(self, nodes=5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(nodes, nodes))

    def forward(self, x):
        return torch.matmul(x, self.W)

    def strengthen_edge(self, i, j, amount=0.01):
        self.W.data[i, j] += amount
