
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedAudioMotorNet(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=16, alpha=0.01):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.shortcut = nn.Linear(input_dim, output_dim)
        self.alpha = alpha  # Hebbian strength

    def forward(self, x):
        cognitive = F.relu(self.fc1(x))
        cognitive = self.fc2(cognitive)
        shortcut = self.shortcut(x)
        return cognitive + shortcut, cognitive, shortcut

    def hebbian_update(self, x, y):
        # Hebbian rule: outer product
        
        hebb = torch.matmul(y.T, x) / x.shape[0]
        self.shortcut.weight.data += self.alpha * hebb
