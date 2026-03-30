
import torch
import torch.optim as optim
from advanced_model import AdvancedAudioMotorNet

def generate_data(n=1000, input_dim=32, output_dim=16):
    X = torch.randn(n, input_dim)
    W_true = torch.randn(input_dim, output_dim)
    Y = X @ W_true
    return X, Y

def train():
    model = AdvancedAudioMotorNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    X, Y = generate_data()
    history = []

    for epoch in range(300):
        optimizer.zero_grad()
        pred, cognitive, shortcut = model(X)

        energy = cognitive.norm()
        loss = loss_fn(pred, Y) + 0.001 * energy

        loss.backward()
        optimizer.step()

        model.hebbian_update(X, pred.detach())

        with torch.no_grad():
            shortcut_norm = model.shortcut.weight.norm().item()
            cognitive_norm = model.fc1.weight.norm().item()

        history.append((loss.item(), shortcut_norm, cognitive_norm))

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    return history

if __name__ == "__main__":
    train()
