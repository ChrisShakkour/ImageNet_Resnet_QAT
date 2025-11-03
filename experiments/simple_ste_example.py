import os
import csv
import torch
from torch import nn
from torch.autograd import Function

N_BITS = 4
N_EPOCHS = 200 
ACC_TOL = 0.1  # tolerance for considering a regression prediction "correct"

class QuantizeSTE(Function):
    @staticmethod
    def forward(ctx, weight, bits=8):
        # symmetric uniform quantization
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax

        # compute scale from max absolute weight
        scale = weight.abs().max() / qmax
        scale = scale.clamp(min=1e-8)

        # quantize + dequantize
        w_q = torch.clamp(torch.round(weight / scale), qmin, qmax)
        return w_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass grad directly
        grad_weight = grad_output.clone()
        return grad_weight, None   # None for bits


class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.scale = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bits = bits

    def forward(self, x):
        # quantize weights only
        q_weight = QuantizeSTE.apply(self.weight, self.bits)
        return torch.matmul(x, q_weight.t()) + self.bias


class SimpleQuantNet(nn.Module):
    def __init__(self, bits=8):
        super().__init__()
        self.fc1 = QuantLinear(2, 4, bits)
        self.relu = nn.ReLU()
        self.fc2 = QuantLinear(4, 1, bits)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# tiny toy dataset: learn y = x1 + x2
torch.manual_seed(0)
x = torch.randn(100, 2)
y = x.sum(dim=1, keepdim=True)

model = SimpleQuantNet(bits=N_BITS)
optimizer = torch.optim.SGD(model.parameters(), lr=0.09)
criterion = nn.MSELoss()
# record losses per epoch for plotting
losses = []
# record accuracies per epoch (for this toy regression we define accuracy as
# fraction of predictions within ACC_TOL of the target)
accuracies = []

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        print(f"Epoch {epoch:03d} | Loss = {loss.item():.6f}")
    # append loss for plotting later
    losses.append(loss.item())
    # compute accuracy (fraction within tolerance)
    with torch.no_grad():
        preds = out.detach()
        correct = (preds.sub(y).abs() < ACC_TOL).sum().item()
        total = y.numel()
        acc = correct / total
        accuracies.append(acc)

# After training, save metrics and plot epochs vs loss, epochs vs accuracy,
# and accuracy vs loss (scatter). Also save CSV of metrics.
os.makedirs('logs', exist_ok=True)

# save CSV of metrics
csv_path = 'logs/metrics.csv'
try:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'accuracy'])
        for e, l, a in zip(range(1, len(losses) + 1), losses, accuracies):
            writer.writerow([e, l, a])
    print(f"Saved metrics CSV to {csv_path}")
except Exception as e:
    print(f"Could not save metrics CSV: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for headless environments
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(losses) + 1))

    # Epochs vs Loss
    plt.figure()
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epochs vs Loss')
    plt.grid(True)
    out_path = 'logs/epochs_vs_loss.png'
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved epochs vs loss plot to {out_path}")

    # Epochs vs Accuracy
    plt.figure()
    plt.plot(epochs, accuracies, marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Epochs vs Accuracy')
    plt.grid(True)
    out_path2 = 'logs/accuracy_vs_epochs.png'
    plt.tight_layout()
    plt.savefig(out_path2)
    print(f"Saved epochs vs accuracy plot to {out_path2}")

    # Accuracy vs Loss (scatter)
    plt.figure()
    plt.scatter(losses, accuracies, c=range(len(losses)), cmap='viridis')
    plt.colorbar(label='Epoch')
    plt.xlabel('Loss')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Loss')
    plt.grid(True)
    out_path3 = 'logs/accuracy_vs_loss.png'
    plt.tight_layout()
    plt.savefig(out_path3)
    print(f"Saved accuracy vs loss plot to {out_path3}")
except Exception as e:
    # don't break training scripts if matplotlib isn't available
    print(f"Could not create plots (matplotlib missing or error): {e}")
