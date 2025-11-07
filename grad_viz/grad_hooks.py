"""
grad_hooks.py
--------------
Attach hooks to PyTorch layers and tensors to inspect gradients
during the backward pass.
"""

import torch
import torch.nn as nn

def register_grad_hooks(model: nn.Module):
    """
    Registers backward hooks on each layer to print gradient shapes
    as they propagate backward.
    """

    def backward_hook(module, grad_input, grad_output):
        print(f"\n==> Backward through: {module.__class__.__name__}")
        print(f"grad_input: {[g.shape if g is not None else None for g in grad_input]}")
        print(f"grad_output: {[g.shape if g is not None else None for g in grad_output]}")

    for layer in model.modules():
        if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and layer != model:
            layer.register_full_backward_hook(backward_hook)

    print("Gradient hooks registered on model.")

def register_tensor_hook(tensor: torch.Tensor, name: str = "tensor"):
    """
    Registers a hook on a tensor to print its gradient during backward.
    """
    def print_grad(grad):
        print(f"Gradient for {name}: {grad}")

    tensor.register_hook(print_grad)
    print(f"Hook registered for tensor '{name}'.")


def main():
    # Example usage
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    register_grad_hooks(model)

    x = torch.randn(5, 10, requires_grad=True)
    y = model(x)
    loss = y.mean()
    loss.backward()


if __name__ == "__main__":
    main()