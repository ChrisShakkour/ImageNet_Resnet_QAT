"""
manual_grad_trace.py
--------------------
Compute and print gradients manually using torch.autograd.grad.
"""

import torch

def manual_grad_trace(output: torch.Tensor, inputs: torch.Tensor, create_graph: bool = False):
    """
    Computes gradients of output w.r.t. inputs using autograd.grad.
    Prints and returns the gradients.
    """
    grads = torch.autograd.grad(output, inputs, create_graph=create_graph)
    print(f"Manual gradients computed:\n{grads}")
    return grads

def main():
    # Example usage
    x = torch.randn(3, 3, requires_grad=True)
    y = x ** 2 + 3 * x + 2
    z = y.mean()

    print("Computing manual gradients of z w.r.t x:")
    manual_grad_trace(z, x)

if __name__ == "__main__":
    main()