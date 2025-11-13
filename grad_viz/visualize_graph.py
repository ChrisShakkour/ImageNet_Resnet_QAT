"""
visualize_graph.py
------------------
Visualize the forward/backward computation graph using torchviz.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

def visualize_graph(
    output: torch.Tensor,
    model: Optional[nn.Module] = None,
    params: Optional[Dict[str, torch.Tensor]] = None,
    filename: str = "backward_graph",
    fmt: str = "png"
):
    """
    Visualizes the computation graph (forward/backward) using torchviz.
    """
    try:
        from torchviz import make_dot
    except ImportError:
        raise ImportError("Please install torchviz: pip install torchviz")

    param_dict = {}
    if model:
        param_dict.update(dict(model.named_parameters()))
    if params:
        param_dict.update(params)

    dot = make_dot(output, params=param_dict)
    dot.render(filename, format=fmt)
    print(f"Computation graph saved as {filename}.{fmt}")


def main():
    # Example usage
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    x = torch.randn(5, 10, requires_grad=True)
    y = model(x)
    loss = y.mean()
    loss.backward()

    visualize_graph(loss, model=model, filename="computation_graph", fmt="png")

if __name__ == "__main__":
    main()