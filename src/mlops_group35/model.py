import torch
from torch import nn

"""Neural network model definitions"""


class Model(nn.Module):

    """Simple MLP regressor for 1D input → 1D output."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure shape [batch, 1]
        if x.ndim == 1:
            x = x.unsqueeze(1)
        return self.net(x)


if __name__ == "__main__":
    model = Model()
    x = torch.rand(4, 1)
    y = model(x)
    print(f"Input shape: {x.shape} -> Output shape: {y.shape}")
