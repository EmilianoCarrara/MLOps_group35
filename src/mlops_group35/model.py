import logging

import torch
from torch import nn

"""Neural network model definitions"""


logger = logging.getLogger(__name__)


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
        logger.debug("Initialized Model with hidden_dim=%d", hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure shape [batch, 1]
        if x.ndim == 1:
            logger.debug("Input tensor was 1D (shape=%s). Unsqueezing to [batch, 1].", tuple(x.shape))
            x = x.unsqueeze(1)
        return self.net(x)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Running Model smoke test (__main__)")
    model = Model()
    x = torch.rand(4, 1)
    logger.info("Created input tensor with shape=%s dtype=%s device=%s", tuple(x.shape), x.dtype, x.device)

    y = model(x)
    logger.info("Model forward pass completed. Output shape=%s dtype=%s device=%s", tuple(y.shape), y.dtype, y.device)

    print(f"Input shape: {x.shape} -> Output shape: {y.shape}")
