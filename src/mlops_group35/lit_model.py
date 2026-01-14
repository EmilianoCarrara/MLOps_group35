import logging
import lightning as L
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


from mlops_group35.model import Model

logger = logging.getLogger(__name__)


class LitModel(L.LightningModule):
    def __init__(self, hidden_dim: int = 32, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model(hidden_dim)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


def train():
    # Fake data
    x = torch.rand(1000, 1)
    y = 2 * x + 0.5
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    model = LitModel(hidden_dim=32, lr=1e-3)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    train()
