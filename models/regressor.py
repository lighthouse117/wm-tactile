import torch
import numpy as np

from models import nets


class AngleRegressor:
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        lr=1e-3,
        optimizer=torch.optim.Adam,
        device="cuda",
    ):
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.mlp = nets.MLP(z_dim, hidden_dim, 1).to(device)
        self.optimizer = optimizer(self.mlp.parameters(), lr=lr)

        self.best_loss = np.inf

    def train(self, latent, angle):
        angle = angle.to(self.device)
        loss = self.mlp.loss(latent, angle)

        self.mlp.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {
            "loss": loss.cpu().detach().numpy(),
        }
        return metrics

    def test(self, latent, angle):
        angle = angle.to(self.device)
        loss = self.mlp.loss(latent, angle)

        metrics = {
            "loss": loss.cpu().detach().numpy(),
        }
        return metrics