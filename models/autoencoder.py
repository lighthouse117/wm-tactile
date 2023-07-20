import torch
import numpy as np

from models import nets


class VisualTactileAutoencoder:
    def __init__(
        self,
        z_dim: int,
        modality="concat",
        lr=1e-3,
        optimizer=torch.optim.Adam,
        device="cuda",
    ):
        self.z_dim = z_dim
        self.modality = modality
        self.device = device

        self.vae = nets.ConvVAE(z_dim).to(device)
        self.optimizer = optimizer(self.vae.parameters(), lr=lr)

    def train(self, front, tactile):
        x = self._input(front, tactile)

        # KL_loss, reconstruction_lossの各項の計算
        KL_loss, reconstruction_loss = self.vae.loss(x)

        # エビデンス下界の最大化のためマイナス付きの各項の値を最小化するようにパラメータを更新
        loss = KL_loss + reconstruction_loss

        self.vae.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {
            "loss_total": loss.cpu().detach().numpy(),
            "loss_kl": KL_loss.cpu().detach().numpy(),
            "loss_recon": reconstruction_loss.cpu().detach().numpy(),
        }

        return metrics

    def get_latent(self, front, tactile):
        x = self._input(front, tactile)

        with torch.no_grad():
            z = self.vae.get_z(x)

        return z

    def report_images(self, front, tactile):
        x = self._input(front, tactile)

        with torch.no_grad():
            recon, _, _ = self.vae(x)

        # Returns input images and reconstructed images
        input_image = np.transpose(x[0].cpu().detach().numpy(), (1, 2, 0))
        recon_image = np.transpose(recon[0].cpu().detach().numpy(), (1, 2, 0))

        return input_image, recon_image

    def _input(self, front, tactile):
        front = front.to(self.device)
        tactile = tactile.to(self.device)

        if self.modality == "concat":
            x = torch.cat([front, tactile], dim=-1)
        elif self.modality == "vision":
            x = front
        elif self.modality == "tactile":
            x = tactile
        else:
            raise NotImplementedError

        return x
