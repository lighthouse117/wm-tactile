import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class VisualTactileVAE(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(VisualTactileVAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 10 * 18, z_dim)
        self.fc_logvar = nn.Linear(128 * 10 * 18, z_dim)

        # Decoder
        self.fc_decode = nn.Linear(z_dim, 128 * 10 * 18)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=(1, 2), output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv_out = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def _encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def _sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decoder(self, z):
        x = F.relu(self.fc_decode(z))
        x = x.view(-1, 128, 10, 18)
        x = F.relu(self.deconv1(x))
        # print('1')
        # print(x.shape)
        x = F.relu(self.deconv2(x))
        # print('2')
        # print(x.shape)
        x = F.relu(self.deconv3(x))
        # print('3')
        # print(x.shape)
        x = torch.sigmoid(self.conv_out(x))
        return x

    def forward(self, x):
        mu, logvar = self._encoder(x)
        z = self._sample_z(mu, logvar)
        x_recon = self._decoder(z)
        return x_recon, mu, logvar

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._encoder(x)

        # KL loss(正則化項)の計算. mean, stdは (batch_size , z_dim)
        # torch.sumは上式のJ(=z_dim)に関するもの. torch.meanはbatch_sizeに関するものなので,
        # 上式には書いてありません.
        KL = -0.5 * torch.mean(
            torch.sum(1 + torch_log(std**2) - mean**2 - std**2, dim=1)
        )

        z = self._sample_z(mean, std)
        y = self._decoder(z)

        # reconstruction loss(負の再構成誤差)の計算. x, yともに (batch_size , 784)
        # torch.sumは上式のD(=784)に関するもの. torch.meanはbatch_sizeに関するもの.
        reconstruction = torch.mean(
            torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1)
        )

        return KL, -reconstruction

    def get_z(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self._encoder(x)
        z = self._sample_z(mu, logvar)
        return z

class VisualVAE(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(VisualVAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 10 * 10, z_dim)
        self.fc_logvar = nn.Linear(128 * 10 * 10, z_dim)

        # Decoder
        self.fc_decode = nn.Linear(z_dim, 128 * 10 * 10)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv_out = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def _encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def _sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decoder(self, z):
        x = F.relu(self.fc_decode(z))
        x = x.view(-1, 128, 10, 10)
        x = F.relu(self.deconv1(x))
        # print('1')
        # print(x.shape)
        x = F.relu(self.deconv2(x))
        # print('2')
        # print(x.shape)
        x = F.relu(self.deconv3(x))
        # print('3')
        # print(x.shape)
        x = torch.sigmoid(self.conv_out(x))
        return x

    def forward(self, x):
        mu, logvar = self._encoder(x)
        z = self._sample_z(mu, logvar)
        x_recon = self._decoder(z)
        return x_recon, mu, logvar

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._encoder(x)

        # KL loss(正則化項)の計算. mean, stdは (batch_size , z_dim)
        # torch.sumは上式のJ(=z_dim)に関するもの. torch.meanはbatch_sizeに関するものなので,
        # 上式には書いてありません.
        KL = -0.5 * torch.mean(
            torch.sum(1 + torch_log(std**2) - mean**2 - std**2, dim=1)
        )

        z = self._sample_z(mean, std)
        y = self._decoder(z)

        # reconstruction loss(負の再構成誤差)の計算. x, yともに (batch_size , 784)
        # torch.sumは上式のD(=784)に関するもの. torch.meanはbatch_sizeに関するもの.
        reconstruction = torch.mean(
            torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1)
        )

        return KL, -reconstruction

    def get_z(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self._encoder(x)
        z = self._sample_z(mu, logvar)
        return z


class TactileVAE(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(TactileVAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 10 * 8, z_dim)
        self.fc_logvar = nn.Linear(128 * 10 * 8, z_dim)

        # Decoder
        self.fc_decode = nn.Linear(z_dim, 128 * 10 * 8)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=(1, 2), output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv_out = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def _encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def _sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decoder(self, z):
        x = F.relu(self.fc_decode(z))
        x = x.view(-1, 128, 10, 8)
        x = F.relu(self.deconv1(x))
        # print('1')
        # print(x.shape)
        x = F.relu(self.deconv2(x))
        # print('2')
        # print(x.shape)
        x = F.relu(self.deconv3(x))
        # print('3')
        # print(x.shape)
        x = torch.sigmoid(self.conv_out(x))
        return x

    def forward(self, x):
        mu, logvar = self._encoder(x)
        z = self._sample_z(mu, logvar)
        x_recon = self._decoder(z)
        return x_recon, mu, logvar

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._encoder(x)

        # KL loss(正則化項)の計算. mean, stdは (batch_size , z_dim)
        # torch.sumは上式のJ(=z_dim)に関するもの. torch.meanはbatch_sizeに関するものなので,
        # 上式には書いてありません.
        KL = -0.5 * torch.mean(
            torch.sum(1 + torch_log(std**2) - mean**2 - std**2, dim=1)
        )

        z = self._sample_z(mean, std)
        y = self._decoder(z)

        # reconstruction loss(負の再構成誤差)の計算. x, yともに (batch_size , 784)
        # torch.sumは上式のD(=784)に関するもの. torch.meanはbatch_sizeに関するもの.
        reconstruction = torch.mean(
            torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1)
        )

        return KL, -reconstruction

    def get_z(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self._encoder(x)
        z = self._sample_z(mu, logvar)
        return z


def torch_log(x: torch.Tensor) -> torch.Tensor:
    """torch.log(0)によるnanを防ぐ"""
    return torch.log(torch.clamp(x, min=1e-10))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        out = self.fc4(h3)
        return out

    def loss(self, x, target) -> torch.Tensor:
        y = torch.squeeze(self.forward(x))
        target = torch.tensor(target, dtype=torch.float32)

        loss = nn.MSELoss()(y, target)
        return loss
