import torch
from torch import nn
import numpy as np
import enum
import torchvision

from models import nets
from utils.modality import Modality


class AIFDecoder:
    def __init__(
        self,
        z_dim: int = 1,
        lr=1e-3,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss(),
        device="cuda",
    ):
        self.device = device
        self.z_dim = z_dim
        self.conv_decoder = ConvDecoder().to(self.device)
        self.optimizer = optimizer(self.conv_decoder.parameters(), lr=lr)
        self.criterion = criterion
        self.best_loss = np.inf

    def train(self, latent, target_image):
        latent = latent.to(self.device)
        target_image = target_image.to(self.device)

        output_image = self.conv_decoder(latent)
        loss = self.criterion(output_image, target_image)

        self.conv_decoder.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {
            "loss": loss.cpu().detach().numpy(),
        }
        return metrics

    def report_images(self, latent, target_image):
        latent = latent.to(self.device)
        with torch.no_grad():
            output_images = self.conv_decoder(latent).cpu().detach()

        num_images = 8

        result_images = torch.cat(
            [target_image[:num_images], output_images[:num_images]], dim=0
        )

        grid = torchvision.utils.make_grid(
            result_images, nrow=2, padding=2, pad_value=0
        )

        return result_images

    def test(self, latent, target_image):
        latent = latent.to(self.device)
        target_image = target_image.to(self.device)

        with torch.no_grad():
            output_image = self.conv_decoder(latent)

        loss = self.criterion(output_image, target_image)

        metrics = {
            "loss": loss.cpu().detach().numpy(),
        }
        return metrics


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()

        # Two fully connected layers of neurons (feedforward architecture)
        self.ff_layers = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 7 * 10 * 16),  # 1120 neurons
            nn.ReLU(),
        )

        # Sequential upsampling using the deconvolutional layers & smoothing out checkerboard artifacts with conv layers
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(16, 64, 4, stride=2, padding=1),  # deconv1
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # conv1
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),  # deconv2
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # conv2
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # deconv3
            nn.Sigmoid(),  # Squeezing the output to 0-1 range
        )

    def forward(self, x):
        x = self.ff_layers(x)
        x = x.view(
            -1, 16, 7, 10
        )  # Reshaping the output of the fully connected layers so that it is compatible with the conv layers
        x = self.conv_layers(x)
        return x
