from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from typing import Tuple
from tqdm import tqdm


class ProjectDataset(Dataset):
    def __init__(
        self, front_dir, tactile_dir, transform_front=None, transform_tactile=None
    ):
        self.transform_front = transform_front
        self.transform_tactile = transform_tactile

        # Get all image paths from the specified directories
        self.front_paths = sorted(glob.glob(os.path.join(front_dir, "*.png")))
        self.tactile_paths = sorted(glob.glob(os.path.join(tactile_dir, "*.png")))

        # Ensure the number of images in both directories are the same
        assert len(self.front_paths) == len(self.tactile_paths)

    def __len__(self):
        return len(self.front_paths)

    def __getitem__(self, idx):
        # Load images
        front_img = Image.open(self.front_paths[idx])
        tactile_img = Image.open(self.tactile_paths[idx])

        # Extract label from the filename
        label = float(
            os.path.basename(self.front_paths[idx]).split("_")[-1].replace(".png", "")
        )

        # Apply transformations (if any)
        if self.transform_front or self.transform_tactile:
            front_img = self.transform_front(front_img)
            tactile_img = self.transform_tactile(tactile_img)

        return front_img, tactile_img, label


def torch_log(x: torch.Tensor) -> torch.Tensor:
    """torch.log(0)によるnanを防ぐ．"""
    return torch.log(torch.clamp(x, min=1e-10))


# VAEモデルの実装
class ConvVAE(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(ConvVAE, self).__init__()

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


def main():
    # Define any transformations you want to apply to the images
    transform_front = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(80),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize the tensor with mean and std for each channel
            # Add any other transformations you want
        ]
    )

    transform_tactile = transforms.Compose(
        [
            transforms.Resize((80, 60)),  # Convert image to grayscale
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensor with mean and std for each channel
            # Add any other transformations you want
        ]
    )

    # Create the dataset
    dataset = ProjectDataset(
        "dataset/front",
        "dataset/tactile",
        transform_front=transform_front,
        transform_tactile=transform_tactile,
    )

    # Create a data loader
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    front_imgs, tactile_imgs, labels = next(iter(dataloader))

    # Function to display images
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Display the first image of the batch
    print("Front Image:")
    imshow(front_imgs[0])
    print("Tactile Image:")
    imshow(tactile_imgs[0])
    print("Label:", labels[0])

    # x1 = next(iter(dataloader))
    # image_a = x1[1][0]

    # x2 = next(iter(dataloader))
    # image_b = x2[1][-1]

    # imshow(torch.abs(image_a - image_b))

    return

    rng = np.random.RandomState(1234)
    random_state = 42

    device = "cuda"

    z_dim = 2048
    n_epochs = 10

    model = ConvVAE(z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        losses = []
        KL_losses = []
        reconstruction_losses = []

        model.train()
        for x in tqdm(dataloader):
            x0 = x[0].to(device)
            x1 = x[1].to(device)

            x = torch.cat([x0, x1], dim=-1)

            # print(x.shape)

            model.zero_grad()

            # KL_loss, reconstruction_lossの各項の計算
            KL_loss, reconstruction_loss = model.loss(x)

            # エビデンス下界の最大化のためマイナス付きの各項の値を最小化するようにパラメータを更新
            loss = KL_loss + reconstruction_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())
            KL_losses.append(KL_loss.cpu().detach().numpy())
            reconstruction_losses.append(reconstruction_loss.cpu().detach().numpy())

        # losses_val = []
        # model.eval()
        # for x, t in dataloader_valid:

        #     x = x.to(device)

        #     KL_loss, reconstruction_loss = model.loss(x)

        #     loss = KL_loss + reconstruction_loss

        #     losses_val.append(loss.cpu().detach().numpy())

        # print('EPOCH: %d    Train Lower Bound: %lf (KL_loss: %lf. reconstruction_loss: %lf)    Valid Lower Bound: %lf' %
        #       (epoch+1, np.average(losses), np.average(KL_losses), np.average(reconstruction_losses), np.average(losses_val)))

        print(
            "EPOCH: %d    Train Lower Bound: %lf (KL_loss: %lf. reconstruction_loss: %lf)"
            % (
                epoch + 1,
                np.average(losses),
                np.average(KL_losses),
                np.average(reconstruction_losses),
            )
        )

        # モデルの状態を保存
        torch.save(model.state_dict(), "./model.pth")

        # オプティマイザの状態を保存
        torch.save(optimizer.state_dict(), "./optimizer.pth")


if __name__ == "__main__":
    main()
