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
import wandb

from tqdm import tqdm

from utils.data import ProjectDataset
from models.vae import ConvVAE


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
        plt.savefig("test.png")

    rng = np.random.RandomState(1234)
    random_state = 42

    device = "cuda"

    z_dim = 2048
    n_epochs = 10

    model = ConvVAE(z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    wandb.watch(model)

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
    wandb.init(project="wm-tactile")
    main()
