import numpy as np
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import types

from utils.data import ProjectDataset, transform_front, transform_tactile
from models.autoencoder import VisualTactileAutoencoder
from models.regressor import AngleRegressor


def main(config):
    dataset = ProjectDataset(
        "dataset/front",
        "dataset/tactile",
        transform_front=transform_front,
        transform_tactile=transform_tactile,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    vae = VisualTactileAutoencoder(
        config.vae_latent_dim, modality=config.vae_modality, lr=config.vae_lr
    )
    reg = AngleRegressor(config.vae_latent_dim, config.reg_hidden_dim, lr=config.reg_lr)


    # ===== Train VAE =====

    print("Start training VAE.")
    for epoch in range(config.vae_epochs):
        losses_total = []
        losses_kl = []
        losses_recon = []

        for front, tactile, angle in tqdm(dataloader):
            metrics = vae.train(front, tactile)

            losses_total.append(metrics["loss_total"])
            losses_kl.append(metrics["loss_kl"])
            losses_recon.append(metrics["loss_recon"])

        if config.use_wandb:
            input_image, recon_image = vae.report_images(front, tactile)
            wandb.log(
                {
                    "epoch": epoch,
                    "Loss/vae_total": np.average(losses_total),
                    "Loss/vae_kl": np.average(losses_kl),
                    "Loss/vae_recon": np.average(losses_recon),
                    "reconstruction": [
                        wandb.Image(input_image),
                        wandb.Image(recon_image),
                    ],
                }
            )

        print(
            "EPOCH: %d    Train Lower Bound: %lf (KL_loss: %lf. reconstruction_loss: %lf)"
            % (
                epoch + 1,
                np.average(losses_total),
                np.average(losses_kl),
                np.average(losses_recon),
            )
        )

    print("Finish training VAE.")

    # ===== Train Regressor =====
    print("Start training Regressor.")
    for epoch in range(config.reg_epochs):
        losses = []

        for front, tactile, angle in tqdm(dataloader):
            latent = vae.get_latent(front, tactile)
            metrics = reg.train(latent, angle)

            losses.append(metrics["loss"])

        if config.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "Loss/reg_mse": np.average(losses),
                }
            )

        print("EPOCH: %d    Train Loss: %lf" % (epoch + 1, np.average(losses)))

    print("Finish training MLP.")


if __name__ == "__main__":
    # load configs from yaml file
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        config = types.SimpleNamespace(**config)

    if config.use_wandb:
        wandb.init(project="wm-tactile")

    main(config)
