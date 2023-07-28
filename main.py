import numpy as np
import wandb
import yaml
import torch
import types
import pprint
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime, timezone, timedelta

from utils.data import ProjectDataset, transform_front, transform_tactile
from models.autoencoder import FusionAutoencoder
from models.regressor import AngleRegressor
from utils.modality import Modality
from utils.images import imsave_numpy, imsave_pil, imsave_torch, UnNormalize


def main(config):
    # Create result folder
    exp_id = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/" + exp_id, exist_ok=True)

    # ===== Load Dataset =====

    dataset_path = f"dataset/{config.vision_view}-view"

    dataset = ProjectDataset(
        dataset_path + "/front",
        dataset_path + "/tactile",
        transform_front=transform_front,
        transform_tactile=transform_tactile,
    )

    train_size = int(len(dataset) * 0.8)

    train_dataset, test_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=True
    )

    vae = FusionAutoencoder(
        config.vae_latent_dim,
        modality=Modality(config.vae_modality),
        lr=config.vae_lr,
    )
    reg = AngleRegressor(
        config.vae_latent_dim,
        config.reg_hidden_dim,
        lr=config.reg_lr,
    )

    unnorm = UnNormalize(mean=0.5, std=0.5)

    # front, tactile, angle = next(iter(train_dataloader))
    # imsave_torch(tactile, "results/" + exp_id + "/tactile.png")
    # imsave_torch(front, "results/" + exp_id + "/front.png")
    # return

    # ===== Train VAE =====

    print("Start training VAE.")
    for epoch in range(config.vae_epochs):
        losses_total = []
        losses_kl = []
        losses_recon = []

        for front, tactile, angle in tqdm(train_dataloader):
            metrics = vae.train(front, tactile)

            losses_total.append(metrics["loss_total"])
            losses_kl.append(metrics["loss_kl"])
            losses_recon.append(metrics["loss_recon"])

        if config.use_wandb:
            input_image, recon_image = vae.report_images(front, tactile)
            wandb.log(
                {
                    "epoch": epoch,
                    "Train/loss_vae_total": np.average(losses_total),
                    "Train/loss_vae_kl": np.average(losses_kl),
                    "Train/loss_vae_recon": np.average(losses_recon),
                    "Train/reconstruction": [
                        wandb.Image(input_image),
                        wandb.Image(recon_image),
                        # wandb.Image(unnorm(input_image)),
                        # wandb.Image(unnorm(recon_image)),
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

        # Save model if the loss is lower than the best
        if np.average(losses_total) < vae.best_loss:
            vae.best_loss = np.average(losses_total)
            torch.save(vae.vae.state_dict(), "results/" + exp_id + "/vae.pth")
            print("Saved vae.pth")

    print("Finish training VAE.")

    # ===== Train Regressor =====
    print("Start training Regressor.")
    for epoch in range(epoch + 1, epoch + 1 + config.reg_epochs):
        losses = []

        for front, tactile, angle in tqdm(train_dataloader):
            latent = vae.get_latent(front, tactile)
            metrics = reg.train(latent, angle)

            losses.append(metrics["loss"])

        if config.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "Train/loss_reg_mse": np.average(losses),
                }
            )

        print("EPOCH: %d    Train Loss: %lf" % (epoch + 1, np.average(losses)))

        # Save model if the loss is lower than the best
        if np.average(losses) < reg.best_loss:
            reg.best_loss = np.average(losses)
            torch.save(reg.mlp.state_dict(), "results/" + exp_id + "/reg.pth")
            print("Saved reg.pth")

    print("Finish training MLP.")

    # ===== Test =====
    print("Start testing.")
    test_losses = []
    test_accs = []

    for front, tactile, angle in tqdm(test_dataloader):
        latent = vae.get_latent(front, tactile)
        metrics = reg.test(latent, angle)

        test_losses.append(metrics["loss"])

    if config.use_wandb:
        input_image, recon_image = vae.report_images(front, tactile)
        wandb.log(
            {
                "Test/loss_reg_mse": np.average(test_losses),
                "Test/reconstruction": [
                    wandb.Image(input_image),
                    wandb.Image(recon_image),
                    # wandb.Image(unnorm(input_image)),
                    # wandb.Image(unnorm(recon_image)),
                ],
            }
        )

    print("Test Loss: %lf" % np.average(test_losses))


if __name__ == "__main__":
    # load configs from yaml file
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        pprint.pprint(config)

    config = types.SimpleNamespace(**config)

    if config.use_wandb:
        wandb.init(project="wm-tactile")
        wandb.config.update(config)

    main(config)
