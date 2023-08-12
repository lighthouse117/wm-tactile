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

from utils.data import TactileRGBDataset, transform_tactile
from models.aif_decoder import AIFDecoder
from utils.modality import Modality
from utils.images import imsave_numpy, imsave_pil, imsave_torch, UnNormalize


def main(config):
    # Create result folder
    exp_id = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/" + exp_id, exist_ok=True)

    # ===== Load Dataset =====
    dataset_path = f"dataset/{config.vision_view}-view"
    dataset = TactileRGBDataset(
        dataset_path,
        transform=transform_tactile,
        bg_path=dataset_path + "/tactile_bg.png",
        remove_bg=config.remove_tactile_bg,
    )

    # Split dataset into train and test
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

    # Model
    decoder = AIFDecoder(
        lr=config.aif_decoder_lr,
    )

    # unnorm = UnNormalize(mean=0.5, std=0.5)

    # front, tactile, angle = next(iter(train_dataloader))
    # imsave_torch(tactile, "results/" + exp_id + "/tactile.png")
    # imsave_torch(front, "results/" + exp_id + "/front.png")
    # return

    # ===== Train Decoder =====

    print("Start training decoder.")
    for epoch in range(config.aif_decoder_epochs):
        losses = []

        for angles, tactiles in tqdm(train_dataloader):
            metrics = decoder.train(angles, tactiles)
            losses.append(metrics["loss"])

        if config.use_wandb:
            result_images = decoder.report_images(angles, tactiles)
            wandb.log(
                {
                    "epoch": epoch,
                    "Train/loss_decoder_mse": np.average(losses),
                    "Train/images": [
                        wandb.Image(
                            result_images, caption="top: target, bottom: output"
                        ),
                    ],
                }
            )

        result_images = decoder.report_images(angles, tactiles)
        imsave_torch(
            result_images, "results/" + exp_id + "/epoch_" + str(epoch) + ".png"
        )

        print("EPOCH: %d    Train Loss: %lf" % (epoch + 1, np.average(losses)))

        # Save model if the loss is lower than the best
        if np.average(losses) < decoder.best_loss:
            decoder.best_loss = np.average(losses)
            torch.save(
                decoder.conv_decoder.state_dict(),
                "results/" + exp_id + "/aif_decoder.pth",
            )
            print("Saved aif_decoder.pth")

    print("Finish training decoder.")

    # ===== Test =====
    print("Start testing.")
    test_losses = []

    for angles, tactiles in tqdm(test_dataloader):
        metrics = decoder.test(angles, tactiles)
        test_losses.append(metrics["loss"])

    if config.use_wandb:
        result_images = decoder.report_images(angles, tactiles)
        wandb.log(
            {
                "Test/loss_decoder_mse": np.average(test_losses),
                "Test/images": [
                    wandb.Image(result_images, caption="top: target, bottom: output"),
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
        wandb.init(project="tactile-aif")
        wandb.config.update(config)

    main(config)
