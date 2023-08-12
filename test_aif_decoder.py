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
    MODEL_PATH = "results/" + "20230812_124826" + "/aif_decoder.pth"

    decoder = AIFDecoder()

    # Load parameters
    decoder.conv_decoder.load_state_dict(torch.load(MODEL_PATH))
    print("Loaded model from", MODEL_PATH)

    # TEST_ANGLES = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    TEST_ANGLES = [0.0, -15.0, -30.0, -45.0, -60.0, -75.0, -90.0]

    input = torch.tensor(TEST_ANGLES).unsqueeze(1)
    print(input.shape)
    output = decoder.test_images(input)

    imsave_torch(output, "test.png")

    if config.use_wandb:
        wandb_images = []
        for i, angle in enumerate(TEST_ANGLES):
            wandb_images.append(wandb.Image(output[i], caption=f"{angle:.1f}"))
        wandb.log({"Test": wandb_images})


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
