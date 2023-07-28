from PIL import Image
import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

from utils.images import SubtractTactileBG, AdjustBrightnessAndContrast, MinMaxNormalize


transform_front = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop(80),
        # transforms.CenterCrop(100),
        # transforms.Resize((80, 80)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)

transform_tactile = transforms.Compose(
    [
        transforms.ToTensor(),
        SubtractTactileBG("dataset/tactile_bg.png"),
        MinMaxNormalize(),
        AdjustBrightnessAndContrast(brightness_factor=2, contrast_factor=3),
        transforms.Resize((80, 60)),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


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
        print(f"Found {len(self.front_paths)} front images.")
        print(f"Found {len(self.tactile_paths)} tactile images.")
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
