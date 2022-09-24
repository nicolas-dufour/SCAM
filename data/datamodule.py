import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
from .image_dataset import ImageDataset
import albumentations as A
from hydra.utils import instantiate


class ImageDataModule(pl.LightningDataModule):
    """
    Module to load image data
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):

        train_transforms = instantiate(self.cfg.train_aug)
        val_transforms = instantiate(self.cfg.val_aug)
        train_path = Path(self.cfg.path) / Path("train")
        test_path = Path(self.cfg.path) / Path("test")

        self.train_dataset = ImageDataset(
            train_path,
            self.cfg.num_labels_orig,
            train_transforms,
            image_extension=self.cfg.image_extension,
            label_merge_strat=self.cfg.label_merge_strat,
        )
        self.test_dataset = ImageDataset(
            test_path,
            self.cfg.num_labels_orig,
            val_transforms,
            image_extension=self.cfg.image_extension,
            label_merge_strat=self.cfg.label_merge_strat,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
        )
