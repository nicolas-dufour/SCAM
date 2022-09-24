from torch.utils.data import Dataset
import torch
import os
from pathlib import Path
import PIL
from PIL import Image
import numpy as np
import torch.nn.functional as F


class ImageDataset(Dataset):
    """
    Image Dataset in jpg format
    """

    def __init__(
        self,
        path,
        num_labels,
        transforms=None,
        image_extension="jpg",
        label_merge_strat="none",
    ):
        super().__init__()
        self.image_dir = Path(path) / Path("images")
        self.labels = Path(path) / Path("labels")

        self.image_extension = image_extension

        images_list = sorted(os.listdir(self.image_dir))

        self.image_list_filtered = []

        for image_name in images_list:
            if image_name.endswith(self.image_extension):
                self.image_list_filtered.append(image_name.split(".")[0])

        self.transforms = transforms
        self.num_labels = num_labels
        self.label_merge_strat = label_merge_strat

    def __getitem__(self, index):
        image_name = self.image_list_filtered[index]
        image = np.array(
            Image.open(
                self.image_dir / Path(f"{image_name}.{self.image_extension}")
            ).convert("RGB")
        )
        segmentation_mask = np.array(
            Image.open(self.labels / Path(f"{image_name}.png")).resize(
                image.shape[:-1][::-1], resample=Image.NEAREST
            )
        )
        if self.transforms:
            augmented = self.transforms(image=image, mask=segmentation_mask)
            image = augmented["image"]
            segmentation_mask = augmented["mask"]
        segmentation_mask = self.onehot_encode_labels(segmentation_mask)
        if self.label_merge_strat == "body_background":
            background = segmentation_mask[0]
            body = segmentation_mask[1:].max(dim=0).values
            segmentation_mask = torch.stack([background, body], dim=0)

        elif self.label_merge_strat == "body_face_background":
            background = segmentation_mask[0]
            face = (
                torch.stack(
                    [
                        segmentation_mask[1],
                        segmentation_mask[2],
                        segmentation_mask[4],
                        segmentation_mask[13],
                    ],
                    dim=0,
                )
                .max(dim=0)
                .values
            )
            segmentation_mask[4] = torch.zeros_like(segmentation_mask[4])
            segmentation_mask[13] = torch.zeros_like(segmentation_mask[13])
            body = segmentation_mask[3:].max(dim=0).values
            segmentation_mask = torch.stack([background, face, body], dim=0)
        return image, segmentation_mask

    def __len__(self):
        return len(self.image_list_filtered)

    def onehot_encode_labels(self, labels):
        height, width = labels.shape
        labels = labels.unsqueeze(0).long()
        input_label = torch.FloatTensor(self.num_labels, height, width).zero_()
        input_semantics = input_label.scatter_(0, labels, 1.0)
        return input_semantics
