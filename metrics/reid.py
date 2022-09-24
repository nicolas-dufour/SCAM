from typing import Any, Callable, List, Optional, Union
from torchmetrics.metric import Metric
import torch
from tqdm import tqdm
from metrics.models.fastreid.model import build_resnet_backbone
import torch.nn.functional as F


class ReidSubjectSuperiority(Metric):
    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.reid_model = build_resnet_backbone()

        self.add_state("more_similar_to_subject", torch.tensor(0), dist_reduce_fx="sum")

        self.add_state(
            "num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        swap_images,
        background_images,
        background_segmask,
        subject_images,
        subject_segmask,
    ):
        self.num_samples += swap_images.shape[0]
        _, _, h, w = swap_images.shape
        background_background_segmask = 1 - background_segmask[:, 0, :, :]
        h_max = (background_background_segmask.sum(dim=2) > 0).long().argmax(dim=1)
        w_max = (background_background_segmask.sum(dim=1) > 0).long().argmax(dim=1)

        h_rev = [i for i in range(background_background_segmask.shape[1])]

        h_min = h - (background_background_segmask.sum(dim=2) > 0)[
            :, h_rev[::-1]
        ].long().argmax(dim=1)
        w_rev = [i for i in range(background_background_segmask.shape[2])]
        w_min = w - (background_background_segmask.sum(dim=1) > 0)[
            :, w_rev[::-1]
        ].long().argmax(dim=1)
        background_bbox = torch.stack([h_max, h_min, w_max, w_min], dim=1)

        subject_segmask = 1 - subject_segmask[:, 0, :, :]
        h_max = (subject_segmask.sum(dim=2) > 0).long().argmax(dim=1).long()
        w_max = (subject_segmask.sum(dim=1) > 0).long().argmax(dim=1).long()
        h_min = h - (subject_segmask.sum(dim=2) > 0)[:, h_rev[::-1]].long().argmax(
            dim=1
        )
        w_min = w - (subject_segmask.sum(dim=1) > 0)[:, w_rev[::-1]].long().argmax(
            dim=1
        )
        subject_bbox = torch.stack([h_max, h_min, w_max, w_min], dim=1)

        swap_images_cropped = []

        subject_images_cropped = []

        background_images_cropped = []

        for i in range(swap_images.shape[0]):
            swap_image = swap_images[i]
            background_image = background_images[i]
            subject_image = subject_images[i]
            swap_image_bbox = background_bbox[i]
            subject_image_bbox = subject_bbox[i]

            swap_images_cropped.append(
                F.interpolate(
                    swap_image[
                        :,
                        swap_image_bbox[0] : swap_image_bbox[1],
                        swap_image_bbox[2] : swap_image_bbox[3],
                    ].unsqueeze(0),
                    size=(256, 128),
                    mode="bilinear",
                )
            )

            subject_images_cropped.append(
                F.interpolate(
                    subject_image[
                        :,
                        subject_image_bbox[0] : subject_image_bbox[1],
                        subject_image_bbox[2] : subject_image_bbox[3],
                    ].unsqueeze(0),
                    size=(256, 128),
                    mode="bilinear",
                )
            )
            background_images_cropped.append(
                F.interpolate(
                    background_image[
                        :,
                        swap_image_bbox[0] : swap_image_bbox[1],
                        swap_image_bbox[2] : swap_image_bbox[3],
                    ].unsqueeze(0),
                    size=(256, 128),
                    mode="bilinear",
                )
            )
        swap_images_cropped = torch.cat(swap_images_cropped, dim=0)
        subject_images_cropped = torch.cat(subject_images_cropped, dim=0)
        background_images_cropped = torch.cat(background_images_cropped, dim=0)

        swap_images_features = self.reid_model(swap_images_cropped).mean(dim=(2, 3))
        subject_images_features = self.reid_model(subject_images_cropped).mean(
            dim=(2, 3)
        )
        background_images_features = self.reid_model(background_images_cropped).mean(
            dim=(2, 3)
        )

        background_cos_sim = F.cosine_similarity(
            swap_images_features, background_images_features, dim=1
        )
        subject_cos_sim = F.cosine_similarity(
            swap_images_features, subject_images_features, dim=1
        )

        self.more_similar_to_subject += (subject_cos_sim > background_cos_sim).sum()

    def compute(self):
        return self.more_similar_to_subject / self.num_samples


class ReidSubjectSimilarity(Metric):
    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.reid_model = build_resnet_backbone()

        self.add_state(
            "subject_cosine_sim", torch.tensor(0).float(), dist_reduce_fx="sum"
        )

        self.add_state(
            "num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        swap_images,
        background_images,
        background_segmask,
        subject_images,
        subject_segmask,
    ):
        self.num_samples += swap_images.shape[0]
        _, _, h, w = swap_images.shape
        background_background_segmask = 1 - background_segmask[:, 0, :, :]
        h_max = (background_background_segmask.sum(dim=2) > 0).long().argmax(dim=1)
        w_max = (background_background_segmask.sum(dim=1) > 0).long().argmax(dim=1)

        h_rev = [i for i in range(background_background_segmask.shape[1])]

        h_min = h - (background_background_segmask.sum(dim=2) > 0)[
            :, h_rev[::-1]
        ].long().argmax(dim=1)
        w_rev = [i for i in range(background_background_segmask.shape[2])]
        w_min = w - (background_background_segmask.sum(dim=1) > 0)[
            :, w_rev[::-1]
        ].long().argmax(dim=1)
        background_bbox = torch.stack([h_max, h_min, w_max, w_min], dim=1)

        subject_segmask = 1 - subject_segmask[:, 0, :, :]
        h_max = (subject_segmask.sum(dim=2) > 0).long().argmax(dim=1)
        w_max = (subject_segmask.sum(dim=1) > 0).long().argmax(dim=1)
        h_min = h - (subject_segmask.sum(dim=2) > 0)[:, h_rev[::-1]].long().argmax(
            dim=1
        )
        w_min = w - (subject_segmask.sum(dim=1) > 0)[:, w_rev[::-1]].long().argmax(
            dim=1
        )
        subject_bbox = torch.stack([h_max, h_min, w_max, w_min], dim=1)

        swap_images_cropped = []

        subject_images_cropped = []

        for i in range(swap_images.shape[0]):
            swap_image = swap_images[i]
            subject_image = subject_images[i]
            swap_image_bbox = background_bbox[i]
            subject_image_bbox = subject_bbox[i]

            swap_images_cropped.append(
                F.interpolate(
                    swap_image[
                        :,
                        swap_image_bbox[0] : swap_image_bbox[1],
                        swap_image_bbox[2] : swap_image_bbox[3],
                    ].unsqueeze(0),
                    size=(256, 128),
                    mode="bilinear",
                )
            )
            subject_images_cropped.append(
                F.interpolate(
                    subject_image[
                        :,
                        subject_image_bbox[0] : subject_image_bbox[1],
                        subject_image_bbox[2] : subject_image_bbox[3],
                    ].unsqueeze(0),
                    size=(256, 128),
                    mode="bilinear",
                )
            )
        swap_images_cropped = torch.cat(swap_images_cropped, dim=0)
        subject_images_cropped = torch.cat(subject_images_cropped, dim=0)

        swap_images_features = self.reid_model(swap_images_cropped).mean(dim=(2, 3))
        subject_images_features = self.reid_model(subject_images_cropped).mean(
            dim=(2, 3)
        )

        subject_cos_sim = F.cosine_similarity(
            swap_images_features, subject_images_features, dim=1
        ).sum()

        self.subject_cosine_sim += subject_cos_sim

    def compute(self):
        return self.subject_cosine_sim / self.num_samples


class ReidBackgroundSuperiority(Metric):
    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.reid_model = build_resnet_backbone()

        self.add_state(
            "background_cosine_sim", torch.tensor(0).float(), dist_reduce_fx="sum"
        )

        self.add_state(
            "num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        swap_images,
        background_images,
        background_segmask,
        subject_images,
        subject_segmask,
    ):
        self.num_samples += swap_images.shape[0]
        _, _, h, w = swap_images.shape
        background_background_segmask = 1 - background_segmask[:, 0, :, :]
        h_max = (background_background_segmask.sum(dim=2) > 0).long().argmax(dim=1)
        w_max = (background_background_segmask.sum(dim=1) > 0).long().argmax(dim=1)

        h_rev = [i for i in range(background_background_segmask.shape[1])]

        h_min = h - (background_background_segmask.sum(dim=2) > 0)[
            :, h_rev[::-1]
        ].long().argmax(dim=1)
        w_rev = [i for i in range(background_background_segmask.shape[2])]
        w_min = w - (background_background_segmask.sum(dim=1) > 0)[
            :, w_rev[::-1]
        ].long().argmax(dim=1)
        background_bbox = torch.stack([h_max, h_min, w_max, w_min], dim=1)

        swap_images_cropped = []

        background_images_cropped = []

        for i in range(swap_images.shape[0]):
            swap_image = swap_images[i]
            background_image = background_images[i]
            swap_image_bbox = background_bbox[i]

            swap_images_cropped.append(
                F.interpolate(
                    swap_image[
                        :,
                        swap_image_bbox[0] : swap_image_bbox[1],
                        swap_image_bbox[2] : swap_image_bbox[3],
                    ].unsqueeze(0),
                    size=(256, 128),
                    mode="bilinear",
                )
            )
            background_images_cropped.append(
                F.interpolate(
                    background_image[
                        :,
                        swap_image_bbox[0] : swap_image_bbox[1],
                        swap_image_bbox[2] : swap_image_bbox[3],
                    ].unsqueeze(0),
                    size=(256, 128),
                    mode="bilinear",
                )
            )
        swap_images_cropped = torch.cat(swap_images_cropped, dim=0)
        background_images_cropped = torch.cat(background_images_cropped, dim=0)

        swap_images_features = self.reid_model(swap_images_cropped).mean(dim=(2, 3))
        background_images_features = self.reid_model(background_images_cropped).mean(
            dim=(2, 3)
        )

        background_cos_sim = F.cosine_similarity(
            swap_images_features, background_images_features, dim=1
        ).sum()

        self.background_cosine_sim += background_cos_sim

    def compute(self):
        return self.background_cosine_sim / self.num_samples
