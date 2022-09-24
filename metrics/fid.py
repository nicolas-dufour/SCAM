from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
from torchvision.transforms import Resize
from pathlib import Path
from torchmetrics.metric import Metric
from tqdm import tqdm

from utils.utils import remap_image_torch


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    All credit to:
        https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:
        import scipy

        m = input.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        import scipy

        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


class NoTrainInceptionV3(FeatureExtractorInceptionV3):
    def __init__(
        self,
        name: str,
        features_list: List[str],
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(name, features_list, feature_extractor_weights_path)
        self.eval()
        self.num_out_feat = int(features_list[0])

    def train(self, mode):
        return super().train(False)

    def forward(self, x: Tensor) -> Tensor:
        x = remap_image_torch(x)
        out = super().forward(x)
        return out[0].reshape(x.shape[0], -1)


def compute_fid(mu_1, mu_2, sigma_1, sigma_2, eps=1e-6):
    mean_diff = mu_1 - mu_2

    mean_dist = mean_diff.dot(mean_diff)

    covmean = sqrtm(sigma_1.mm(sigma_2))

    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma_1.size(0), device=mu_1.device, dtype=mu_1.dtype) * eps
        covmean = sqrtm((sigma_1 + offset).mm(sigma_2 + offset))
    return (
        mean_dist
        + torch.trace(sigma_1)
        + torch.trace(sigma_2)
        - 2 * torch.trace(covmean)
    )


class FID(Metric):
    def __init__(
        self,
        feature_extractor,
        real_features_path,
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

        self.feature_extractor = feature_extractor

        mean_real = torch.load(f"{real_features_path}/mean.pt")
        sigma_real = torch.load(f"{real_features_path}/sigma.pt")

        self.add_state("mean_real", default=mean_real, dist_reduce_fx="mean")
        self.add_state("sigma_real", default=sigma_real, dist_reduce_fx="mean")

        self.add_state(
            "generated_features_sum",
            torch.zeros(self.feature_extractor.num_out_feat, dtype=torch.double),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "generated_features_cov_sum",
            torch.zeros(
                (
                    self.feature_extractor.num_out_feat,
                    self.feature_extractor.num_out_feat,
                ),
                dtype=torch.double,
            ),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "generated_features_num_samples",
            torch.tensor(0).long(),
            dist_reduce_fx="sum",
        )

    def update(self, images):
        features = self.feature_extractor(images).double()
        self.generated_features_sum += features.sum(dim=0)
        self.generated_features_cov_sum += features.t().mm(features)
        self.generated_features_num_samples += images.shape[0]

    def compute(self):
        mean_real = self.mean_real
        mean_generated = (
            self.generated_features_sum / self.generated_features_num_samples
        ).unsqueeze(dim=0)

        sigma_real = self.sigma_real
        sigma_generated = (
            self.generated_features_cov_sum
            - self.generated_features_num_samples
            * mean_generated.t().mm(mean_generated)
        ) / (self.generated_features_num_samples - 1)
        return compute_fid(mean_real, mean_generated[0], sigma_real, sigma_generated)


def compute_fid_for_dataset(dataloader, save_path, device="cpu"):
    feature_extractor = NoTrainInceptionV3(
        name="inception-v3-compat", features_list=[str(2048)]
    ).to(device)
    real_features_sum = torch.zeros(
        feature_extractor.num_out_feat, dtype=torch.double, device=device
    )
    real_features_cov_sum = torch.zeros(
        (
            feature_extractor.num_out_feat,
            feature_extractor.num_out_feat,
        ),
        dtype=torch.double,
        device=device,
    )
    real_features_num_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images, _ = batch
            images = images.to(device)
            features = feature_extractor(images).double()
            real_features_num_samples += features.shape[0]
            real_features_sum += features.sum(dim=0)
            real_features_cov_sum += features.t().mm(features)
    mean = (real_features_sum / real_features_num_samples).unsqueeze(0)
    sigma = (real_features_cov_sum - real_features_num_samples * mean.t().mm(mean)) / (
        real_features_num_samples - 1
    )
    mean = mean.squeeze(0)
    torch.save(mean.cpu(), Path(save_path) / Path("mean.pt"))
    torch.save(sigma.cpu(), Path(save_path) / Path("sigma.pt"))
