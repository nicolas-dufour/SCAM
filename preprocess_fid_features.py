from metrics.fid import compute_fid_features
from data.datamodule import ImageDataModule
import hydra
from pathlib import Path


@hydra.main(config_path="configs", config_name="config")
def compute_fid_for_dataset(cfg):
    datamodule = ImageDataModule(cfg.dataset)
    datamodule.setup()
    train_path = Path(cfg.dataset.path) / Path("train/stats")
    train_path.mkdir(parents=True, exist_ok=True)
    print(f"Computing FID features for train set and saving to {train_path}")
    compute_fid_features(datamodule.train_dataloader(), train_path, device=cfg.compnode.accelerator)

    test_path = Path(cfg.dataset.path) / Path("test/stats")
    test_path.mkdir(parents=True, exist_ok=True)
    print(f"Computing FID features for test set and saving to {test_path}")
    compute_fid_features(datamodule.test_dataloader(), test_path, device=cfg.compnode.accelerator)

if __name__ == "__main__":
    compute_fid_for_dataset()