from re import T
from data.datamodule import ImageDataModule
from gan import Geppetto
import hydra
import shutil
import wandb
import os
from utils.callbacks import GANImageLog, StopIfNaN, LogAttention
from pytorch_lightning.utilities.rank_zero import _get_rank

from pathlib import Path

from omegaconf import OmegaConf


@hydra.main(config_path="configs", config_name="config")
def run(cfg):

    # print(OmegaConf.to_yaml(cfg, resolve=True))

    dict_config = OmegaConf.to_container(cfg, resolve=True)

    Path(cfg.checkpoints.dirpath).mkdir(parents=True, exist_ok=True)

    shutil.copyfile(".hydra/config.yaml", f"{cfg.checkpoints.dirpath}/config.yaml")

    log_dict = {}

    log_dict["model"] = dict_config["model"]

    log_dict["dataset"] = dict_config["dataset"]

    datamodule = ImageDataModule(cfg.dataset)

    # logger.log_hyperparams(dict_config)

    checkpoint_callback = hydra.utils.instantiate(cfg.checkpoints)

    image_log_callback = GANImageLog()

    stop_if_nan_callback = StopIfNaN(
        ["train/gen_loss_step", "train/disc_loss_step"]
    )

    progress_bar = hydra.utils.instantiate(cfg.progress_bar)
    
    callbacks = [
        checkpoint_callback,
        image_log_callback,
        stop_if_nan_callback,
        progress_bar,
    ]
    if cfg.model.name == "SCAM":
        callbacks.append(LogAttention())

    rank = _get_rank()

    if os.path.isfile(Path(cfg.checkpoints.dirpath) / Path("wandb_id.txt")):
        with open(
            Path(cfg.checkpoints.dirpath) / Path("wandb_id.txt"), "r"
        ) as wandb_id_file:
            wandb_id = wandb_id_file.readline()
    else:
        wandb_id = wandb.util.generate_id()
        print(f"generated id{wandb_id}")
        if rank == 0:
            with open(
                Path(cfg.checkpoints.dirpath) / Path("wandb_id.txt"), "w"
            ) as wandb_id_file:
                wandb_id_file.write(str(wandb_id))

    if (Path(cfg.checkpoints.dirpath) / Path("last.ckpt")).exists():

        print("Loading checkpoints")
        checkpoint_path = Path(cfg.checkpoints.dirpath) / Path("last.ckpt")

        logger = hydra.utils.instantiate(cfg.logger, id=wandb_id, resume="allow")
        model = Geppetto.load_from_checkpoint(checkpoint_path, cfg=cfg.model)
        logger.watch(model)
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            strategy=cfg.trainer.strategy,
            logger=logger,
            callbacks=callbacks,
            resume_from_checkpoint=str(checkpoint_path),
        )
    else:
        logger = hydra.utils.instantiate(cfg.logger, id=wandb_id, resume="allow")
        logger._wandb_init.update({"config": log_dict})
        model = Geppetto(cfg.model)
        logger.watch(model)
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            strategy=cfg.trainer.strategy,
            logger=logger,
            callbacks=callbacks,
        )
    # trainer.fit_loop.epoch_loop.batch_loop.connect(optimizer_loop=YieldLoop())

    trainer.fit(model, datamodule)

    trainer.test(model, dataloaders=datamodule, ckpt_path="best")


if __name__ == "__main__":
    run()
