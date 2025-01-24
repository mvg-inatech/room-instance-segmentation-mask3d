import logging
import os
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import torch
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from utils.utils import (
    flatten_dict,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning import Trainer, seed_everything

logger = logging.getLogger(__name__)


def get_checkpoint_path(cfg: DictConfig):
    resume_from_checkpoint = None

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        checkpoint_path = f"{cfg.general.save_dir}/{cfg.general.experiment_id}/last-epoch.ckpt"
        if os.path.isfile(checkpoint_path):
            resume_from_checkpoint = checkpoint_path

    return resume_from_checkpoint


def get_parameters(cfg: DictConfig):
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    resume_from_checkpoint = get_checkpoint_path(cfg)

    for log in cfg.logging:
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))

    torch.set_float32_matmul_precision(cfg.general.float32_matmul_precision)

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        print(f"Resuming from checkpoint: {cfg.general.checkpoint}")
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
        resume_from_checkpoint = cfg.general.checkpoint

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
    return cfg, model, loggers, resume_from_checkpoint


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml", version_base="1.1")
def train(cfg: DictConfig):
    cfg, model, loggers, resume_from_checkpoint = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())

    runner = Trainer(
        logger=loggers,
        accelerator=cfg.general.accelerator,
        callbacks=callbacks,
        default_root_dir=str(cfg.general.save_dir),
        log_every_n_steps=1,
        **cfg.trainer,
    )
    runner.fit(model, ckpt_path=resume_from_checkpoint)


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml", version_base="1.1")
def test(cfg: DictConfig):
    cfg, model, loggers, resume_from_checkpoint = get_parameters(cfg)
    runner = Trainer(
        logger=loggers,
        accelerator=cfg.general.accelerator,
        default_root_dir=str(cfg.general.save_dir),
        log_every_n_steps=1,
        **cfg.trainer,
    )
    runner.test(model, ckpt_path=resume_from_checkpoint)


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml", version_base="1.1")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())  # because hydra wants to change dir for some reason

    if cfg["general"]["train_mode"]:
        train(cfg)
    else:
        test(cfg)

    logger.info("Stopped (finished or aborted)")


if __name__ == "__main__":
    main()
