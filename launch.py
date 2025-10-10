import gc
import os
import time
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig

torch.serialization.add_safe_globals([ListConfig])
from pathlib import Path

import pytorch_lightning as pl

from core.registry import REG
from dataloader.main_datamodule import MainDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from utils.callbacks import (CodeSnapshotCallback, ConfigSnapshotCallback,
                             CustomProgressBar)
import models, systems

class PrintStep(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        if getattr(trainer, "is_global_zero", True):  # only print on rank 0 in distributed training
            print(f"[RESUME CHECK] global_step = {trainer.global_step}")

def _resolve_resume_ckpt(exp_conf):
    """
    Resolve exp.resume into an absolute checkpoint path or None.
    - If a filename is given: join with ckpt_path
    - If an absolute path is given: use it directly
    - If False/empty: return None
    """
    resume = getattr(exp_conf, "resume", False)
    if not resume:
        return None

    # three supported forms:
    # 1) 'last.ckpt' or 'epoch=...-step=...ckpt' (relative filename -> join with ckpt_path)
    # 2) absolute '/path/to/*.ckpt'
    # 3) a directory (not recommended): try 'last.ckpt' inside that dir
    p = Path(str(resume))
    if p.suffix == ".ckpt" and p.is_absolute():
        ckpt = p
    elif p.suffix == ".ckpt":
        ckpt = Path(exp_conf.ckpt_path) / p.name
    else:
        # may be a directory; try to locate last.ckpt
        candidate = Path(exp_conf.ckpt_path if not p.is_absolute() else p) / "last.ckpt"
        ckpt = candidate

    return str(ckpt) if Path(ckpt).is_file() else None

@hydra.main(version_base=None, config_path="configs")
def main(config : DictConfig) -> None:
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    cfg = config.conf

    # ----------------- Devices / acceleration -----------------
    # infer device count from CUDA_VISIBLE_DEVICES
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible == "" or visible == "-1":
        accelerator = "cpu"
        devices = 1
    else:
        n_vis = len([x for x in visible.split(",") if x.strip() != ""])
        n_cuda = torch.cuda.device_count()
        accelerator = "gpu" if n_vis > 0 and n_cuda > 0 else "cpu"
        devices = min(n_vis or 1, n_cuda or 1) if accelerator == "gpu" else 1

    if cfg.exp.phase == 'train':
        cfg.exp.trial_name = (datetime.now().strftime('@%Y%m%d-%H%M%S') + cfg.exp.tag)

    cfg.exp.exp_path = os.path.join(cfg.exp.exp_path, cfg.exp.name)
    cfg.exp.save_dir = os.path.join(cfg.exp.exp_path, cfg.exp.trial_name, 'save')
    cfg.exp.ckpt_path = os.path.join(cfg.exp.exp_path, cfg.exp.trial_name, 'ckpt')
    cfg.exp.code_path = os.path.join(cfg.exp.exp_path, cfg.exp.trial_name, 'code')
    cfg.exp.config_path = os.path.join(cfg.exp.exp_path, cfg.exp.trial_name, 'configs')

    if 'seed' not in cfg.exp:
        cfg.exp.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(cfg.exp.seed)

    datamodule = MainDataModule(dataset_config=cfg.dataset)
    system = REG.build('system', cfg=config.conf, name=config.conf.system.name)

    # ----------------- Callbacks / Loggers -----------------
    callbacks = []
    loggers = []
    if cfg.exp.phase == 'train':
        callbacks += [
            ModelCheckpoint(
                dirpath=cfg.exp.ckpt_path,
                **cfg.checkpoint,  # e.g., save_top_k=0, every_n_train_steps=...
            ),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(cfg.exp.code_path, use_version=False),
            ConfigSnapshotCallback(cfg, cfg.exp.config_path, use_version=False),
            CustomProgressBar(refresh_rate=1),
            PrintStep()
        ]

        loggers += [
            WandbLogger(
                id=cfg.exp.trial_name,  # fixed run id (use your trial_name)
                resume="allow",  # allow resuming/appending
                name=cfg.exp.trial_name,
                project=cfg.exp.name,
                save_dir=cfg.exp.exp_path,
                log_model=False,  # set True if you want to upload checkpoints to W&B
                reinit=True
            )
        ]

    # ----------------- Build Trainer -----------------
    trainer = Trainer(
        accelerator="auto",
        devices=devices,
        callbacks=callbacks if callbacks else None,
        logger=loggers if loggers else False,
        **cfg.trainer
    )

    # ----------------- train / validate / test / predict -----------------
    resume_ckpt = _resolve_resume_ckpt(cfg.exp)
    print(f"[INFO] Resuming from checkpoint: {resume_ckpt}")

    phase = config.conf.exp.phase
    print(f"[INFO] Starting phase: {phase}")
    if phase == "train":
        if resume_ckpt:
            trainer.fit(system, datamodule=datamodule, ckpt_path=resume_ckpt)
        else:
            trainer.fit(system, datamodule=datamodule)

        if cfg.exp.test_after_train:
            trainer.test(system, datamodule=datamodule, ckpt_path=resume_ckpt)
        if cfg.exp.predict_after_train:
            trainer.predict(system, datamodule=datamodule, ckpt_path=resume_ckpt)

    elif phase == "val":
        if not resume_ckpt:
            raise FileNotFoundError("Validation phase requires providing exp.resume=*.ckpt or a valid path.")
        trainer.validate(system, datamodule=datamodule, ckpt_path=resume_ckpt)

    elif phase == "test":
        if not resume_ckpt:
            raise FileNotFoundError("Test phase requires providing exp.resume=*.ckpt or a valid path.")
        trainer.test(system, datamodule=datamodule, ckpt_path=resume_ckpt)

    elif phase == "predict":
        if not resume_ckpt:
            raise FileNotFoundError("Predict/Export phase requires providing exp.resume=*.ckpt or a valid path.")
        trainer.predict(system, datamodule=datamodule, ckpt_path=resume_ckpt)

    else:
        raise ValueError(f"Unknown phase: {phase}")
    gc.collect()


if __name__ == '__main__':
    main()
