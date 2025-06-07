from typing import List, Optional
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import pandas as pd
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase
from api import utils
from api.callback import NeptuneModelLogger
from pathlib import Path
import shutil
import torch
from datetime import datetime
log = utils.get_logger(__name__)
def train(config: DictConfig) -> Optional[float]:

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        print("seed", config.seed)
        seed_everything(config.seed, workers=True)

    # If required
    # Init Dataloaders
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, _convert_="partial"
    )

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init Lightning callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks: List[Callback] = [lr_monitor]
    if "callbacks" in config:
        for cb_name, cb_conf in config.callbacks.items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                try:
                    callback = hydra.utils.instantiate(cb_conf)
                    callbacks.append(callback)
                except Exception as e:
                    log.warning(f"Failed to instantiate callback {cb_name}: {str(e)}")

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for lg_name, lg_conf in config.logger.items():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                try:
                    logger_instance = hydra.utils.instantiate(lg_conf)
                    logger.append(logger_instance)
                    log.info(f"Successfully initialized logger: {lg_name}")
                except Exception as e:
                    log.warning(f"Failed to initialize logger {lg_name}: {str(e)}")

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if config.mode == "train":

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=model,
            trainer=trainer,
        )
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
        print(f"Best model checkpoint saved at: {ckpt_path}")
    elif config.mode == "test":
        # Testing mode (Pretrained Model)
        pretrained_ckpt_path = config.get("pretrained_ckpt_path")
        assert pretrained_ckpt_path is not None, "Pretrained ckpt path must be specified in test mode!"
        log.info(f"Loading pretrained model from checkpoint: {pretrained_ckpt_path}")
        model_class = type(model)  # Get the class of the instantiated model
        model = model_class.load_from_checkpoint(pretrained_ckpt_path, strict=False)  # Load the checkpoint
        # Run testing
        log.info("Starting testing!")
        with torch.no_grad():
            trainer.test(model=model, datamodule=datamodule)
    else:
        raise ValueError(f"Unknown mode '{config.mode}'. Use 'train' or 'test'.")
    test_metrics = trainer.callback_metrics
    metrics_dict = {k: v.item() if hasattr(v, 'item') else v
                    for k, v in test_metrics.items()}
    return metrics_dict
