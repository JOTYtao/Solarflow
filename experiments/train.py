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
from datetime import datetime
from core import utils
from core.callbacks import NeptuneModelLogger
from pathlib import Path
import shutil
import torch
from datetime import datetime

log = utils.get_logger(__name__)


def save_metrics_to_csv(metrics_dict, save_dir, filename="metrics.csv"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics_dict])
    csv_path = save_dir / filename
    df.to_csv(csv_path, index=False)
    log.info(f"Metrics saved to {csv_path}")


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

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
    callbacks: List[Callback] = [lr_monitor, NeptuneModelLogger(model_name=config.model_name)]
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

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

        # Train the model
        # if config.trainer.auto_lr_find or config.trainer.auto_scale_batch_size:
        #     log.info("Starting tuning!")
        #     trainer.tune(model=model, datamodule=datamodule)
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
        if ckpt_path:
            default_root_dir = Path(trainer.default_root_dir)
            best_ckpt_path = default_root_dir / "best.ckpt"
            shutil.copyfile(ckpt_path, best_ckpt_path)
            print(f"Copied best model to: {best_ckpt_path}")
        else:
            print("No best model found to save as 'best.ckpt'")

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
    save_dir = Path(config.paths.output_dir) / "test_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    test_metrics = trainer.callback_metrics
    print(test_metrics)
    metrics_dict = {k: v.item() if hasattr(v, 'item') else v
                    for k, v in test_metrics.items()}

    save_dir = Path(config.paths.output_dir) / "metrics"
    save_metrics_to_csv(metrics_dict, save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_metrics_to_csv(
        metrics_dict,
        save_dir,
        f"metrics_{timestamp}.csv"
    )
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_dict[optimized_metric]
    return metrics_dict
