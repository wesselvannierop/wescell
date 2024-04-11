from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import set_float32_matmul_precision

# Local imports
from general.data.data_pair_creation import CollateFnWithTransforms, DataTransformPair
from general.data.dataloaders.evican import EvicanDataset
from general.data.dataloaders.generic import GenericDataModule, GenericDataset
from general.data.dataloaders.helpers import TrainValDataModule
from general.data.illumination_correction import IlluminationCorrection
from general.model import UnsupervisedWesCell


def training(config: dict) -> float:
    """Trains, using PyTorch Lightning, an unsupervised model for cell segmentation.

    Args:
        config (dict): Dict with the config.

    Returns:
        float: Best Jaccard score achieved by the model.
    """
    set_float32_matmul_precision(config["float32_matmul_precision"])

    # Init config
    print("Config:")
    print(config)
    data_config = config["data"]
    train_config = config["train"]
    pl.seed_everything(config["seed"], workers=True)

    # Setup logger / TensorBoard
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    ex_name = f"wescell-{time}-" + config["ex_name"]
    checkpoint_dir = Path(train_config["checkpoint_dir"]) / ex_name
    checkpoint_dir.mkdir()
    logger = TensorBoardLogger(
        save_dir=train_config["checkpoint_dir"],
        name=ex_name,
        flush_secs=10,
    )

    # Train transforms (make data pair)
    train_transforms = DataTransformPair(
        crop_size=data_config["crop_size"],
        normalization=data_config["normalization"],
        jitter_strength=data_config["jitter_strength"],
        blur_strength=data_config["blur_strength"],
        rotations=data_config["rotations"],
        illumination_correction=data_config["illumination_correction"],
    )

    # Validation transforms
    val_target_transform = T.Compose(
        [
            T.CenterCrop((data_config["val_crop"], data_config["val_crop"])),
            T.ToTensor(),
        ]
    )
    val_image_transform = T.Compose(
        [
            T.CenterCrop((data_config["val_crop"], data_config["val_crop"])),
            T.ToTensor(),
            IlluminationCorrection(
                **data_config["illumination_correction"]
            ),  # NOTE: ic is also part of data augmentation
            T.Normalize(*data_config["normalization"]),
        ]
    )

    assert (
        data_config["evican_dir"] == None or data_config["data_dir"] == None
    ), "evican_dir and data_dir are mutually exclusive!"

    # Train datamodule
    train_data_module = GenericDataModule(
        dataset=EvicanDataset if data_config["evican_dir"] else GenericDataset,
        data_dir=data_config["evican_dir"] if data_config["evican_dir"] else data_config["data_dir"],
        batch_size=train_config["batch_size"],
        val_batch_size=train_config["val_batch_size"],
        train_transforms=train_transforms,
        val_image_transform=val_image_transform,
        val_target_transform=val_target_transform,
        num_workers=config["num_workers"],
        color_mode="L",  # L = grayscale
        return_name=True,
        return_train_targets=False,
        collate_fn=CollateFnWithTransforms(names=True),
        **train_config["data_loader_kwargs"],
    )

    # Validation datamodule
    val_data_module = GenericDataModule(
        dataset=GenericDataset,
        data_dir=data_config["labeled_data_dir"],
        val_batch_size=train_config["val_batch_size"],
        val_image_transform=val_image_transform,
        val_target_transform=val_target_transform,
        num_workers=config["num_workers"],
        color_mode="L",  # L = grayscale
        return_name=True,
        **train_config["data_loader_kwargs"],
    )

    # Combine datamodules
    data_module = TrainValDataModule(train_data_module, val_data_module)

    # Training module
    module = UnsupervisedWesCell(
        nr_of_clusters=train_config["nr_of_clusters"],
        consider_neighbouring_pixels=train_config["consider_neighbouring_pixels"],
        entropy_coeff=train_config["entropy_coeff"],
        optimizer=train_config["optimizer"],
        lr=train_config["lr"],
    )

    # Checkpoint saving
    checkpoint_callback = ModelCheckpoint(
        monitor="ValMetrics/jaccard",
        mode="max",
        filename="ckp-{epoch:02d}",
        save_top_k=train_config["save_top_k"],
        verbose=True,
        every_n_epochs=train_config["save_checkpoint_every_n_epochs"],
    )
    callbacks = [checkpoint_callback]

    # Setup trainer
    trainer = pl.Trainer(
        default_root_dir=checkpoint_dir,
        val_check_interval=train_config["val_check_interval"],
        logger=logger,
        max_steps=train_config["max_steps"],
        accelerator="auto",
        fast_dev_run=train_config["fast_dev_run"],
        log_every_n_steps=3,  # Log every 3 batches
        callbacks=callbacks,
        check_val_every_n_epoch=None,
    )

    # Run training
    trainer.fit(
        module,
        datamodule=data_module,
    )

    # Return best Jaccard score at the end of training
    return (
        checkpoint_callback.best_model_score.item()
        if checkpoint_callback.best_model_score
        else None
    )
