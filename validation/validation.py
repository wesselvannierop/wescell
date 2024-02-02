import pytorch_lightning as pl
import torchvision.transforms as T

from general.data.dataloaders.generic import GenericDataModule, GenericDataset
from general.data.illumination_correction import IlluminationCorrection

# Local imports
from general.model import UnsupervisedWesCell


def model_eval(
    ckpt_path: str,
    val_dir: str,
    normalization: tuple[float, float],
    batch_size: int = 1,
    device: str = "cuda",
    num_workers: int = 0,
    mode: str = "val",
    save_folder: str = None,
    illumination_correction: dict = {},
) -> list:
    """Validate a specific model.

    Args:
        ckpt_path (str): The path to the model checkpoint file.
        val_dir (str): The directory containing the validation data.
        normalization (tuple): A tuple of normalization parameters for preprocessing the data (mean, std).
        batch_size (int, optional): The batch size for validation. Defaults to 1.
        device (str, optional): The device to use for validation (e.g., "cuda" or "cpu"). Defaults to "cuda".
        num_workers (int, optional): The number of worker threads for data loading. Defaults to 0.
        mode (str, optional): The evaluation mode (e.g., "val" for validation). Defaults to "val".
        save_folder (str, optional): The folder to save the evaluation results. Defaults to None.
        illumination_correction (dict, optional): A dictionary of illumination correction parameters. Defaults to {}.

    Returns:
        list: List of dictionaries with metrics logged during the validation phase
    """
    # Transforms
    val_target_transform = [
        T.ToTensor(),
    ]
    val_image_transform = [
        T.ToTensor(),
        IlluminationCorrection(**illumination_correction),
        T.Normalize(*normalization),
    ]

    module = UnsupervisedWesCell.load_from_checkpoint(ckpt_path, save_folder=save_folder)

    # Init data module
    val_data_module = GenericDataModule(
        dataset=GenericDataset,
        data_dir=val_dir,
        batch_size=batch_size,
        val_batch_size=batch_size,
        train_transforms=None,
        val_image_transform=T.Compose(val_image_transform),
        val_target_transform=T.Compose(val_target_transform),
        num_workers=num_workers,
        color_mode="L",
        return_name=True,
    )

    # Run the val dataset
    if device == "cuda":
        accelerator = "gpu"
    else:
        accelerator = device
    trainer = pl.Trainer(
        logger=False,
        accelerator=accelerator,
    )

    if mode == "val":
        return trainer.validate(module, val_data_module)
    elif mode == "test":
        return trainer.test(module, val_data_module)
