from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


class GenericDataset(VisionDataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        split="train",
        color_mode="RGB",
        return_name=False,
        return_targets=False,
        limit_images=None,
    ):
        """This will load data from a directory and apply the corresponding transformations.
        Optionally, it can return the names of the files and load targets.

        By default, this class will load the files in data_dir / split and expects images to be .jpg and mask to be .png
        For this class to work for a different dataset structure, implement self.setup() and make sure to return self.
        An example for this can be found in `general.data.dataloaders.evican`.

        Args:
            data_dir (str): Path to folder that contains the dataset.
            transform (Optional[Callable], optional): Transform for the images. Mutually exclusive to `transforms`. Defaults to None.
            target_transform (Optional[Callable], optional): Transform for the targets. Mutually exclusive to `transforms`. Defaults to None.
            transforms (Optional[Callable], optional): Transforms for both the images and the targets. Mutually exclusive to `transform` and `target_transform`. Defaults to None.
            split (str, optional): Data split. Defaults to "train".
            color_mode (str, optional): Load the images in this color mode (uses PIL images so `"L"` for grayscale). Defaults to "RGB".
            return_name (bool, optional): Optionally the name of the images. Defaults to False.
            return_targets (bool, optional): Optionally return the targets of the images. Defaults to False.
            limit_images (_type_, optional): Limit the amount of images loaded from the folder. Defaults to None.
        """
        super().__init__(data_dir, transforms, transform, target_transform)
        assert split in [
            "train",
            "val",
            "test",
            "predict",
        ], "Only train, test and val splits are implemented."
        self.split = split
        self.color_mode = color_mode
        self.return_name = return_name
        self.return_targets = return_targets
        self.limit_images = limit_images

    def setup(self):
        """Load image paths to self.files and self.mask_files.

        Returns: self
        """
        dataset_path = Path(self.root) / self.split
        self.files = self.files_in_path_with_ext(dataset_path, ".jpg")
        if self.return_targets:
            self.mask_files = self.files_in_path_with_ext(dataset_path, ".png")
        self.assertions()
        return self

    def assertions(self):
        assert len(self.files) > 0, "No images found."
        if self.return_targets:
            assert len(self.mask_files) > 0, "No masks found."
            assert len(self.mask_files) == len(self.files), "Unequal number of images and masks."

    def __len__(self):
        return len(self.files)

    def load_mask(self, index):
        return Image.open(str(self.mask_files[index])).convert("L")

    def load_image(self, index):
        return Image.open(str(self.files[index])).convert(self.color_mode)

    def get_name(self, index):
        return str(self.files[index].stem)

    def __getitem__(self, index):
        image = self.load_image(index)
        if self.return_targets:
            mask = self.load_mask(index)
            res = self.transforms(image, mask)
            if self.return_name:
                return *res, self.get_name(index)
        else:
            res = self.transforms(image)
            if self.return_name:
                return res, self.get_name(index)
        return res

    @staticmethod
    def files_in_path_with_ext(path: Path, ext):
        files = [file for file in path.iterdir() if file.suffix == ext]
        return sorted(files)


class GenericDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset: VisionDataset = GenericDataset,
        dataset_kwargs: dict = {},
        batch_size: int = 1,
        val_batch_size: int = 1,
        train_transforms: Callable = None,
        val_image_transform: Callable = None,
        val_target_transform: Callable = None,
        num_workers: int = 0,
        color_mode: str = "RGB",
        return_name: bool = False,
        return_train_targets: bool = True,
        collate_fn: Callable = None,
        limit_images: int = None,
        **data_loader_kwargs,
    ) -> None:
        """DataModule for PyTorch Lightning with options for easy access.

        Args:
            dataset (VisionDataset): The dataset class that you want to use for PyTorch Lightning. Defaults to GenericDataset.
            data_dir (str): Path to folder that contains the dataset.
            dataset_kwargs (dict, optional): Arguments that should be sent to the dataset class. Defaults to {}.
            batch_size (int, optional): Train batch size. Defaults to 1.
            val_batch_size (int, optional): Validation / Test / Predict batch size. Defaults to 1.
            train_transforms (Callable, optional): Callable for the transforms for the train split for both the images and the targets. Defaults to None.
            val_image_transform (Callable, optional): Callable for the transform for the validation / test / predict split for the images. Defaults to None.
            val_target_transform (Callable, optional): Callable for the transform for the validation / test / predict split for the targets. Defaults to None.
            num_workers (int, optional): Amount of processes spawned for dataloading. Defaults to 0.
            color_mode (str, optional): Load the images in this color mode (uses PIL images so `"L"` for grayscale). Defaults to "RGB".
            return_name (bool, optional): Optionally the name of the images. Defaults to False.
            return_train_targets (bool, optional): Optionally don't return the targets of the images. Defaults to True.
            collate_fn (Callable, optional): Merges a list of samples to form a mini-batch of Tensor(s). Defaults to None.
            limit_images (int, optional): Limit the amount of images loaded from the folder. Defaults to None.
        """
        super().__init__()
        self.dataset = dataset

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_transforms = train_transforms
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform
        self.num_workers = num_workers
        self.return_train_targets = return_train_targets
        self.data_loader_kwargs = data_loader_kwargs
        self.collate_fn = collate_fn

        self.common_dataset_setup = (
            dict(
                color_mode=color_mode,
                return_name=return_name,
                limit_images=limit_images,
            )
            | dataset_kwargs
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_set = self.dataset(
                self.data_dir,
                transforms=self.train_transforms,
                split="train",
                return_targets=self.return_train_targets,
                **self.common_dataset_setup,
            ).setup()
        if stage == "validate":
            self.val_set = self.dataset(
                self.data_dir,
                transform=self.val_image_transform,
                target_transform=self.val_target_transform,
                split="val",
                return_targets=True,
                **self.common_dataset_setup,
            ).setup()
        if stage == "test":
            self.test_set = self.dataset(
                self.data_dir,
                transform=self.val_image_transform,
                target_transform=self.val_target_transform,
                split="test",
                return_targets=True,
                **self.common_dataset_setup,
            ).setup()
        if stage == "predict":
            self.predict_set = self.dataset(
                self.data_dir,
                transforms=self.val_image_transform,
                split="predict",
                return_targets=False,
                **self.common_dataset_setup,
            ).setup()

    def _dataloader(self, dataset, batch_size=None, collate_fn=None, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=(batch_size if batch_size else self.val_batch_size),
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            **self.data_loader_kwargs,
        )

    def train_dataloader(self):
        return self._dataloader(
            self.train_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True
        )

    def val_dataloader(self):
        return self._dataloader(self.val_set)

    def test_dataloader(self):
        return self._dataloader(self.test_set)

    def predict_dataloader(self):
        return self._dataloader(self.predict_set)
