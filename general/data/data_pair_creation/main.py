from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms as T

# Local imports
from general.data.illumination_correction import IlluminationCorrection


class RandomCropToTensor:
    def __init__(self, crop_size) -> None:
        self.random_crop = T.RandomCrop((crop_size, crop_size))

    def __call__(self, sample: Image):
        # Pad if too small
        if sample.size < self.random_crop.size:
            sample = TF.center_crop(sample, self.random_crop.size)

        # Random crop
        crop = self.random_crop.get_params(sample, self.random_crop.size)
        sample = TF.crop(sample, *crop)

        return TF.to_tensor(sample), crop


class RandomApply(T.RandomApply):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p)

    def forward(self, img, transform):
        if self.p < torch.rand(1):
            return img, transform
        for t in self.transforms:
            img, transform = t(img, transform)
        return img, transform


class Compose(T.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, transforms):
        for t in self.transforms:
            img, transforms = t(img, transforms)
        return img, transforms


class DataTransformPair:
    def __init__(
        self,
        crop_size: int,
        normalization: Tuple[float, float],
        jitter_strength: float = 1.0,
        blur_strength: float = 1.0,
        rotations: list[float] = [0, -90, 90, 180],
        illumination_correction: dict = {},
    ) -> None:
        """Creates the data pair that is required for the UnsupervisedWesCell model.

        Args:
            crop_size (int): Set the random crop size (square).
            normalization (tuple): Give a tuple with the [mean, std] of the dataset.
            jitter_strength (float, optional): How much color jitter (brightness / contrast changes) is applied. Defaults to 1.0.
            blur_strength (float, optional): How much blurring is applied. Defaults to 1.0.
            rotations (list, optional): What angles of rotation to choose from.
            illumination_correction (dict, optional): Settings for the illumination correction filter. Defaults to {}.
        """

        self.rotations = rotations
        self.color_jitter = T.ColorJitter(0.8 * jitter_strength, 0.8 * jitter_strength, 0, 0)
        self.blur = T.GaussianBlur(kernel_size=31, sigma=[blur_strength * 0.1, blur_strength * 2.0])
        self.hflip = TF.hflip

        self.normalize = T.Normalize(*normalization)
        self.preprocess = RandomCropToTensor(crop_size)
        self.illumination_correction = IlluminationCorrection(**illumination_correction)

        self.transforms = Compose(
            [
                RandomApply([self.illumination_correction_fn], 0.3),
                RandomApply([self.color_jitter_fn], 0.9),
                RandomApply([self.blur_fn], 0.6),
                self.rotate_fn,
                RandomApply([self.hflip_fn], 0.5),
            ]
        )

    def illumination_correction_fn(self, sample, transforms):
        return self.illumination_correction(sample), transforms

    def rotate_fn(self, sample, transforms):
        i = torch.randint(low=0, high=len(self.rotations), size=(1,))
        transforms["rotate"] = self.rotations[i]
        return TF.rotate(sample, self.rotations[i]), transforms

    def blur_fn(self, sample, transforms):
        transformed = self.blur(sample)
        return transformed, transforms

    def color_jitter_fn(self, sample, transforms):
        transformed = self.color_jitter(sample)
        return transformed, transforms

    def hflip_fn(self, sample, transforms):
        transforms["hflip"] = True
        return TF.hflip(sample), transforms

    @staticmethod
    def inv(samples, transforms):
        if len(samples.shape) <= 3:
            samples = samples.unsqueeze(0)
        inverted = []
        for sample, transform in zip(samples, transforms):
            if transform["hflip"]:
                sample = TF.hflip(sample)
            sample = TF.rotate(sample, -transform["rotate"])
            inverted.append(sample)
        return torch.stack(inverted, dim=0)

    def __call__(self, sample):
        transformed_pair = []
        transform_pair = []
        preprocessed, _ = self.preprocess(sample)
        for _ in range(2):
            transformed, transform = self.transforms(preprocessed, {"hflip": False, "rotate": 0})
            normalized = self.normalize(transformed)
            transformed_pair.append(normalized)
            transform_pair.append(transform)

        return transformed_pair, transform_pair


class CollateFnWithTransforms:
    def __init__(self, targets=False, names=True) -> None:
        """Alterative collate_fn for pytorch's dataloader because the geometric transformations have
        to be inverted. This allows the transformations to be sent by the dataloader to the train
        script.

        Args:
            targets (bool, optional): Set to True if targets are returned. Defaults to False.
            names (bool, optional): Set to True if filenames are returned. Defaults to True.
        """
        self.targets = targets
        self.names = names

    def images_and_transforms(self, data):
        images1 = torch.stack([d[0][0][0] for d in data], dim=0)
        images2 = torch.stack([d[0][0][1] for d in data], dim=0)
        transforms1 = [d[0][1][0] for d in data]
        transforms2 = [d[0][1][1] for d in data]
        return (images1, images2), (transforms1, transforms2)

    def __call__(self, data):
        res = list(self.images_and_transforms(data))
        if self.targets:
            target = [d[1] for d in data]
            if torch.is_tensor(target[0]):
                target = torch.stack([d[1] for d in data], dim=0)
            res.append(target)
        if self.names:
            names = [d[self.targets + 1] for d in data]
            res.append(names)
        return res
