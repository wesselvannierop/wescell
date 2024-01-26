import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class IlluminationCorrection(torch.nn.Module):
    def __init__(
        self, gauss_size: int = 31, sigma=75, downscale_gb_factor=8, disable=False
    ) -> None:
        """Initialize an instance of the IlluminationCorrection module.

        Args:
            gauss_size (int, optional): The size of the Gaussian kernel used for smoothing the image. Defaults to 31.
            sigma (int, optional): The standard deviation of the Gaussian kernel. Defaults to 75.
            downscale_gb_factor (int, optional): The downscale factor for downsampling the image before applying the Gaussian blur. Defaults to 8.
            disable (bool, optional): Whether to disable the illumination correction. Defaults to False.
        """
        super().__init__()
        if gauss_size % 2 == 0:
            gauss_size += 1
        if sigma is None:
            sigma = 0.3 * ((gauss_size - 1) * 0.5 - 1) + 0.8  # like cv2 does

        self.gb = T.GaussianBlur(gauss_size, sigma)
        self.downscale_gb_factor = downscale_gb_factor  # 8 is a good default

        if disable:
            self.forward = lambda x: x

    def efficient_gb(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """Perform gaussian blurring on a downsampled version of the images and unsample the result.

        Args:
            img (torch.FloatTensor): The input image.

        Returns:
            torch.FloatTensor: The estimated background of the image.
        """
        w, h = TF.get_image_size(img)
        downscale_to = (h // self.downscale_gb_factor, w // self.downscale_gb_factor)
        return TF.resize(
            self.gb(TF.resize(img, downscale_to, antialias=True)), (h, w), antialias=True
        )

    def forward(self, img):
        assert torch.is_tensor(img)
        gb = self.efficient_gb(img)
        return self.illumination_correction(img, gb)

    def illumination_correction(
        self, img: torch.FloatTensor, gb: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Subtracts the estimated background from the original image and adds the mean of the estimated background.

        Args:
            img (torch.FloatTensor): Input image
            gb (torch.FloatTensor): Gaussian blurred image (estimation of the background)

        Returns:
            torch.FloatTensor: Illumination corrected image
        """
        img -= gb - gb.mean()
        return torch.clip(img, 0, 1)
