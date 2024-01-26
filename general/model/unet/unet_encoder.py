import torch
import torch.nn as nn

# Local imports
from .unet_parts import DoubleConv, Down
from general.model.sobel import Sobel


class UNetEncoder(nn.Module):
    def __init__(self, embed_dims: list = [16, 32, 64, 128]):
        """A class representing the encoder module of the U-Net architecture.

        Args:
            embed_dims (list, optional): A list of embedding dimensions for each encoding layer.
                                         Defaults to [16, 32, 64, 128].
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.sobel = Sobel()
        self.inc = DoubleConv(2, self.embed_dims[0])
        self.down1 = Down(self.embed_dims[0], self.embed_dims[1])
        self.down2 = Down(self.embed_dims[1], self.embed_dims[2])
        self.down3 = Down(self.embed_dims[2], self.embed_dims[3])

    def forward(self, img: torch.FloatTensor):
        x = self.sobel(img)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4
