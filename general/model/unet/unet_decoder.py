import torch
import torch.nn as nn

# Local imports
from .unet_parts import Up, OutConv


class UNetDecoder(nn.Module):
    def __init__(self, embed_dims: list, n_classes: int = 1, bilinear: bool = False) -> None:
        """Initializes the UNetDecoder module.

        Args:
            embed_dims (list): A list of integers representing the number of channels in each decoder block.
            n_classes (int, optional): The number of output classes. Defaults to 1.
            bilinear (bool, optional): A flag indicating whether to use bilinear interpolation for upsampling. Defaults to False.
        """
        super().__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.up1 = Up(embed_dims[-1], embed_dims[-2] // factor, bilinear)
        self.up2 = Up(embed_dims[-2], embed_dims[-3] // factor, bilinear)
        self.up3 = Up(embed_dims[-3], embed_dims[-4] // factor, bilinear)
        self.outc = OutConv(embed_dims[0], n_classes)

    def forward(self, emb1, emb2, emb3, emb4):
        x = self.up1(emb4, emb3)
        x = self.up2(x, emb2)
        x = self.up3(x, emb1)
        logits = self.outc(x)
        return logits


class UNetDecoderWithSoftmax(UNetDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args):
        return torch.softmax(super().forward(*args), dim=1)
