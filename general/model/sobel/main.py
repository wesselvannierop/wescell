import torch
import torch.nn as nn


class Sobel(nn.Module):
    def __init__(self):
        """Sobel filter that is prepended (channel-wise) before the image."""
        super().__init__()
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) / 2
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) / 4
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)

        return torch.cat([x, img], dim=1)
