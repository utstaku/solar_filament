"""EdgeAttNet: the plain U-Net with edge-guided self-attention at its bottleneck.

Re-implemented from the paper's description (arXiv 2509.02964, on MAGFiLO), not from the authors' code. A
learnable edge map of the input -- 3 x 3 convolutions initialised as Sobel filters -- is pooled to the
bottleneck resolution and added to the features from which the attention's queries and keys are made, not
to its values. Positions on the same faint edges (a filament's border, its barbs) then attend to each other,
while what is passed along stays the image features. Everything else is `models.unet.UNet`.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.unet import UNet


class EdgeMap(nn.Module):
    """(B, C, H, W) image -> (B, width, H, W) edge responses; starts as |Sobel x|, |Sobel y| per channel."""

    def __init__(self, in_channels: int = 2, width: int = 16):
        super().__init__()
        self.sobel = nn.Conv2d(in_channels, 2 * in_channels, 3, padding=1, groups=in_channels, bias=False)
        kernel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4
        with torch.no_grad():
            self.sobel.weight.copy_(torch.stack([kernel, kernel.T]).repeat(in_channels, 1, 1).unsqueeze(1))
        self.mix = nn.Sequential(nn.Conv2d(2 * in_channels, width, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.sobel(x).abs())


class EdgeGuidedAttention(nn.Module):
    """Multi-head self-attention over the bottleneck positions, queries and keys guided by the edge map."""

    def __init__(self, channels: int, edge_width: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(1, channels)
        self.edge = nn.Conv2d(edge_width, channels, 1)
        self.qk = nn.Conv2d(channels, 2 * channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)  # starts as the identity: the plain U-Net's bottleneck
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        features = self.norm(x)
        guided = features + self.edge(F.adaptive_avg_pool2d(edges, (h, w)))
        q, k = self.qk(guided).chunk(2, dim=1)
        v = self.v(features)

        def split(t):  # (B, C, H, W) -> (B, heads, HW, C / heads)
            return t.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)

        attended = F.scaled_dot_product_attention(split(q), split(k), split(v))
        return x + self.out(attended.transpose(-1, -2).reshape(b, c, h, w))


class EdgeAttNet(UNet):
    def __init__(self, in_channels: int = 2, base: int = 32, depth: int = 4, edge_width: int = 16, heads: int = 8):
        super().__init__(in_channels=in_channels, base=base, depth=depth)
        self.edge_map = EdgeMap(in_channels, edge_width)
        self.attention = EdgeGuidedAttention(base * 2**depth, edge_width, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edges = self.edge_map(x)
        skips = []
        for level, encoder in enumerate(self.encoders):
            skips.append(encoder(x if level == 0 else self.pool(skips[-1])))
        y = self.attention(skips.pop(), edges)
        for decoder in self.decoders:
            skip = skips.pop()
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = decoder(torch.cat([y, skip], dim=1))
        return self.head(y)
