"""
Residual Block for AlphaZero Network

Implements the residual block used in the backbone of the AlphaZero network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard residual block with two convolutional layers.

    Architecture:
        Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> + -> ReLU -> Output
                                      ^                        |
                                      |------------------------
    """

    def __init__(self, channels: int = 256):
        """
        Initialize residual block.

        Args:
            channels: Number of input/output channels
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3, stride=1, padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3, stride=1, padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of same shape
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    """
    Bottleneck residual block (1x1 -> 3x3 -> 1x1) for deeper networks.

    This is more efficient than standard residual block for very deep networks.
    """

    def __init__(self, channels: int = 256, bottleneck_ratio: int = 4):
        """
        Initialize bottleneck block.

        Args:
            channels: Number of input/output channels
            bottleneck_ratio: Reduction ratio for bottleneck
        """
        super(ResNetBottleneck, self).__init__()

        bottleneck_channels = channels // bottleneck_ratio

        self.conv1 = nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck block."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = F.relu(out)

        return out


class PreActivationResidualBlock(nn.Module):
    """
    Pre-activation residual block (ResNet v2).

    BN -> ReLU -> Conv -> BN -> ReLU -> Conv (skip connection added after)
    """

    def __init__(self, channels: int = 256):
        """Initialize pre-activation residual block."""
        super(PreActivationResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3, stride=1, padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3, stride=1, padding=1,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pre-activation block."""
        residual = x

        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += residual

        return out


def create_residual_block(
    block_type: str = "standard",
    channels: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create residual blocks.

    Args:
        block_type: Type of block ("standard", "bottleneck", "preactivation")
        channels: Number of channels
        **kwargs: Additional arguments for specific block types

    Returns:
        Residual block module
    """
    if block_type == "standard":
        return ResidualBlock(channels)
    elif block_type == "bottleneck":
        return ResNetBottleneck(channels, **kwargs)
    elif block_type == "preactivation":
        return PreActivationResidualBlock(channels)
    else:
        raise ValueError(f"Unknown block type: {block_type}")


if __name__ == "__main__":
    # Test residual blocks
    print("Testing ResidualBlock...")

    batch_size = 4
    channels = 256
    height, width = 10, 9

    x = torch.randn(batch_size, channels, height, width)

    # Standard block
    block = ResidualBlock(channels)
    out = block(x)
    print(f"Standard block: {x.shape} -> {out.shape}")
    assert out.shape == x.shape

    # Bottleneck block
    bottleneck = ResNetBottleneck(channels)
    out = bottleneck(x)
    print(f"Bottleneck block: {x.shape} -> {out.shape}")
    assert out.shape == x.shape

    # Pre-activation block
    preact = PreActivationResidualBlock(channels)
    out = preact(x)
    print(f"Pre-activation block: {x.shape} -> {out.shape}")
    assert out.shape == x.shape

    print("\n✓ All residual blocks test passed!")
