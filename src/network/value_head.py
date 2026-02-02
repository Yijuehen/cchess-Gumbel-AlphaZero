"""
Value Head for AlphaZero Network

Outputs a scalar value in [-1, 1] representing the expected game outcome
from the current position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """
    Value head for the AlphaZero network.

    Takes the feature map from the backbone and outputs a scalar value
    in the range [-1, 1] representing the expected outcome.

    Architecture:
        Input (C, H, W) -> Conv1x1 -> Flatten -> Linear -> ReLU -> Linear -> Tanh
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_dim: int = 256,
        board_height: int = 10,
        board_width: int = 9
    ):
        """
        Initialize value head.

        Args:
            input_channels: Number of channels from backbone
            hidden_dim: Size of hidden layer
            board_height: Board height (10 for Chinese Chess)
            board_width: Board width (9 for Chinese Chess)
        """
        super(ValueHead, self).__init__()

        self.input_channels = input_channels
        self.board_size = board_height * board_width

        # Value head: 1x1 convolution to 1 channel
        self.value_conv = nn.Conv2d(
            input_channels, 1,
            kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(1)

        # Fully connected layers
        self.value_fc1 = nn.Linear(self.board_size, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Value tensor of shape (batch, 1) with values in [-1, 1]
        """
        # 1x1 convolution
        x = self.value_conv(x)
        x = self.value_bn(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # First fully connected
        x = F.relu(self.value_fc1(x))

        # Output layer with tanh activation
        value = torch.tanh(self.value_fc2(x))

        return value


class ValueHeadLarge(nn.Module):
    """
    Larger value head with more hidden layers.
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_dim1: int = 256,
        hidden_dim2: int = 128,
        board_height: int = 10,
        board_width: int = 9
    ):
        """Initialize larger value head."""
        super(ValueHeadLarge, self).__init__()

        self.board_size = board_height * board_width

        self.value_conv = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)

        self.value_fc1 = nn.Linear(self.board_size, hidden_dim1)
        self.value_fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.value_fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through larger value head."""
        x = self.value_conv(x)
        x = self.value_bn(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.value_fc1(x))
        x = F.relu(self.value_fc2(x))
        value = torch.tanh(self.value_fc3(x))
        return value


class ValueHeadWithPooling(nn.Module):
    """
    Value head with global average pooling before FC layers.
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_dim: int = 256,
        board_height: int = 10,
        board_width: int = 9
    ):
        """Initialize value head with pooling."""
        super(ValueHeadWithPooling, self).__init__()

        # Instead of conv to 1 channel, use all channels with pooling
        self.value_fc1 = nn.Linear(input_channels, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with global average pooling."""
        # Global average pooling over spatial dimensions
        x = x.mean(dim=[2, 3])  # (batch, channels)

        x = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(x))
        return value


class ValueHeadWithDropout(nn.Module):
    """
    Value head with dropout for regularization during training.
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_dim: int = 256,
        dropout_prob: float = 0.3,
        board_height: int = 10,
        board_width: int = 9
    ):
        """Initialize value head with dropout."""
        super(ValueHeadWithDropout, self).__init__()

        self.board_size = board_height * board_width

        self.value_conv = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)

        self.value_fc1 = nn.Linear(self.board_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout."""
        x = self.value_conv(x)
        x = self.value_bn(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.value_fc1(x))
        x = self.dropout(x)
        value = torch.tanh(self.value_fc2(x))
        return value


def create_value_head(
    head_type: str = "standard",
    input_channels: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create value heads.

    Args:
        head_type: Type of value head ("standard", "large", "pooling", "dropout")
        input_channels: Number of input channels
        **kwargs: Additional arguments for specific head types

    Returns:
        Value head module
    """
    if head_type == "standard":
        return ValueHead(input_channels, **kwargs)
    elif head_type == "large":
        return ValueHeadLarge(input_channels, **kwargs)
    elif head_type == "pooling":
        return ValueHeadWithPooling(input_channels, **kwargs)
    elif head_type == "dropout":
        return ValueHeadWithDropout(input_channels, **kwargs)
    else:
        raise ValueError(f"Unknown value head type: {head_type}")


if __name__ == "__main__":
    # Test value heads
    print("Testing ValueHead...")

    batch_size = 4
    channels = 256
    height, width = 10, 9

    x = torch.randn(batch_size, channels, height, width)

    # Standard value head
    value = ValueHead(channels)
    output = value(x)
    print(f"Standard value head: {x.shape} -> {output.shape}")
    assert output.shape == (batch_size, 1)

    # Check output range
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    assert output.min() >= -1 and output.max() <= 1

    # Large value head
    value_large = ValueHeadLarge(channels, hidden_dim1=256, hidden_dim2=128)
    output_large = value_large(x)
    print(f"Large value head: {x.shape} -> {output_large.shape}")
    assert output_large.shape == (batch_size, 1)

    # Value head with pooling
    value_pool = ValueHeadWithPooling(channels)
    output_pool = value_pool(x)
    print(f"Pooling value head: {x.shape} -> {output_pool.shape}")
    assert output_pool.shape == (batch_size, 1)

    # Value head with dropout
    value_dropout = ValueHeadWithDropout(channels, dropout_prob=0.3)
    output_dropout = value_dropout(x)
    print(f"Dropout value head: {x.shape} -> {output_dropout.shape}")
    assert output_dropout.shape == (batch_size, 1)

    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter counts:")
    print(f"  Standard: {count_parameters(value):,}")
    print(f"  Large: {count_parameters(value_large):,}")
    print(f"  Pooling: {count_parameters(value_pool):,}")
    print(f"  Dropout: {count_parameters(value_dropout):,}")

    print("\n✓ All value heads test passed!")
