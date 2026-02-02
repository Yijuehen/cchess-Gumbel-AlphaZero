"""
AlphaZero Network for Chinese Chess

Dual-head neural network with ResNet backbone, policy head, and value head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
from pathlib import Path

# Add parent directory to import from encoding module
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.resnet_block import ResidualBlock, create_residual_block
from network.policy_head import PolicyHead, create_policy_head
from network.value_head import ValueHead, create_value_head


class AlphaZeroNet(nn.Module):
    """
    AlphaZero network for Chinese Chess.

    Architecture:
        Input (14, 10, 9) -> Initial Conv -> Residual Blocks × N
                                           ├─> Policy Head -> Logits (2086)
                                           └─> Value Head -> Scalar [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 14,
        hidden_channels: int = 256,
        num_residual_blocks: int = 15,
        action_space_size: int = 2086,
        policy_head_type: str = "standard",
        value_head_type: str = "standard",
        residual_block_type: str = "standard"
    ):
        """
        Initialize AlphaZero network.

        Args:
            in_channels: Number of input channels (14 for Chinese Chess)
            hidden_channels: Number of hidden channels in backbone
            num_residual_blocks: Number of residual blocks
            action_space_size: Size of action space
            policy_head_type: Type of policy head
            value_head_type: Type of value head
            residual_block_type: Type of residual block
        """
        super(AlphaZeroNet, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_residual_blocks = num_residual_blocks
        self.action_space_size = action_space_size

        # Initial convolution
        self.initial_conv = nn.Conv2d(
            in_channels, hidden_channels,
            kernel_size=3, stride=1, padding=1,
            bias=False
        )
        self.initial_bn = nn.BatchNorm2d(hidden_channels)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            create_residual_block(residual_block_type, hidden_channels)
            for _ in range(num_residual_blocks)
        ])

        # Policy head
        self.policy_head = create_policy_head(
            policy_head_type,
            input_channels=hidden_channels,
            action_space_size=action_space_size,
            board_height=10,
            board_width=9
        )

        # Value head
        self.value_head = create_value_head(
            value_head_type,
            input_channels=hidden_channels,
            board_height=10,
            board_width=9
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 14, 10, 9)

        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: (batch, action_space_size)
            - value: (batch, 1) with values in [-1, 1]
        """
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a prediction (with softmax on policy logits).

        Args:
            x: Input tensor

        Returns:
            Tuple of (policy_probs, value)
        """
        policy_logits, value = self.forward(x)
        policy_probs = F.softmax(policy_logits, dim=1)
        return policy_probs, value

    def get_action_probabilities(
        self,
        x: torch.Tensor,
        legal_actions_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get action probabilities with optional legal action masking.

        Args:
            x: Input tensor
            legal_actions_mask: Optional boolean mask for legal actions

        Returns:
            Action probabilities (sum to 1 over legal actions)
        """
        policy_logits, _ = self.forward(x)

        if legal_actions_mask is not None:
            # Mask illegal actions
            policy_logits = policy_logits.masked_fill(~legal_actions_mask, -1e9)

        return F.softmax(policy_logits, dim=1)


class AlphaZeroNetSmall(AlphaZeroNet):
    """Smaller AlphaZero network for faster training/inference."""

    def __init__(
        self,
        in_channels: int = 14,
        hidden_channels: int = 128,
        num_residual_blocks: int = 10,
        action_space_size: int = 2086
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            action_space_size=action_space_size
        )


class AlphaZeroNetLarge(AlphaZeroNet):
    """Larger AlphaZero network for better performance."""

    def __init__(
        self,
        in_channels: int = 14,
        hidden_channels: int = 512,
        num_residual_blocks: int = 20,
        action_space_size: int = 2086
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            action_space_size=action_space_size,
            policy_head_type="large",
            value_head_type="large"
        )


def create_alphazero_net(
    net_type: str = "standard",
    **kwargs
) -> AlphaZeroNet:
    """
    Factory function to create AlphaZero networks.

    Args:
        net_type: Type of network ("small", "standard", "large")
        **kwargs: Additional arguments for network initialization

    Returns:
        AlphaZero network
    """
    if net_type == "small":
        return AlphaZeroNetSmall(**kwargs)
    elif net_type == "standard":
        return AlphaZeroNet(**kwargs)
    elif net_type == "large":
        return AlphaZeroNetLarge(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {net_type}")


def save_checkpoint(model: AlphaZeroNet, filepath: str, optimizer=None, epoch=None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': model.in_channels,
            'hidden_channels': model.hidden_channels,
            'num_residual_blocks': model.num_residual_blocks,
            'action_space_size': model.action_space_size,
        }
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: AlphaZeroNet = None, optimizer=None) -> AlphaZeroNet:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')

    if model is None:
        # Create model from saved config
        config = checkpoint['model_config']
        model = AlphaZeroNet(**config)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {filepath}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    return model


if __name__ == "__main__":
    # Test AlphaZero network
    print("Testing AlphaZeroNet...")

    batch_size = 4
    in_channels = 14
    height, width = 10, 9
    action_space = 2086

    x = torch.randn(batch_size, in_channels, height, width)

    # Standard network
    print("\n=== Standard Network ===")
    net = AlphaZeroNet(
        in_channels=in_channels,
        hidden_channels=256,
        num_residual_blocks=15,
        action_space_size=action_space
    )

    policy_logits, value = net(x)
    print(f"Input: {x.shape}")
    print(f"Policy logits: {policy_logits.shape}")
    print(f"Value: {value.shape}")

    assert policy_logits.shape == (batch_size, action_space)
    assert value.shape == (batch_size, 1)

    # Test predict
    policy_probs, value = net.predict(x)
    print(f"Policy probs sum: {policy_probs.sum(dim=1)}")
    assert torch.allclose(policy_probs.sum(dim=1), torch.ones(batch_size))

    # Test with legal action masking
    legal_mask = torch.zeros(batch_size, action_space, dtype=torch.bool)
    legal_mask[:, :100] = True  # First 100 actions are legal

    masked_probs = net.get_action_probabilities(x, legal_mask)
    print(f"Masked probs sum: {masked_probs.sum(dim=1)}")
    # Check that illegal actions have zero probability
    assert (masked_probs[:, 100:] == 0).all()

    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter count: {count_parameters(net):,}")

    # Test small network
    print("\n=== Small Network ===")
    net_small = AlphaZeroNetSmall(action_space_size=action_space)
    policy_small, value_small = net_small(x)
    print(f"Policy: {policy_small.shape}, Value: {value_small.shape}")
    print(f"Parameter count: {count_parameters(net_small):,}")

    # Test large network
    print("\n=== Large Network ===")
    net_large = AlphaZeroNetLarge(action_space_size=action_space)
    policy_large, value_large = net_large(x)
    print(f"Policy: {policy_large.shape}, Value: {value_large.shape}")
    print(f"Parameter count: {count_parameters(net_large):,}")

    # Test save/load
    print("\n=== Save/Load ===")
    save_checkpoint(net, "test_checkpoint.pth")
    net_loaded = load_checkpoint("test_checkpoint.pth")
    policy_loaded, value_loaded = net_loaded(x)
    assert torch.allclose(policy_logits, policy_loaded)
    assert torch.allclose(value, value_loaded)
    print("Save/load test passed!")

    print("\n✓ All AlphaZero network tests passed!")
