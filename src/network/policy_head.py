"""
Policy Head for AlphaZero Network

Outputs action probabilities over the action space (2086 possible moves).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    Policy head for the AlphaZero network.

    Takes the feature map from the backbone and outputs logits for each possible action.

    Architecture:
        Input (C, H, W) -> Conv1x1 -> Flatten -> Linear -> Action Logits
    """

    def __init__(
        self,
        input_channels: int = 256,
        action_space_size: int = 2086,
        board_height: int = 10,
        board_width: int = 9
    ):
        """
        Initialize policy head.

        Args:
            input_channels: Number of channels from backbone
            action_space_size: Size of action space (number of possible moves)
            board_height: Board height (10 for Chinese Chess)
            board_width: Board width (9 for Chinese Chess)
        """
        super(PolicyHead, self).__init__()

        self.input_channels = input_channels
        self.action_space_size = action_space_size
        self.board_size = board_height * board_width

        # Policy head: 1x1 convolution to reduce channels
        self.policy_conv = nn.Conv2d(
            input_channels, 2,
            kernel_size=1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(2)

        # Fully connected layer
        self.policy_fc = nn.Linear(
            2 * self.board_size,
            action_space_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy head.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Logits of shape (batch, action_space_size)
        """
        # 1x1 convolution
        x = self.policy_conv(x)
        x = self.policy_bn(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        logits = self.policy_fc(x)

        return logits


class PolicyHeadLarge(nn.Module):
    """
    Larger policy head with an additional hidden layer.

    This can provide better performance but has more parameters.
    """

    def __init__(
        self,
        input_channels: int = 256,
        action_space_size: int = 2086,
        hidden_dim: int = 256,
        board_height: int = 10,
        board_width: int = 9
    ):
        """Initialize larger policy head."""
        super(PolicyHeadLarge, self).__init__()

        self.board_size = board_height * board_width

        self.policy_conv = nn.Conv2d(input_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)

        self.policy_fc1 = nn.Linear(2 * self.board_size, hidden_dim)
        self.policy_fc2 = nn.Linear(hidden_dim, action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through larger policy head."""
        x = self.policy_conv(x)
        x = self.policy_bn(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.policy_fc1(x))
        logits = self.policy_fc2(x)
        return logits


class PolicyHeadWithAttention(nn.Module):
    """
    Policy head with attention mechanism for better action selection.
    """

    def __init__(
        self,
        input_channels: int = 256,
        action_space_size: int = 2086,
        num_attention_heads: int = 4,
        board_height: int = 10,
        board_width: int = 9
    ):
        """Initialize policy head with attention."""
        super(PolicyHeadWithAttention, self).__init__()

        self.board_size = board_height * board_width
        self.num_heads = num_attention_heads

        # Policy conv
        self.policy_conv = nn.Conv2d(input_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=2,
            num_heads=num_attention_heads,
            batch_first=True
        )

        # Output layers
        self.policy_fc = nn.Linear(2 * self.board_size, action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        x = self.policy_conv(x)
        x = self.policy_bn(x)

        batch_size = x.size(0)
        # Reshape for attention: (batch, seq_len, embed_dim)
        x_flat = x.view(batch_size, 2, -1).permute(0, 2, 1)

        # Apply attention
        x_attn, _ = self.attention(x_flat, x_flat, x_flat)

        # Reshape back
        x = x_attn.permute(0, 2, 1).contiguous().view(batch_size, -1)

        logits = self.policy_fc(x)
        return logits


def create_policy_head(
    head_type: str = "standard",
    input_channels: int = 256,
    action_space_size: int = 2086,
    **kwargs
) -> nn.Module:
    """
    Factory function to create policy heads.

    Args:
        head_type: Type of policy head ("standard", "large", "attention")
        input_channels: Number of input channels
        action_space_size: Size of action space
        **kwargs: Additional arguments for specific head types

    Returns:
        Policy head module
    """
    if head_type == "standard":
        return PolicyHead(input_channels, action_space_size, **kwargs)
    elif head_type == "large":
        return PolicyHeadLarge(input_channels, action_space_size, **kwargs)
    elif head_type == "attention":
        return PolicyHeadWithAttention(input_channels, action_space_size, **kwargs)
    else:
        raise ValueError(f"Unknown policy head type: {head_type}")


if __name__ == "__main__":
    # Test policy heads
    print("Testing PolicyHead...")

    batch_size = 4
    channels = 256
    height, width = 10, 9
    action_space = 2086

    x = torch.randn(batch_size, channels, height, width)

    # Standard policy head
    policy = PolicyHead(channels, action_space)
    logits = policy(x)
    print(f"Standard policy head: {x.shape} -> {logits.shape}")
    assert logits.shape == (batch_size, action_space)

    # Test softmax
    probs = F.softmax(logits, dim=1)
    print(f"Probabilities sum: {probs.sum(dim=1)}")
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))

    # Large policy head
    policy_large = PolicyHeadLarge(channels, action_space, hidden_dim=256)
    logits_large = policy_large(x)
    print(f"Large policy head: {x.shape} -> {logits_large.shape}")
    assert logits_large.shape == (batch_size, action_space)

    # Attention policy head
    policy_attn = PolicyHeadWithAttention(channels, action_space, num_attention_heads=2)
    logits_attn = policy_attn(x)
    print(f"Attention policy head: {x.shape} -> {logits_attn.shape}")
    assert logits_attn.shape == (batch_size, action_space)

    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter counts:")
    print(f"  Standard: {count_parameters(policy):,}")
    print(f"  Large: {count_parameters(policy_large):,}")
    print(f"  Attention: {count_parameters(policy_attn):,}")

    print("\n✓ All policy heads test passed!")
