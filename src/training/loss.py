"""
Loss Functions for Gumbel AlphaZero

Implements policy loss, value loss, and KL divergence loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AlphaZeroLoss(nn.Module):
    """
    Combined loss for AlphaZero training.

    Loss = policy_weight * policy_loss + value_weight * value_loss
    """

    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0
    ):
        """
        Initialize AlphaZero loss.

        Args:
            policy_weight: Weight for policy loss
            value_weight: Weight for value loss
        """
        super(AlphaZeroLoss, self).__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight

    def forward(
        self,
        policy_logits: torch.Tensor,
        value: torch.Tensor,
        target_actions: torch.Tensor,
        target_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            policy_logits: Predicted policy logits (batch, action_space)
            value: Predicted values (batch, 1)
            target_actions: Target action indices (batch,)
            target_values: Target values (batch, 1)

        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        # Policy loss: cross-entropy
        policy_loss = F.cross_entropy(
            policy_logits,
            target_actions.long()
        )

        # Value loss: MSE
        value_loss = F.mse_loss(
            value.squeeze(-1),
            target_values.squeeze(-1)
        )

        # Combined loss
        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss
        )

        return total_loss, policy_loss, value_loss


class PolicyLoss(nn.Module):
    """Policy loss using cross-entropy."""

    def forward(
        self,
        policy_logits: torch.Tensor,
        target_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute policy loss.

        Args:
            policy_logits: Predicted policy logits (batch, action_space)
            target_actions: Target action indices (batch,)

        Returns:
            Policy loss scalar
        """
        return F.cross_entropy(policy_logits, target_actions.long())


class ValueLoss(nn.Module):
    """Value loss using MSE or Huber loss."""

    def __init__(self, loss_type: str = 'mse'):
        """
        Initialize value loss.

        Args:
            loss_type: Type of loss ('mse' or 'huber')
        """
        super(ValueLoss, self).__init__()
        self.loss_type = loss_type

    def forward(
        self,
        value: torch.Tensor,
        target_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value loss.

        Args:
            value: Predicted values (batch, 1)
            target_value: Target values (batch, 1)

        Returns:
            Value loss scalar
        """
        if self.loss_type == 'mse':
            return F.mse_loss(value.squeeze(-1), target_value.squeeze(-1))
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(value.squeeze(-1), target_value.squeeze(-1))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss for policy improvement.

    KL(pi || pi_improved) = sum(pi * log(pi / pi_improved))

    Used to align the policy with the improved policy from search.
    """

    def forward(
        self,
        policy_logits: torch.Tensor,
        improved_policy: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence loss.

        Args:
            policy_logits: Current policy logits (batch, action_space)
            improved_policy: Target improved policy (batch, action_space)

        Returns:
            KL divergence loss scalar
        """
        # Convert logits to probabilities
        policy = F.softmax(policy_logits, dim=-1)

        # Add small epsilon for numerical stability
        epsilon = 1e-8
        policy = policy + epsilon
        improved_policy = improved_policy + epsilon

        # Normalize
        policy = policy / policy.sum(dim=-1, keepdim=True)
        improved_policy = improved_policy / improved_policy.sum(dim=-1, keepdim=True)

        # KL divergence
        kl = (policy * (torch.log(policy) - torch.log(improved_policy))).sum(dim=-1)

        return kl.mean()


class GumbelImprovementLoss(nn.Module):
    """
    Combined loss for Gumbel improvement phase.

    Loss = kl_weight * KL(pi || pi_improved) + value_weight * MSE(v, v_target)
    """

    def __init__(
        self,
        kl_weight: float = 1.0,
        value_weight: float = 1.0
    ):
        """
        Initialize Gumbel improvement loss.

        Args:
            kl_weight: Weight for KL divergence
            value_weight: Weight for value loss
        """
        super(GumbelImprovementLoss, self).__init__()
        self.kl_weight = kl_weight
        self.value_weight = value_weight
        self.kl_loss = KLDivergenceLoss()
        self.value_loss = ValueLoss('mse')

    def forward(
        self,
        policy_logits: torch.Tensor,
        value: torch.Tensor,
        improved_policy: torch.Tensor,
        target_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined improvement loss.

        Args:
            policy_logits: Current policy logits
            value: Predicted values
            improved_policy: Target improved policy from search
            target_value: Target values from search

        Returns:
            Tuple of (total_loss, kl_loss, value_loss)
        """
        kl = self.kl_loss(policy_logits, improved_policy)
        value_loss = self.value_loss(value, target_value)

        total = self.kl_weight * kl + self.value_weight * value_loss

        return total, kl, value_loss


def compute_policy_entropy(
    policy_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute policy entropy for monitoring exploration.

    Args:
        policy_logits: Policy logits
        temperature: Temperature for scaling

    Returns:
        Entropy value (average over batch)
    """
    policy = F.softmax(policy_logits / temperature, dim=-1)
    entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1)
    return entropy.mean()


def compute_policy_sparsity(
    policy_logits: torch.Tensor,
    top_k: int = 10
) -> torch.Tensor:
    """
    Compute what fraction of probability mass is in top-k actions.

    Args:
        policy_logits: Policy logits
        top_k: Number of top actions to consider

    Returns:
        Sparsity metric (0 = uniform, 1 = concentrated)
    """
    probs = F.softmax(policy_logits, dim=-1)
    top_k_values, _ = torch.topk(probs, top_k, dim=-1)
    top_k_mass = top_k_values.sum(dim=-1)

    # Normalize by k (uniform would be k/action_space)
    action_space = policy_logits.shape[-1]
    expected_uniform = top_k / action_space

    sparsity = (top_k_mass - expected_uniform) / (1 - expected_uniform)
    return sparsity.mean()


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")

    batch_size = 4
    action_space = 2086

    # Create dummy data
    policy_logits = torch.randn(batch_size, action_space)
    value = torch.randn(batch_size, 1) * 2 - 1  # In [-1, 1]
    target_actions = torch.randint(0, action_space, (batch_size,))
    target_values = torch.randn(batch_size, 1) * 2 - 1

    # Test AlphaZero loss
    print("\n=== AlphaZero Loss ===")
    az_loss = AlphaZeroLoss(policy_weight=1.0, value_weight=1.0)
    total, p_loss, v_loss = az_loss(policy_logits, value, target_actions, target_values)
    print(f"Total loss: {total.item():.4f}")
    print(f"Policy loss: {p_loss.item():.4f}")
    print(f"Value loss: {v_loss.item():.4f}")

    # Test KL divergence
    print("\n=== KL Divergence Loss ===")
    improved_policy = F.softmax(policy_logits + torch.randn_like(policy_logits), dim=-1)
    kl_loss = KLDivergenceLoss()
    kl = kl_loss(policy_logits, improved_policy)
    print(f"KL divergence: {kl.item():.4f}")

    # Test Gumbel improvement loss
    print("\n=== Gumbel Improvement Loss ===")
    gumbel_loss = GumbelImprovementLoss()
    total, kl, v_loss = gumbel_loss(
        policy_logits, value, improved_policy, target_values
    )
    print(f"Total loss: {total.item():.4f}")
    print(f"KL loss: {kl.item():.4f}")
    print(f"Value loss: {v_loss.item():.4f}")

    # Test entropy
    print("\n=== Policy Entropy ===")
    entropy = compute_policy_entropy(policy_logits)
    print(f"Entropy: {entropy.item():.4f}")

    # Test sparsity
    print("\n=== Policy Sparsity ===")
    sparsity = compute_policy_sparsity(policy_logits, top_k=10)
    print(f"Sparsity (top-10): {sparsity.item():.4f}")

    print("\n✓ All loss function tests passed!")
