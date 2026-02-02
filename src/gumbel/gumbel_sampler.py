"""
Gumbel Sampler for Gumbel AlphaZero

Implements Gumbel noise sampling for exploration in the Sequential Halving algorithm.
"""

import torch
import numpy as np
from typing import Tuple


def sample_gumbel(shape: Tuple[int, ...], eps: float = 1e-20) -> torch.Tensor:
    """
    Sample from Gumbel(0, 1) distribution.

    Gumbel distribution is used for the Gumbel-Max trick:
        g_i = -log(-log(u_i)), where u_i ~ Uniform(0, 1)

    This can be used to sample from a categorical distribution by:
        argmax_i (log(p_i) + g_i)

    Args:
        shape: Shape of the output tensor
        eps: Small constant for numerical stability

    Returns:
        Tensor of Gumbel noise samples
    """
    # Sample from uniform distribution
    u = torch.rand(shape) + eps

    # Apply inverse CDF of Gumbel distribution
    # Gumbel(0,1): g = -log(-log(u))
    return -torch.log(-torch.log(u))


def add_gumbel_noise(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Add Gumbel noise to logits for stochastic selection.

    Args:
        logits: Logits tensor (before softmax)
        temperature: Temperature parameter for scaling

    Returns:
        Logits with Gumbel noise added
    """
    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Sample and add Gumbel noise
    gumbel_noise = sample_gumbel(logits.shape)

    return scaled_logits + gumbel_noise


def gumbel_softmax_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False
) -> torch.Tensor:
    """
    Sample from categorical distribution using Gumbel-Max trick.

    Args:
        logits: Logits tensor of shape (batch_size, num_categories)
        temperature: Temperature parameter (lower -> more deterministic)
        hard: If True, returns one-hot samples using straight-through estimator

    Returns:
        Sampled probabilities of same shape as logits
    """
    # Add Gumbel noise
    noisy_logits = add_gumbel_noise(logits, temperature)

    # Apply softmax
    samples = torch.softmax(noisy_logits, dim=-1)

    if hard:
        # Straight-through Gumbel-Softmax
        # Create one-hot with gradient through soft samples
        index = samples.max(dim=-1, keepdim=True)[1]
        hard = torch.zeros_like(samples).scatter_(-1, index, 1.0)
        samples = hard - samples.detach() + samples

    return samples


def gumbel_top_k(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k actions using Gumbel noise.

    This is the core operation for Sequential Halving in Gumbel AlphaZero.

    Args:
        logits: Policy logits of shape (action_space_size,)
        k: Number of top actions to select
        temperature: Temperature for scaling

    Returns:
        Tuple of (top_k_indices, top_k_values)
    """
    # Add Gumbel noise
    noisy_logits = add_gumbel_noise(logits, temperature)

    # Get top-k
    top_k_values, top_k_indices = torch.topk(noisy_logits, k)

    return top_k_indices, top_k_values


def sample_with_nucleus(
    logits: torch.Tensor,
    p: float = 0.95,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Nucleus sampling with Gumbel noise.

    Select from the smallest set of top tokens whose cumulative probability
    exceeds threshold p.

    Args:
        logits: Policy logits
        p: Nucleus sampling threshold (0 < p < 1)
        temperature: Temperature parameter

    Returns:
        Tuple of (sampled_indices, probabilities)
    """
    # Add Gumbel noise and apply softmax
    noisy_logits = add_gumbel_noise(logits, temperature)
    probs = torch.softmax(noisy_logits, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Find cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p

    # Shift to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Set probabilities of removed tokens to 0
    sorted_probs[sorted_indices_to_remove] = 0.0

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample from the filtered distribution
    sample_idx = torch.multinomial(sorted_probs, 1)

    return sorted_indices[sample_idx], sorted_probs[sample_idx]


def deterministic_gumbel_max(
    logits: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """
    Deterministic Gumbel-Max using a fixed seed.

    Useful for reproducibility during testing.

    Args:
        logits: Policy logits

    Returns:
        Tuple of (argmax_index, max_value)
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    noisy_logits = add_gumbel_noise(logits)
    argmax_idx = torch.argmax(noisy_logits)

    return argmax_idx.item(), noisy_logits[argmax_idx]


class GumbelSampler:
    """
    Gumbel sampler with configurable parameters.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        seed: int = None
    ):
        """
        Initialize Gumbel sampler.

        Args:
            temperature: Temperature parameter
            seed: Random seed (None for random)
        """
        self.temperature = temperature

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from logits with Gumbel noise."""
        return add_gumbel_noise(logits, self.temperature)

    def sample_top_k(self, logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample top-k actions."""
        return gumbel_top_k(logits, k, self.temperature)

    def sample_softmax(self, logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Sample using Gumbel-Softmax."""
        return gumbel_softmax_sample(logits, self.temperature, hard)

    def set_temperature(self, temperature: float):
        """Update temperature parameter."""
        self.temperature = temperature


if __name__ == "__main__":
    # Test Gumbel sampler
    print("Testing Gumbel Sampler...")

    # Create test logits
    logits = torch.randn(10)

    print(f"\nOriginal logits: {logits}")
    print(f"Original probs: {torch.softmax(logits, dim=-1)}")

    # Test Gumbel noise sampling
    gumbel_noise = sample_gumbel((10,))
    print(f"\nGumbel noise: {gumbel_noise}")

    # Test adding Gumbel noise
    noisy_logits = add_gumbel_noise(logits)
    print(f"\nNoisy logits: {noisy_logits}")

    # Test Gumbel top-k
    k = 3
    top_k_indices, top_k_values = gumbel_top_k(logits, k)
    print(f"\nTop {k} indices: {top_k_indices}")
    print(f"Top {k} values: {top_k_values}")

    # Test Gumbel-Softmax
    soft_samples = gumbel_softmax_sample(logits.unsqueeze(0))
    print(f"\nSoftmax samples: {soft_samples}")
    print(f"Sum: {soft_samples.sum()}")

    # Test hard Gumbel-Softmax
    hard_samples = gumbel_softmax_sample(logits.unsqueeze(0), hard=True)
    print(f"\nHard samples: {hard_samples}")
    print(f"One-hot: {hard_samples[hard_samples == 1].nonzero()}")

    # Test Gumbel sampler class
    sampler = GumbelSampler(temperature=1.0)
    top_k = sampler.sample_top_k(logits, 5)
    print(f"\nSampler top-k: {top_k}")

    print("\n✓ Gumbel sampler tests passed!")
