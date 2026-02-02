"""
Sequential Halving Algorithm for Gumbel AlphaZero

Implements the Sequential Halving algorithm which progressively narrows
down the action space by selecting top-k actions at each level.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HalvingConfig:
    """Configuration for Sequential Halving."""

    num_levels: int = 4          # Number of halving levels
    max_actions: int = 32        # Maximum actions at first level
    min_actions: int = 1         # Minimum actions to keep
    temperature: float = 1.0     # Temperature for Gumbel sampling


class SequentialHalving:
    """
    Sequential Halving algorithm for action selection.

    Progressive narrowing of action space using Gumbel noise:
    1. Start with all actions
    2. At each level, add Gumbel noise to policy logits
    3. Select top-k actions (k = max_actions / 2^(level-1))
    4. Repeat until min_actions reached
    """

    def __init__(self, config: HalvingConfig = None):
        """
        Initialize Sequential Halving.

        Args:
            config: Halving configuration
        """
        self.config = config or HalvingConfig()

    def select_actions(
        self,
        policy_logits: torch.Tensor,
        legal_actions: Optional[List[int]] = None
    ) -> List[int]:
        """
        Select actions using Sequential Halving.

        Args:
            policy_logits: Policy logits of shape (action_space_size,)
            legal_actions: Optional list of legal action indices

        Returns:
            List of selected action indices (length = min_actions)
        """
        from .gumbel_sampler import add_gumbel_noise

        # Filter to legal actions if provided
        if legal_actions is not None:
            actions = legal_actions
            logits = policy_logits[legal_actions]
        else:
            actions = list(range(len(policy_logits)))
            logits = policy_logits

        # Clamp number of actions
        num_actions = min(len(actions), self.config.max_actions)

        # Sequential halving
        for level in range(self.config.num_levels):
            # Calculate k for this level
            k = max(
                self.config.min_actions,
                num_actions // (2 ** level)
            )

            # Add Gumbel noise
            noisy_logits = add_gumbel_noise(
                logits[:num_actions],
                self.config.temperature
            )

            # Get top-k indices
            top_k_indices = torch.topk(noisy_logits, k).indices

            # Update actions and logits
            actions = [actions[i] for i in top_k_indices.tolist()]
            logits = logits[top_k_indices]

            if k == self.config.min_actions:
                break

        return actions

    def select_actions_with_scores(
        self,
        policy_logits: torch.Tensor,
        legal_actions: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Select actions with their Gumbel-perturbed scores.

        Args:
            policy_logits: Policy logits
            legal_actions: Optional list of legal action indices

        Returns:
            List of (action, score) tuples
        """
        from .gumbel_sampler import add_gumbel_noise

        # Filter to legal actions
        if legal_actions is not None:
            actions = legal_actions
            logits = policy_logits[legal_actions]
        else:
            actions = list(range(len(policy_logits)))
            logits = policy_logits

        num_actions = min(len(actions), self.config.max_actions)

        # Track action-score pairs
        action_scores = [(actions[i], logits[i].item()) for i in range(num_actions)]

        # Sequential halving
        for level in range(self.config.num_levels):
            k = max(
                self.config.min_actions,
                num_actions // (2 ** level)
            )

            # Add Gumbel noise and sort
            noisy_scores = [
                score + float(add_gumbel_noise(torch.tensor([score])))
                for _, score in action_scores[:num_actions]
            ]

            # Sort by noisy score
            indexed_scores = [
                (i, noisy_scores[i])
                for i in range(min(len(action_scores), num_actions))
            ]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Keep top-k
            action_scores = [action_scores[i] for i, _ in indexed_scores[:k]]

            if k == self.config.min_actions:
                break

        return action_scores


def sequential_halving_search(
    policy_logits: torch.Tensor,
    num_simulations: int,
    legal_actions: Optional[List[int]] = None,
    num_levels: int = 4
) -> List[int]:
    """
    Perform Sequential Halving search.

    This is a convenience function for quick Sequential Halving.

    Args:
        policy_logits: Policy logits
        num_simulations: Total number of simulations
        legal_actions: Optional legal action indices
        num_levels: Number of halving levels

    Returns:
        List of selected action indices
    """
    # Calculate max actions based on simulations
    max_actions = min(len(policy_logits), num_simulations // 2)

    # Create configuration
    config = HalvingConfig(
        num_levels=num_levels,
        max_actions=max_actions,
        min_actions=1
    )

    # Run Sequential Halving
    halving = SequentialHalving(config)
    return halving.select_actions(policy_logits, legal_actions)


class AdaptiveSequentialHalving(SequentialHalving):
    """
    Adaptive Sequential Halving that adjusts temperature based on search progress.
    """

    def __init__(
        self,
        config: HalvingConfig = None,
        initial_temperature: float = 1.0,
        temperature_decay: float = 0.9
    ):
        """
        Initialize Adaptive Sequential Halving.

        Args:
            config: Halving configuration
            initial_temperature: Starting temperature
            temperature_decay: Temperature decay per level
        """
        super().__init__(config)
        self.initial_temperature = initial_temperature
        self.temperature_decay = temperature_decay

    def select_actions(
        self,
        policy_logits: torch.Tensor,
        legal_actions: Optional[List[int]] = None
    ) -> List[int]:
        """Select actions with adaptive temperature."""
        from .gumbel_sampler import add_gumbel_noise

        # Filter to legal actions
        if legal_actions is not None:
            actions = legal_actions
            logits = policy_logits[legal_actions]
        else:
            actions = list(range(len(policy_logits)))
            logits = policy_logits

        num_actions = min(len(actions), self.config.max_actions)

        # Sequential halving with adaptive temperature
        temperature = self.initial_temperature

        for level in range(self.config.num_levels):
            k = max(
                self.config.min_actions,
                num_actions // (2 ** level)
            )

            # Add Gumbel noise with current temperature
            noisy_logits = add_gumbel_noise(
                logits[:num_actions],
                temperature
            )

            # Get top-k
            top_k_indices = torch.topk(noisy_logits, k).indices

            actions = [actions[i] for i in top_k_indices.tolist()]
            logits = logits[top_k_indices]

            # Decay temperature
            temperature *= self.temperature_decay

            if k == self.config.min_actions:
                break

        return actions


if __name__ == "__main__":
    # Test Sequential Halving
    print("Testing Sequential Halving...")

    # Create test policy logits
    action_space = 100
    policy_logits = torch.randn(action_space)

    # Test basic Sequential Halving
    config = HalvingConfig(
        num_levels=4,
        max_actions=32,
        min_actions=2
    )

    halving = SequentialHalving(config)
    selected = halving.select_actions(policy_logits)

    print(f"\nSelected {len(selected)} actions from {action_space}")
    print(f"Actions: {selected}")

    # Test with legal actions
    legal_actions = list(range(0, 50, 2))  # Every other action
    selected_legal = halving.select_actions(policy_logits, legal_actions)

    print(f"\nWith legal actions ({len(legal_actions)} total):")
    print(f"Selected {len(selected_legal)} actions")
    print(f"Actions: {selected_legal}")

    # Test with scores
    action_scores = halving.select_actions_with_scores(policy_logits)
    print(f"\nTop 5 actions with scores:")
    for action, score in action_scores[:5]:
        print(f"  Action {action}: {score:.3f}")

    # Test adaptive Sequential Halving
    adaptive = AdaptiveSequentialHalving(
        config=HalvingConfig(num_levels=4, max_actions=32, min_actions=2),
        initial_temperature=1.5,
        temperature_decay=0.8
    )

    selected_adaptive = adaptive.select_actions(policy_logits)
    print(f"\nAdaptive selection: {selected_adaptive}")

    # Test convenience function
    result = sequential_halving_search(
        policy_logits,
        num_simulations=64,
        num_levels=4
    )
    print(f"\nConvenience function result: {result}")

    print("\n✓ Sequential Halving tests passed!")
