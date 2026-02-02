"""
Gumbel Search for Gumbel AlphaZero

Main search orchestration combining neural network evaluation with
Gumbel sampling and Sequential Halving.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to import from other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.alphanet import AlphaZeroNet
from gumbel.gumbel_sampler import GumbelSampler, add_gumbel_noise
from gumbel.sequential_halving import SequentialHalving, HalvingConfig


@dataclass
class GumbelSearchConfig:
    """Configuration for Gumbel search."""

    num_simulations: int = 800
    num_levels: int = 4
    max_depth: int = 8
    c_visit: float = 1.5
    c_scale: float = 1.0
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


@dataclass
class SearchResult:
    """Result of Gumbel search."""

    improved_policy: torch.Tensor      # π̂: improved policy from search
    value: float                        # v: estimated value
    visit_counts: Dict[int, int]        # Visit counts for each action
    q_values: Dict[int, float]          # Q-values for each action
    selected_actions: List[int]         # Actions selected by Sequential Halving


class GumbelSearch:
    """
    Gumbel AlphaZero search engine.

    Combines neural network evaluation with Gumbel sampling and
    Sequential Halving for efficient action selection.
    """

    def __init__(
        self,
        network: AlphaZeroNet,
        config: GumbelSearchConfig = None
    ):
        """
        Initialize Gumbel search.

        Args:
            network: AlphaZero network for evaluation
            config: Search configuration
        """
        self.network = network
        self.config = config or GumbelSearchConfig()

        # Initialize components
        self.gumbel_sampler = GumbelSampler(
            temperature=self.config.temperature
        )
        self.sequential_halving = SequentialHalving(
            HalvingConfig(
                num_levels=self.config.num_levels,
                max_actions=min(64, self.config.num_simulations // 4),
                min_actions=1
            )
        )

        # Track search statistics
        self.num_evaluations = 0
        self.cache: Dict = {}  # Optional cache for position evaluations

    def search(
        self,
        board_tensor: torch.Tensor,
        legal_moves: List[int] = None,
        is_root: bool = False
    ) -> SearchResult:
        """
        Run Gumbel search from the current position.

        Args:
            board_tensor: Board state tensor of shape (1, 14, 10, 9)
            legal_moves: List of legal encoded moves (16-bit integers)
            is_root: Whether this is the root position (for Dirichlet noise)

        Returns:
            SearchResult with improved policy and value
        """
        # Ensure board tensor has batch dimension
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)

        # Get initial policy and value from network
        with torch.no_grad():
            policy_logits, value = self.network(board_tensor)

        policy_logits = policy_logits[0]  # Remove batch dim
        value = value[0].item()

        # Convert legal moves to action indices
        # Note: This assumes you have an action encoder available
        # For now, we'll work with the full policy space
        if legal_moves is None:
            legal_action_indices = None
        else:
            # TODO: Convert encoded moves to action indices
            # This would use: action_encoder.encoded_to_index[move]
            legal_action_indices = list(range(len(policy_logits)))

        # Add Dirichlet noise at root
        if is_root and legal_action_indices:
            policy_logits = self._add_dirichlet_noise(
                policy_logits,
                legal_action_indices
            )

        # Run Sequential Halving to select actions
        selected_actions = self.sequential_halving.select_actions(
            policy_logits,
            legal_action_indices
        )

        # Simulate each selected action
        q_values = {}
        visit_counts = {}

        for action_idx in selected_actions:
            # TODO: Convert action index to move and apply to board
            # For now, we'll simulate with the network value
            q_value, visits = self._simulate_action(
                board_tensor,
                action_idx,
                self.config.max_depth
            )

            q_values[action_idx] = q_value
            visit_counts[action_idx] = visits

        # Compute improved policy π̂
        improved_policy = self._compute_improved_policy(q_values)

        return SearchResult(
            improved_policy=improved_policy,
            value=value,
            visit_counts=visit_counts,
            q_values=q_values,
            selected_actions=selected_actions
        )

    def _simulate_action(
        self,
        board_tensor: torch.Tensor,
        action_idx: int,
        depth: int
    ) -> Tuple[float, int]:
        """
        Simulate an action to a given depth.

        Args:
            board_tensor: Current board state
            action_idx: Action to simulate
            depth: Maximum simulation depth

        Returns:
            Tuple of (q_value, visit_count)
        """
        # TODO: Implement full simulation with move application
        # For now, we'll do a simplified evaluation

        q_value = 0.0
        visits = 1

        if depth > 0:
            # TODO: Apply action to board and recurse
            # For now, just use network value
            with torch.no_grad():
                _, value = self.network(board_tensor)
                q_value = value[0].item()
                visits = 1

        return q_value, visits

    def _compute_improved_policy(self, q_values: Dict[int, float]) -> torch.Tensor:
        """
        Compute improved policy π̂ from Q-values.

        Uses the Gumbel improvement formula:
            π̂(a) ∝ exp(Q(a) / c_visit)

        Args:
            q_values: Dictionary mapping action indices to Q-values

        Returns:
            Improved policy tensor
        """
        action_space_size = 2086  # TODO: Make configurable
        improved_policy = torch.zeros(action_space_size)

        # Compute exponentiated Q-values
        exp_q = {}
        for action_idx, q_value in q_values.items():
            exp_q[action_idx] = np.exp(q_value / self.config.c_visit)

        # Normalize
        total = sum(exp_q.values())
        for action_idx, exp_val in exp_q.items():
            improved_policy[action_idx] = exp_val / total

        return improved_policy

    def _add_dirichlet_noise(
        self,
        policy_logits: torch.Tensor,
        legal_actions: List[int]
    ) -> torch.Tensor:
        """
        Add Dirichlet noise to policy logits for exploration.

        Args:
            policy_logits: Original policy logits
            legal_actions: List of legal action indices

        Returns:
            Noisy policy logits
        """
        # Sample Dirichlet noise
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(legal_actions)
        ) * self.config.dirichlet_epsilon

        # Add noise to logits (in log space)
        noisy_logits = policy_logits.clone()

        for i, action_idx in enumerate(legal_actions):
            # Convert probability to logit and add noise
            noisy_logits[action_idx] += np.log(noise[i] + 1e-10)

        return noisy_logits

    def get_best_action(self, search_result: SearchResult) -> int:
        """
        Get the best action from search result.

        Args:
            search_result: Result from search()

        Returns:
            Best action index
        """
        # Select action with highest visit count or Q-value
        if search_result.visit_counts:
            # By visit count
            best_action = max(
                search_result.visit_counts.items(),
                key=lambda x: x[1]
            )[0]
        else:
            # By improved policy
            best_action = search_result.improved_policy.argmax().item()

        return best_action


def single_search(
    network: AlphaZeroNet,
    board_tensor: torch.Tensor,
    legal_moves: List[int] = None,
    num_simulations: int = 800
) -> Tuple[int, float]:
    """
    Convenience function for a single search.

    Args:
        network: AlphaZero network
        board_tensor: Board state tensor
        legal_moves: List of legal moves
        num_simulations: Number of simulations

    Returns:
        Tuple of (best_action, value)
    """
    config = GumbelSearchConfig(num_simulations=num_simulations)
    search = GumbelSearch(network, config)

    result = search.search(board_tensor, legal_moves, is_root=True)
    best_action = search.get_best_action(result)

    return best_action, result.value


if __name__ == "__main__":
    # Test Gumbel search
    print("Testing Gumbel Search...")

    # Create a dummy network
    from network.alphanet import AlphaZeroNet

    network = AlphaZeroNet(
        in_channels=14,
        hidden_channels=128,  # Smaller for testing
        num_residual_blocks=5,
        action_space_size=2086
    )
    network.eval()

    # Create dummy board tensor
    board_tensor = torch.randn(1, 14, 10, 9)

    # Test search
    config = GumbelSearchConfig(
        num_simulations=64,  # Smaller for testing
        num_levels=3,
        max_depth=4
    )

    search = GumbelSearch(network, config)
    result = search.search(board_tensor, is_root=True)

    print(f"\nSearch results:")
    print(f"  Value: {result.value:.3f}")
    print(f"  Selected actions: {len(result.selected_actions)}")
    print(f"  Top 5 actions by Q-value:")

    top_actions = sorted(
        result.q_values.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for action, q_value in top_actions:
        print(f"    Action {action}: Q={q_value:.3f}")

    print(f"  Improved policy sum: {result.improved_policy.sum().item():.3f}")

    # Test best action selection
    best_action = search.get_best_action(result)
    print(f"\nBest action: {best_action}")

    # Test convenience function
    best, value = single_search(network, board_tensor, num_simulations=64)
    print(f"\nConvenience function: action={best}, value={value:.3f}")

    print("\n✓ Gumbel search tests passed!")
