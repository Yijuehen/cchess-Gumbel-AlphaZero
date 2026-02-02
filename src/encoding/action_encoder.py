"""
Action Encoder for Gumbel AlphaZero

Handles bidirectional mapping between 4-digit move strings and action indices (0-2085).
The action space includes all possible moves in Chinese Chess.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set
from collections import defaultdict
import sys
from pathlib import Path

# Add parent directory to path to import from Alpha-Beta+NNUE
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "cchess-Alpha-Beta+NNUE" / "src"))

from board.board_representation import ROWS, COLS, parse_board
from movegen.move_generator import (
    generate_moves, encode_move, decode_move,
    format_move, parse_move as parse_move_str
)


class ActionEncoder:
    """
    Bidirectional action encoder for Chinese Chess moves.

    Handles mapping between:
    - 4-digit move strings (e.g., "7747" = from position 77 to 47)
    - Action indices (0 to action_space_size - 1)
    - Encoded moves (16-bit integers)
    """

    def __init__(self, build_from_samples: bool = False, sample_positions: Optional[List[str]] = None):
        """
        Initialize action encoder.

        Args:
            build_from_samples: If True, build mapping from sample positions
            sample_positions: List of board positions to sample moves from
        """
        self.move_to_index: Dict[str, int] = {}
        self.index_to_move: Dict[int, str] = {}
        self.encoded_to_index: Dict[int, int] = {}
        self.index_to_encoded: Dict[int, int] = {}
        self._action_space_size = 0
        self._from_position_moves: Dict[int, Set[int]] = defaultdict(set)  # from_pos -> set of to_pos
        self._to_position_moves: Dict[int, Set[int]] = defaultdict(set)    # to_pos -> set of from_pos

        if build_from_samples and sample_positions:
            self._build_from_samples(sample_positions)
        else:
            # Build theoretical action space (all possible from/to combinations)
            self._build_theoretical()

    def _build_theoretical(self):
        """Build action space from all theoretical move combinations."""
        # Theoretical maximum: 90 positions × 90 positions = 8100 possible moves
        # But actual legal moves are much fewer (~2086)
        # We'll build a complete mapping first, then can prune during training

        index = 0
        for from_pos in range(90):
            for to_pos in range(90):
                if from_pos != to_pos:  # Can't move to same position
                    move_str = f"{from_pos:02d}{to_pos:02d}"
                    encoded = encode_move(from_pos, to_pos)

                    self.move_to_index[move_str] = index
                    self.index_to_move[index] = move_str
                    self.encoded_to_index[encoded] = index
                    self.index_to_encoded[index] = encoded

                    self._from_position_moves[from_pos].add(to_pos)
                    self._to_position_moves[to_pos].add(from_pos)

                    index += 1

        self._action_space_size = index

    def _build_from_samples(self, sample_positions: List[str]):
        """
        Build action space from actual legal moves in sample positions.

        This creates a more compact action space with only actually legal moves.
        """
        unique_moves = set()

        for board_str in sample_positions:
            try:
                board = parse_board(board_str)

                # Generate all moves for Red
                red_moves = generate_moves(board, color=1)
                unique_moves.update(red_moves)

                # Generate all moves for Black
                black_moves = generate_moves(board, color=-1)
                unique_moves.update(black_moves)

            except Exception as e:
                print(f"Warning: Failed to parse board: {e}")
                continue

        # Sort for consistent indexing
        sorted_moves = sorted(unique_moves)

        # Build mappings
        for index, encoded_move in enumerate(sorted_moves):
            from_pos, to_pos = decode_move(encoded_move)
            move_str = format_move(encoded_move)

            self.move_to_index[move_str] = index
            self.index_to_move[index] = move_str
            self.encoded_to_index[encoded_move] = index
            self.index_to_encoded[index] = encoded_move

            self._from_position_moves[from_pos].add(to_pos)
            self._to_position_moves[to_pos].add(from_pos)

        self._action_space_size = len(sorted_moves)

    def encode(self, move_str: str) -> int:
        """
        Encode 4-digit move string to action index.

        Args:
            move_str: 4-digit move string (e.g., "7747")

        Returns:
            Action index (0 to action_space_size - 1)

        Raises:
            KeyError: If move is not in action space
        """
        if move_str not in self.move_to_index:
            raise KeyError(f"Move '{move_str}' not in action space")
        return self.move_to_index[move_str]

    def decode(self, action_idx: int) -> str:
        """
        Decode action index to 4-digit move string.

        Args:
            action_idx: Action index (0 to action_space_size - 1)

        Returns:
            4-digit move string (e.g., "7747")

        Raises:
            KeyError: If action index is out of range
        """
        if action_idx not in self.index_to_move:
            raise KeyError(f"Action index {action_idx} not in action space")
        return self.index_to_move[action_idx]

    def encode_move(self, from_pos: int, to_pos: int) -> int:
        """
        Encode from/to positions to action index.

        Args:
            from_pos: From position (0-89)
            to_pos: To position (0-89)

        Returns:
            Action index
        """
        encoded = encode_move(from_pos, to_pos)
        return self.encoded_to_index.get(encoded, -1)

    def decode_move(self, action_idx: int) -> tuple:
        """
        Decode action index to (from_pos, to_pos).

        Args:
            action_idx: Action index

        Returns:
            Tuple of (from_pos, to_pos)
        """
        encoded = self.index_to_encoded.get(action_idx, -1)
        return decode_move(encoded) if encoded != -1 else (-1, -1)

    def get_action_space_size(self) -> int:
        """Get the total action space size."""
        return self._action_space_size

    def is_valid_move(self, move_str: str) -> bool:
        """Check if a move string is in the action space."""
        return move_str in self.move_to_index

    def is_valid_action(self, action_idx: int) -> bool:
        """Check if an action index is valid."""
        return 0 <= action_idx < self._action_space_size

    def get_moves_from_position(self, from_pos: int) -> List[int]:
        """Get all possible to_positions for a given from_position."""
        return list(self._from_position_moves.get(from_pos, set()))

    def get_moves_to_position(self, to_pos: int) -> List[int]:
        """Get all possible from_positions for a given to_position."""
        return list(self._to_position_moves.get(to_pos, set()))

    def create_action_mask(self, legal_moves: List[int]) -> torch.Tensor:
        """
        Create a binary mask for legal actions.

        Args:
            legal_moves: List of encoded moves (16-bit integers)

        Returns:
            Boolean tensor of shape (action_space_size,) where True means legal
        """
        mask = torch.zeros(self._action_space_size, dtype=torch.bool)

        for encoded_move in legal_moves:
            action_idx = self.encoded_to_index.get(encoded_move, -1)
            if action_idx != -1:
                mask[action_idx] = True

        return mask

    def mask_policy_logits(self, policy_logits: torch.Tensor, legal_moves: List[int]) -> torch.Tensor:
        """
        Mask policy logits to only include legal moves.

        Illegal moves get a large negative logit so they have near-zero probability.

        Args:
            policy_logits: Tensor of shape (action_space_size,)
            legal_moves: List of encoded moves (16-bit integers)

        Returns:
            Masked policy logits with illegal moves set to very negative values
        """
        if policy_logits.shape[0] != self._action_space_size:
            raise ValueError(
                f"Policy logits size {policy_logits.shape[0]} doesn't match "
                f"action space size {self._action_space_size}"
            )

        # Create mask
        mask = self.create_action_mask(legal_moves)

        # Apply mask: illegal moves get -inf
        masked_logits = policy_logits.clone()
        masked_logits[~mask] = -1e9

        return masked_logits

    def batch_encode(self, move_strings: List[str]) -> List[int]:
        """Encode multiple move strings to indices."""
        return [self.encode(move) for move in move_strings]

    def batch_decode(self, action_indices: List[int]) -> List[str]:
        """Decode multiple action indices to move strings."""
        return [self.decode(idx) for idx in action_indices]

    def save(self, filepath: str):
        """Save action encoder to file."""
        import pickle

        data = {
            'move_to_index': self.move_to_index,
            'index_to_move': self.index_to_move,
            'encoded_to_index': self.encoded_to_index,
            'index_to_encoded': self.index_to_encoded,
            'action_space_size': self._action_space_size,
            'from_position_moves': {k: list(v) for k, v in self._from_position_moves.items()},
            'to_position_moves': {k: list(v) for k, v in self._to_position_moves.items()},
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Action encoder saved to {filepath}")

    def load(self, filepath: str):
        """Load action encoder from file."""
        import pickle

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.move_to_index = data['move_to_index']
        self.index_to_move = data['index_to_move']
        self.encoded_to_index = data['encoded_to_index']
        self.index_to_encoded = data['index_to_encoded']
        self._action_space_size = data['action_space_size']
        self._from_position_moves = {k: set(v) for k, v in data['from_position_moves'].items()}
        self._to_position_moves = {k: set(v) for k, v in data['to_position_moves'].items()}

        print(f"Action encoder loaded from {filepath}")
        print(f"Action space size: {self._action_space_size}")

    def __len__(self) -> int:
        """Get action space size."""
        return self._action_space_size

    def __contains__(self, move_str: str) -> bool:
        """Check if move is in action space."""
        return move_str in self.move_to_index


# Global default action encoder
_default_encoder: Optional[ActionEncoder] = None


def get_default_encoder() -> ActionEncoder:
    """Get or create the default action encoder."""
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = ActionEncoder()
    return _default_encoder


def encode_move_to_action(move_str: str) -> int:
    """Encode move string to action index using default encoder."""
    return get_default_encoder().encode(move_str)


def decode_action_to_move(action_idx: int) -> str:
    """Decode action index to move string using default encoder."""
    return get_default_encoder().decode(action_idx)


if __name__ == "__main__":
    # Test action encoder
    print("Testing ActionEncoder...")

    # Create encoder
    encoder = ActionEncoder()
    print(f"Action space size: {encoder.get_action_space_size()}")

    # Test encoding/decoding
    test_moves = ["7747", "7062", "7967", "8070"]
    print(f"\nTesting moves: {test_moves}")

    for move_str in test_moves:
        action_idx = encoder.encode(move_str)
        decoded = encoder.decode(action_idx)
        print(f"  {move_str} -> {action_idx} -> {decoded}")
        assert move_str == decoded, f"Round-trip failed: {move_str} != {decoded}"

    # Test position queries
    print(f"\nMoves from position 77: {encoder.get_moves_from_position(77)}")
    print(f"Moves to position 47: {encoder.get_moves_to_position(47)}")

    # Test action mask
    legal_moves = [encode_move(77, 47), encode_move(70, 62)]
    mask = encoder.create_action_mask(legal_moves)
    print(f"\nLegal moves mask shape: {mask.shape}")
    print(f"Number of legal actions: {mask.sum().item()}")

    # Test policy masking
    policy_logits = torch.randn(encoder.get_action_space_size())
    masked = encoder.mask_policy_logits(policy_logits, legal_moves)
    print(f"Masked logits - inf count: {(masked == -1e9).sum().item()}")
    print(f"Masked logits finite count: {(masked != -1e9).sum().item()}")

    # Test save/load
    encoder.save("test_action_encoder.pkl")
    encoder2 = ActionEncoder()
    encoder2.load("test_action_encoder.pkl")

    assert encoder2.get_action_space_size() == encoder.get_action_space_size()
    print("\n✓ Save/load test passed!")

    print("\n✓ Action encoder test passed!")
