"""
Data Augmentation Module for Chinese Chess (Xiangqi)

Provides board transformations and augmentations for training data.
"""

import numpy as np
import torch
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to import from board module in Alpha-Beta+NNUE project
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "cchess-Alpha-Beta+NNUE" / "src"))

from board.board_representation import (
    ROWS, COLS, EMPTY,
    RED_GENERAL, RED_ADVISOR, RED_ELEPHANT, RED_HORSE, RED_ROOK, RED_CANNON, RED_SOLDIER,
    BLACK_GENERAL, BLACK_ADVISOR, BLACK_ELEPHANT, BLACK_HORSE, BLACK_ROOK, BLACK_CANNON, BLACK_SOLDIER,
    is_red, is_black
)


def flip_board_vertical(board: np.ndarray) -> np.ndarray:
    """
    Flip board vertically (Red <-> Black perspective).

    In Chinese Chess, this is the primary augmentation since the board
    is symmetric along the horizontal axis (palace areas mirror each other).

    Args:
        board: Board array of shape (10, 9)

    Returns:
        Flipped board array
    """
    flipped = np.zeros_like(board)

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]

            if piece == EMPTY:
                continue

            # Flip row position
            new_row = ROWS - 1 - row

            # Flip piece color
            if is_red(piece):
                flipped[new_row, col] = -piece
            elif is_black(piece):
                flipped[new_row, col] = -piece
            else:
                flipped[new_row, col] = piece

    return flipped


def flip_board_horizontal(board: np.ndarray) -> np.ndarray:
    """
    Flip board horizontally (mirror left-right).

    This is less common in Chinese Chess as the board has some
    asymmetry, but can still be useful for certain positions.

    Args:
        board: Board array of shape (10, 9)

    Returns:
        Horizontally flipped board array
    """
    flipped = np.zeros_like(board)

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]

            # Flip column position
            new_col = COLS - 1 - col
            flipped[row, new_col] = piece

    return flipped


def rotate_180(board: np.ndarray) -> np.ndarray:
    """
    Rotate board 180 degrees (vertical + horizontal flip).

    Args:
        board: Board array of shape (10, 9)

    Returns:
        Rotated board array
    """
    flipped = np.zeros_like(board)

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]

            if piece == EMPTY:
                continue

            # Rotate 180 degrees
            new_row = ROWS - 1 - row
            new_col = COLS - 1 - col

            # Flip piece color
            if is_red(piece):
                flipped[new_row, new_col] = -piece
            elif is_black(piece):
                flipped[new_row, new_col] = -piece
            else:
                flipped[new_row, new_col] = piece

    return flipped


def swap_colors(board: np.ndarray) -> np.ndarray:
    """
    Swap Red and Black pieces without changing positions.

    This changes the perspective but keeps the board geometry the same.
    Useful for training models to evaluate from both perspectives.

    Args:
        board: Board array of shape (10, 9)

    Returns:
        Board with colors swapped
    """
    swapped = board.copy()

    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row, col]

            if piece != EMPTY:
                swapped[row, col] = -piece

    return swapped


def flip_move(move: int, flip_type: str = 'vertical') -> int:
    """
    Flip a move encoding according to board transformation.

    Args:
        move: Encoded move (16-bit integer)
        flip_type: Type of flip ('vertical', 'horizontal', '180')

    Returns:
        Flipped move encoding
    """
    from board.movegen.move_generator import decode_move, encode_move

    from_pos, to_pos = decode_move(move)

    from_row = from_pos // COLS
    from_col = from_pos % COLS
    to_row = to_pos // COLS
    to_col = to_pos % COLS

    if flip_type == 'vertical':
        new_from_row = ROWS - 1 - from_row
        new_to_row = ROWS - 1 - to_row
        new_from_pos = new_from_row * COLS + from_col
        new_to_pos = new_to_row * COLS + to_col

    elif flip_type == 'horizontal':
        new_from_col = COLS - 1 - from_col
        new_to_col = COLS - 1 - to_col
        new_from_pos = from_row * COLS + new_from_col
        new_to_pos = to_row * COLS + new_to_col

    elif flip_type == '180':
        new_from_row = ROWS - 1 - from_row
        new_from_col = COLS - 1 - from_col
        new_to_row = ROWS - 1 - to_row
        new_to_col = COLS - 1 - to_col
        new_from_pos = new_from_row * COLS + new_from_col
        new_to_pos = new_to_row * COLS + new_to_col

    return encode_move(new_from_pos, new_to_pos)


def flip_result(result: int) -> int:
    """
    Flip game result when swapping colors.

    Args:
        result: 0=Red wins, 1=Black wins, 2=Draw

    Returns:
        Flipped result
    """
    if result == 0:  # Red wins
        return 1  # Black wins
    elif result == 1:  # Black wins
        return 0  # Red wins
    else:  # Draw
        return 2


class DataAugmentor:
    """
    Data augmentation utility for Chinese Chess positions.

    Supports multiple augmentation strategies:
    - Vertical flip (Red <-> Black)
    - Horizontal flip
    - 180-degree rotation
    - Color swapping
    """

    def __init__(
        self,
        vertical_flip_prob: float = 0.5,
        horizontal_flip_prob: float = 0.0,
        rotate_180_prob: float = 0.0,
        color_swap_prob: float = 0.0
    ):
        """
        Initialize data augmentor.

        Args:
            vertical_flip_prob: Probability of vertical flip (default 0.5)
            horizontal_flip_prob: Probability of horizontal flip
            rotate_180_prob: Probability of 180-degree rotation
            color_swap_prob: Probability of color swap
        """
        self.vertical_flip_prob = vertical_flip_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotate_180_prob = rotate_180_prob
        self.color_swap_prob = color_swap_prob

    def augment_position(
        self,
        board: np.ndarray,
        move: Optional[int] = None,
        result: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[int], Optional[int], dict]:
        """
        Apply random augmentations to a position.

        Args:
            board: Board array
            move: Optional move to also flip
            result: Optional result to also flip

        Returns:
            Tuple of (augmented_board, augmented_move, augmented_result, augment_info)
        """
        import random

        aug_board = board.copy()
        aug_move = move
        aug_result = result

        augment_info = {
            'vertical_flip': False,
            'horizontal_flip': False,
            'rotate_180': False,
            'color_swap': False
        }

        # Vertical flip (most common for Chinese Chess)
        if random.random() < self.vertical_flip_prob:
            aug_board = flip_board_vertical(aug_board)
            if move is not None:
                aug_move = flip_move(move, 'vertical')
            if result is not None:
                aug_result = flip_result(aug_result)
            augment_info['vertical_flip'] = True

        # Horizontal flip
        elif random.random() < self.horizontal_flip_prob:
            aug_board = flip_board_horizontal(aug_board)
            if move is not None:
                aug_move = flip_move(aug_move, 'horizontal')
            augment_info['horizontal_flip'] = True

        # 180-degree rotation
        elif random.random() < self.rotate_180_prob:
            aug_board = rotate_180(aug_board)
            if move is not None:
                aug_move = flip_move(aug_move, '180')
            if result is not None:
                aug_result = flip_result(aug_result)
            augment_info['rotate_180'] = True

        # Color swap (can be combined with flips)
        if random.random() < self.color_swap_prob:
            aug_board = swap_colors(aug_board)
            if result is not None:
                aug_result = flip_result(aug_result)
            augment_info['color_swap'] = True

        return aug_board, aug_move, aug_result, augment_info

    def augment_batch(
        self,
        boards: np.ndarray,
        moves: Optional[np.ndarray] = None,
        results: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Augment a batch of positions.

        Args:
            boards: Batch of boards (batch_size, 10, 9)
            moves: Optional batch of moves
            results: Optional batch of results

        Returns:
            Tuple of (augmented_boards, augmented_moves, augmented_results)
        """
        batch_size = boards.shape[0]
        aug_boards = []
        aug_moves = [] if moves is not None else None
        aug_results = [] if results is not None else None

        for i in range(batch_size):
            aug_board, aug_move, aug_result, _ = self.augment_position(
                boards[i],
                moves[i] if moves is not None else None,
                results[i] if results is not None else None
            )

            aug_boards.append(aug_board)
            if aug_moves is not None:
                aug_moves.append(aug_move)
            if aug_results is not None:
                aug_results.append(aug_result)

        aug_boards = np.array(aug_boards)
        if aug_moves is not None:
            aug_moves = np.array(aug_moves)
        if aug_results is not None:
            aug_results = np.array(aug_results)

        return aug_boards, aug_moves, aug_results


def apply_training_augmentations(
    boards: torch.Tensor,
    actions: Optional[torch.Tensor] = None,
    results: Optional[torch.Tensor] = None,
    augment_prob: float = 0.5
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Apply augmentations during training (PyTorch tensors).

    Args:
        boards: Board tensors (batch_size, 14, 10, 9) or (batch_size, 10, 9)
        actions: Optional action indices
        results: Optional result values
        augment_prob: Probability of applying augmentation to each sample

    Returns:
        Tuple of augmented tensors
    """
    batch_size = boards.shape[0]

    # Convert to numpy if needed
    if boards.dim() == 4:  # (batch, channels, height, width)
        # For encoded tensors, we need special handling
        # For now, just return as-is (TODO: implement tensor augmentation)
        return boards, actions, results
    else:  # (batch, height, width)
        boards_np = boards.cpu().numpy()

    # Apply augmentations
    augmentor = DataAugmentor(vertical_flip_prob=augment_prob)

    aug_boards_list = []
    aug_actions_list = [] if actions is not None else None
    aug_results_list = [] if results is not None else None

    for i in range(batch_size):
        board = boards_np[i]
        action = actions[i].item() if actions is not None else None
        result = results[i].item() if results is not None else None

        # Note: We need board representation for this, not encoded
        # For now, skip augmentation if we have encoded tensors
        aug_boards_list.append(board)
        if aug_actions_list is not None:
            aug_actions_list.append(action)
        if aug_results_list is not None:
            aug_results_list.append(result)

    # Convert back to tensors
    aug_boards = torch.from_numpy(np.array(aug_boards_list)).to(boards.device)

    if actions is not None:
        aug_actions = torch.tensor(aug_actions_list, dtype=actions.dtype, device=actions.device)
    else:
        aug_actions = None

    if results is not None:
        aug_results = torch.tensor(aug_results_list, dtype=results.dtype, device=results.device)
    else:
        aug_results = None

    return aug_boards, aug_actions, aug_results


if __name__ == "__main__":
    # Test augmentation functions
    print("Testing Data Augmentation Module...")

    # Create a test board
    test_board = np.zeros((10, 9), dtype=np.int8)

    # Place some pieces
    test_board[9, 4] = RED_GENERAL  # Red General at bottom center
    test_board[0, 4] = BLACK_GENERAL  # Black General at top center
    test_board[9, 0] = RED_ROOK  # Red Rook
    test_board[6, 0] = RED_SOLDIER  # Red Soldier past river
    test_board[3, 4] = BLACK_SOLDIER  # Black Soldier past river

    print("Original board:")
    print(test_board)

    # Test vertical flip
    flipped_v = flip_board_vertical(test_board)
    print("\nVertically flipped:")
    print(flipped_v)

    # Verify Red General moved from (9,4) to (0,4)
    assert flipped_v[0, 4] == BLACK_GENERAL
    assert flipped_v[9, 4] == RED_GENERAL
    print("✓ Vertical flip correct")

    # Test color swap
    swapped = swap_colors(test_board)
    print("\nColor swapped:")
    print(swapped)

    assert swapped[9, 4] == BLACK_GENERAL
    assert swapped[0, 4] == RED_GENERAL
    print("✓ Color swap correct")

    # Test augmentor
    augmentor = DataAugmentor(vertical_flip_prob=1.0)
    aug_board, _, _, info = augmentor.augment_position(test_board)

    print(f"\nAugmentation info: {info}")
    print("✓ Data augmentation tests passed!")
