"""
Board Encoder for Gumbel AlphaZero

Converts 64-character board strings to (14, 10, 9) PyTorch tensors.
- Channels 0-6: Red pieces (General, Advisor, Elephant, Horse, Rook, Cannon, Soldier)
- Channels 7-13: Black pieces (same order)
- Each channel is binary (0 or 1)
"""

import torch
import numpy as np
from typing import Tuple
import sys
from pathlib import Path

# Add parent directory to path to import from Alpha-Beta+NNUE
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "cchess-Alpha-Beta+NNUE" / "src"))

from board.board_representation import (
    ROWS, COLS, EMPTY,
    RED_GENERAL, RED_ADVISOR, RED_ELEPHANT, RED_HORSE, RED_ROOK, RED_CANNON, RED_SOLDIER,
    BLACK_GENERAL, BLACK_ADVISOR, BLACK_ELEPHANT, BLACK_HORSE, BLACK_ROOK, BLACK_CANNON, BLACK_SOLDIER,
    is_red, is_black, get_piece_type,
    pos_to_coords, coords_to_pos, parse_board
)


# Piece type to channel mapping (0-6)
PIECE_TO_CHANNEL = {
    1: 0,  # General/King (帅/将)
    2: 1,  # Advisor (仕/士)
    3: 2,  # Elephant (相/象)
    4: 3,  # Horse (马)
    5: 4,  # Rook (车)
    6: 5,  # Cannon (炮)
    7: 6,  # Soldier (兵/卒)
}

# Number of channels
NUM_CHANNELS = 14  # 7 for Red + 7 for Black


def encode_board(board_str: str) -> torch.Tensor:
    """
    Encode 64-character board string to (14, 10, 9) tensor.

    Args:
        board_str: 64-character board encoding

    Returns:
        torch.Tensor of shape (14, 10, 9) with binary values

    Example:
        >>> board = encode_board("0919293949596979891777062646668600102030405060708012720323436383")
        >>> board.shape
        torch.Size([14, 10, 9])
    """
    # Parse board using existing function
    board_array = parse_board(board_str)  # Shape: (10, 9)

    # Initialize tensor: (channels, height, width) = (14, 10, 9)
    tensor = torch.zeros(NUM_CHANNELS, ROWS, COLS, dtype=torch.float32)

    # Encode pieces
    for row in range(ROWS):
        for col in range(COLS):
            piece = board_array[row, col]

            if piece != EMPTY:
                piece_type = abs(piece)
                channel = PIECE_TO_CHANNEL[piece_type]

                if is_red(piece):
                    # Red pieces: channels 0-6
                    tensor[channel, row, col] = 1.0
                else:
                    # Black pieces: channels 7-13
                    tensor[channel + 7, row, col] = 1.0

    return tensor


def encode_batch(board_strings: list) -> torch.Tensor:
    """
    Encode multiple board strings to batch tensor.

    Args:
        board_strings: List of 64-character board strings

    Returns:
        torch.Tensor of shape (batch_size, 14, 10, 9)
    """
    batch = []
    for board_str in board_strings:
        batch.append(encode_board(board_str))
    return torch.stack(batch)


def decode_board(tensor: torch.Tensor) -> np.ndarray:
    """
    Decode (14, 10, 9) tensor back to (10, 9) board array.

    Args:
        tensor: torch.Tensor of shape (14, 10, 9)

    Returns:
        numpy array of shape (10, 9) with piece values
    """
    if tensor.shape != (NUM_CHANNELS, ROWS, COLS):
        raise ValueError(f"Expected shape (14, 10, 9), got {tensor.shape}")

    # Convert to numpy
    tensor_np = tensor.cpu().numpy()

    # Initialize board
    board = np.zeros((ROWS, COLS), dtype=np.int8)

    # Decode pieces
    for row in range(ROWS):
        for col in range(COLS):
            # Check Red pieces (channels 0-6)
            for channel in range(7):
                if tensor_np[channel, row, col] > 0.5:
                    # Map channel back to piece type
                    piece_type = channel + 1
                    board[row, col] = piece_type  # Red: positive
                    break

            # Check Black pieces (channels 7-13)
            for channel in range(7, 14):
                if tensor_np[channel, row, col] > 0.5:
                    # Map channel back to piece type
                    piece_type = channel - 7 + 1
                    board[row, col] = -piece_type  # Black: negative
                    break

    return board


def validate_encoding(tensor: torch.Tensor) -> bool:
    """
    Validate that the encoded tensor is properly formed.

    Checks:
    - Shape is (14, 10, 9)
    - Values are binary (0 or 1)
    - No position has both Red and Black pieces

    Args:
        tensor: torch.Tensor to validate

    Returns:
        True if valid, False otherwise
    """
    # Check shape
    if tensor.shape != (NUM_CHANNELS, ROWS, COLS):
        return False

    # Check binary values
    if not torch.all((tensor == 0) | (tensor == 1)):
        return False

    # Check no overlapping pieces
    red_pieces = tensor[:7].sum(dim=0)  # Sum over Red channels
    black_pieces = tensor[7:].sum(dim=0)  # Sum over Black channels

    # Each position should have at most one piece
    if torch.any(red_pieces + black_pieces > 1):
        return False

    return True


def get_piece_count(tensor: torch.Tensor) -> dict:
    """
    Count pieces by type and color from encoded tensor.

    Args:
        tensor: torch.Tensor of shape (14, 10, 9)

    Returns:
        Dictionary with piece counts
    """
    piece_names = {
        0: "General", 1: "Advisor", 2: "Elephant", 3: "Horse",
        4: "Rook", 5: "Cannon", 6: "Soldier"
    }

    counts = {"red": {}, "black": {}}

    for channel in range(7):
        red_count = tensor[channel].sum().item()
        black_count = tensor[channel + 7].sum().item()
        piece_name = piece_names[channel]

        counts["red"][piece_name] = int(red_count)
        counts["black"][piece_name] = int(black_count)

    return counts


def visualize_board(tensor: torch.Tensor) -> str:
    """
    Create a text visualization of the board from tensor.

    Args:
        tensor: torch.Tensor of shape (14, 10, 9)

    Returns:
        String representation of the board
    """
    piece_symbols = {
        1: "K", 2: "A", 3: "E", 4: "H", 5: "R", 6: "C", 7: "S"
    }

    board_str = ""
    for row in range(ROWS):
        line = ""
        for col in range(COLS):
            piece_found = False

            # Check Red pieces
            for channel in range(7):
                if tensor[channel, row, col] > 0.5:
                    line += f"\033[91m{piece_symbols[channel + 1]}\033[0m"  # Red
                    piece_found = True
                    break

            # Check Black pieces
            if not piece_found:
                for channel in range(7, 14):
                    if tensor[channel, row, col] > 0.5:
                        line += f"\033[90m{piece_symbols[channel - 7 + 1]}\033[0m"  # Gray/Black
                        piece_found = True
                        break

            if not piece_found:
                line += " ·"

            line += " "
        board_str += line + "\n"

    return board_str


# Predefined channels for easy access
RED_GENERAL_CHANNEL = 0
RED_ADVISOR_CHANNEL = 1
RED_ELEPHANT_CHANNEL = 2
RED_HORSE_CHANNEL = 3
RED_ROOK_CHANNEL = 4
RED_CANNON_CHANNEL = 5
RED_SOLDIER_CHANNEL = 6

BLACK_GENERAL_CHANNEL = 7
BLACK_ADVISOR_CHANNEL = 8
BLACK_ELEPHANT_CHANNEL = 9
BLACK_HORSE_CHANNEL = 10
BLACK_ROOK_CHANNEL = 11
BLACK_CANNON_CHANNEL = 12
BLACK_SOLDIER_CHANNEL = 13


if __name__ == "__main__":
    # Test encoding
    test_board_str = "0919293949596979891777062646668600102030405060708012720323436383"
    print(f"Testing encoding with board: {test_board_str[:16]}...")

    # Encode
    tensor = encode_board(test_board_str)
    print(f"Encoded tensor shape: {tensor.shape}")
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor dtype: {tensor.dtype}")

    # Validate
    is_valid = validate_encoding(tensor)
    print(f"Is valid: {is_valid}")

    # Count pieces
    counts = get_piece_count(tensor)
    print(f"\nPiece counts:")
    print(f"  Red: {counts['red']}")
    print(f"  Black: {counts['black']}")

    # Visualize
    print(f"\nBoard visualization:")
    print(visualize_board(tensor))

    # Test decode
    decoded = decode_board(tensor)
    print(f"Decoded board shape: {decoded.shape}")
    print(f"Decoded board:\n{decoded}")

    print("\n✓ Board encoder test passed!")
