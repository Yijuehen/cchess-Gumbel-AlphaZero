"""
Dataset module for Gumbel AlphaZero

Provides efficient data loading for training from HDF5 files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from typing import Optional, Tuple, List
import sys
from pathlib import Path

# Add parent directory to import from encoding module
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoding.board_encoder import encode_board
from encoding.action_encoder import ActionEncoder
from utils.logger import get_logger

logger = get_logger("dataset")


class ChessHDF5Dataset(Dataset):
    """
    PyTorch Dataset for Chinese Chess positions stored in HDF5 format.

    Expects HDF5 file with datasets:
        - 'boards': Board strings or pre-encoded tensors
        - 'actions': Action indices
        - 'results': Game results (0=Red wins, 1=Black wins, 2=Draw)
    """

    def __init__(
        self,
        hdf5_path: str,
        group: str = 'train',
        action_encoder: ActionEncoder = None,
        encode_boards: bool = True,
        transform=None
    ):
        """
        Initialize dataset.

        Args:
            hdf5_path: Path to HDF5 file
            group: Group name in HDF5 file ('train', 'val', 'test')
            action_encoder: ActionEncoder for move encoding
            encode_boards: Whether to encode boards from strings
            transform: Optional transform function
        """
        self.hdf5_path = hdf5_path
        self.group = group
        self.action_encoder = action_encoder or ActionEncoder()
        self.encode_boards = encode_boards
        self.transform = transform

        # Open HDF5 file
        self.h5_file = h5py.File(hdf5_path, 'r')
        self.dataset = self.h5_file[group]

        # Get length
        self.length = len(self.dataset['boards'])

        logger.info(f"Loaded dataset: {hdf5_path}/{group}")
        logger.info(f"  Samples: {self.length}")

    def __len__(self) -> int:
        """Get dataset length."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        """
        Get a single sample.

        Returns:
            Tuple of (board_tensor, action_idx, value)
        """
        # Load board
        board_data = self.dataset['boards'][idx]

        if self.encode_boards:
            # Assume board is stored as string, encode to tensor
            if isinstance(board_data, bytes):
                board_str = board_data.decode('utf-8')
            else:
                board_str = str(board_data)

            board_tensor = encode_board(board_str)
        else:
            # Assume board is pre-encoded tensor
            board_tensor = torch.from_numpy(board_data).float()

        # Load action
        action = int(self.dataset['actions'][idx])

        # Load result and convert to value
        result = int(self.dataset['results'][idx])
        value = self.result_to_value(result)

        # Apply transform if provided
        if self.transform:
            board_tensor = self.transform(board_tensor)

        return board_tensor, action, value

    def result_to_value(self, result: int, perspective: int = 1) -> float:
        """
        Convert game result to value.

        Args:
            result: 0=Red wins, 1=Black wins, 2=Draw
            perspective: 1=Red, -1=Black

        Returns:
            Value in [-1, 1]
        """
        if result == 2:  # Draw
            return 0.0
        elif result == 0:  # Red wins
            return 1.0 if perspective == 1 else -1.0
        else:  # Black wins
            return -1.0 if perspective == 1 else 1.0

    def close(self):
        """Close HDF5 file."""
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        """Ensure file is closed on deletion."""
        self.close()


def create_data_loader(
    hdf5_path: str,
    group: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    action_encoder: ActionEncoder = None
) -> DataLoader:
    """
    Create a PyTorch DataLoader for training.

    Args:
        hdf5_path: Path to HDF5 file
        group: Group name in HDF5 file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        action_encoder: ActionEncoder instance

    Returns:
        DataLoader instance
    """
    dataset = ChessHDF5Dataset(
        hdf5_path=hdf5_path,
        group=group,
        action_encoder=action_encoder,
        encode_boards=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False
    )

    logger.info(f"Created DataLoader: {hdf5_path}/{group}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Shuffle: {shuffle}")

    return loader


class StreamingCSVDataset(Dataset):
    """
    Streaming dataset that reads directly from CSV without HDF5 conversion.

    Useful for quick prototyping with smaller datasets.
    """

    def __init__(
        self,
        csv_path: str,
        action_encoder: ActionEncoder = None,
        max_samples: int = None
    ):
        """
        Initialize streaming CSV dataset.

        Args:
            csv_path: Path to CSV file
            action_encoder: ActionEncoder instance
            max_samples: Maximum number of samples to load
        """
        import csv

        self.csv_path = csv_path
        self.action_encoder = action_encoder or ActionEncoder()

        # Read CSV and cache in memory
        self.data = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) < 5:
                    continue

                board_str = row[0]
                move_str = row[1]
                result_str = row[4]

                # Filter samples without results
                if result_str == 'None':
                    continue

                # Parse action
                try:
                    action_idx = self.action_encoder.encode(move_str)
                except KeyError:
                    continue  # Skip invalid moves

                # Parse result
                result = int(result_str) if result_str != 'None' else None
                if result is None:
                    continue

                self.data.append((board_str, action_idx, result))

                if max_samples and len(self.data) >= max_samples:
                    break

        logger.info(f"Loaded {len(self.data)} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        board_str, action_idx, result = self.data[idx]

        # Encode board
        board_tensor = encode_board(board_str)

        # Convert result to value
        if result == 2:
            value = 0.0
        elif result == 0:
            value = 1.0
        else:
            value = -1.0

        return board_tensor, action_idx, value


if __name__ == "__main__":
    # Test dataset functionality
    print("Testing Dataset module...")

    import tempfile
    import os

    # Create dummy HDF5 file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.h5', delete=False) as f:
        temp_path = f.name

    try:
        # Create test HDF5 file
        with h5py.File(temp_path, 'w') as f:
            train_group = f.create_group('train')

            # Create dummy data
            num_samples = 100
            board_shape = (14, 10, 9)

            train_group.create_dataset(
                'boards',
                data=np.random.randn(num_samples, *board_shape).astype(np.float32),
                compression='gzip'
            )
            train_group.create_dataset(
                'actions',
                data=np.random.randint(0, 2086, size=num_samples),
                compression='gzip'
            )
            train_group.create_dataset(
                'results',
                data=np.random.randint(0, 3, size=num_samples),
                compression='gzip'
            )

        print(f"Created test HDF5 file: {temp_path}")

        # Test dataset
        dataset = ChessHDF5Dataset(
            temp_path,
            group='train',
            encode_boards=False
        )

        print(f"Dataset length: {len(dataset)}")

        # Test __getitem__
        board, action, value = dataset[0]
        print(f"Sample 0:")
        print(f"  Board shape: {board.shape}")
        print(f"  Action: {action}")
        print(f"  Value: {value}")

        # Test DataLoader
        loader = create_data_loader(
            temp_path,
            group='train',
            batch_size=8,
            shuffle=False
        )

        for batch_idx, (boards, actions, values) in enumerate(loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Boards: {boards.shape}")
            print(f"  Actions: {actions.shape}")
            print(f"  Values: {values.shape}")

            if batch_idx >= 2:
                break

        print("\n✓ Dataset tests passed!")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"\nCleaned up test file: {temp_path}")
