#!/usr/bin/env python3
"""
Preprocess CSV chess data to HDF5 format for efficient training.

Converts the CSV game data to HDF5 format with encoded boards and actions.
"""

import sys
import os
import csv
import h5py
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoding.board_encoder import encode_board
from src.encoding.action_encoder import ActionEncoder
from src.utils.logger import get_logger

logger = get_logger("preprocess_data")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess CSV chess data to HDF5 format"
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/chess.h5',
        help='Path to output HDF5 file'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Chunk size for processing'
    )

    return parser.parse_args()


def count_csv_rows(csv_path):
    """Count rows in CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # Subtract header


def preprocess_csv(csv_path, hdf5_path, val_ratio=0.1, max_samples=None, chunk_size=10000):
    """
    Preprocess CSV data to HDF5 format.

    Args:
        csv_path: Path to input CSV file
        hdf5_path: Path to output HDF5 file
        val_ratio: Ratio of validation set
        max_samples: Maximum number of samples to process
        chunk_size: Chunk size for processing
    """
    logger.info(f"Preprocessing {csv_path} to {hdf5_path}")

    # Count total rows
    total_rows = count_csv_rows(csv_path)
    logger.info(f"Total rows in CSV: {total_rows:,}")

    if max_samples:
        total_rows = min(total_rows, max_samples)
        logger.info(f"Processing max {total_rows:,} samples")

    # Initialize action encoder
    action_encoder = ActionEncoder()
    logger.info(f"Action space size: {action_encoder.get_action_space_size()}")

    # Create output directory
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

    # Open CSV and HDF5 files
    with open(csv_path, 'r', encoding='utf-8') as csvfile, \
         h5py.File(hdf5_path, 'w') as h5f:

        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        logger.info(f"CSV header: {header}")

        # Prepare datasets
        # We'll store pre-encoded boards and actions
        board_shape = (14, 10, 9)

        # Calculate train/val split
        val_size = int(total_rows * val_ratio)
        train_size = total_rows - val_size

        logger.info(f"Train size: {train_size:,}, Val size: {val_size:,}")

        # Create datasets with compression
        train_group = h5f.create_group('train')
        val_group = h5f.create_group('val')

        # Create resizable datasets
        for group, size in [(train_group, train_size), (val_group, val_size)]:
            group.create_dataset(
                'boards',
                shape=(0, *board_shape),
                maxshape=(size, *board_shape),
                dtype=np.float32,
                compression='gzip',
                chunks=True
            )
            group.create_dataset(
                'actions',
                shape=(0,),
                maxshape=(size,),
                dtype=np.int64,
                compression='gzip'
            )
            group.create_dataset(
                'results',
                shape=(0,),
                maxshape=(size,),
                dtype=np.int8,
                compression='gzip'
            )

        # Process CSV in chunks
        train_count = 0
        val_count = 0

        # Buffers
        train_boards = []
        train_actions = []
        train_results = []
        val_boards = []
        val_actions = []
        val_results = []

        row_count = 0

        for row in tqdm(reader, total=total_rows, desc="Processing"):
            if len(row) < 5:
                continue

            board_str = row[0]
            move_str = row[1]
            result_str = row[4]

            # Filter incomplete games
            if result_str == 'None' or result_str == '':
                continue

            # Parse result
            try:
                result = int(result_str)
            except ValueError:
                continue

            # Encode board
            try:
                board_tensor = encode_board(board_str)
            except Exception as e:
                logger.warning(f"Failed to encode board: {e}")
                continue

            # Encode action
            try:
                action_idx = action_encoder.encode(move_str)
            except KeyError:
                # Skip invalid moves
                continue

            # Determine if train or val (based on count)
            is_train = (row_count % (1 + int(1/val_ratio))) != 0

            if is_train and train_count < train_size:
                train_boards.append(board_tensor.numpy())
                train_actions.append(action_idx)
                train_results.append(result)
                train_count += 1
            elif val_count < val_size:
                val_boards.append(board_tensor.numpy())
                val_actions.append(action_idx)
                val_results.append(result)
                val_count += 1

            row_count += 1

            # Flush buffers periodically
            if len(train_boards) >= chunk_size or len(val_boards) >= chunk_size:
                # Write train data
                if train_boards:
                    current_size = train_group['boards'].shape[0]
                    new_size = current_size + len(train_boards)

                    train_group['boards'].resize(new_size, axis=0)
                    train_group['actions'].resize(new_size, axis=0)
                    train_group['results'].resize(new_size, axis=0)

                    train_group['boards'][current_size:new_size] = np.array(train_boards)
                    train_group['actions'][current_size:new_size] = np.array(train_actions)
                    train_group['results'][current_size:new_size] = np.array(train_results)

                    train_boards.clear()
                    train_actions.clear()
                    train_results.clear()

                # Write val data
                if val_boards:
                    current_size = val_group['boards'].shape[0]
                    new_size = current_size + len(val_boards)

                    val_group['boards'].resize(new_size, axis=0)
                    val_group['actions'].resize(new_size, axis=0)
                    val_group['results'].resize(new_size, axis=0)

                    val_group['boards'][current_size:new_size] = np.array(val_boards)
                    val_group['actions'][current_size:new_size] = np.array(val_actions)
                    val_group['results'][current_size:new_size] = np.array(val_results)

                    val_boards.clear()
                    val_actions.clear()
                    val_results.clear()

        # Flush remaining buffers
        if train_boards:
            current_size = train_group['boards'].shape[0]
            new_size = current_size + len(train_boards)

            train_group['boards'].resize(new_size, axis=0)
            train_group['actions'].resize(new_size, axis=0)
            train_group['results'].resize(new_size, axis=0)

            train_group['boards'][current_size:new_size] = np.array(train_boards)
            train_group['actions'][current_size:new_size] = np.array(train_actions)
            train_group['results'][current_size:new_size] = np.array(train_results)

        if val_boards:
            current_size = val_group['boards'].shape[0]
            new_size = current_size + len(val_boards)

            val_group['boards'].resize(new_size, axis=0)
            val_group['actions'].resize(new_size, axis=0)
            val_group['results'].resize(new_size, axis=0)

            val_group['boards'][current_size:new_size] = np.array(val_boards)
            val_group['actions'][current_size:new_size] = np.array(val_actions)
            val_group['results'][current_size:new_size] = np.array(val_results)

        logger.info(f"Processed {train_count:,} train samples, {val_count:,} val samples")

        # Save action encoder
        encoder_path = hdf5_path.replace('.h5', '_action_encoder.pkl')
        action_encoder.save(encoder_path)
        logger.info(f"Saved action encoder to {encoder_path}")

    logger.info(f"Preprocessing complete! Output: {hdf5_path}")

    # Print file size
    file_size = os.path.getsize(hdf5_path) / (1024 ** 2)  # MB
    logger.info(f"HDF5 file size: {file_size:.2f} MB")


def main():
    """Main entry point."""
    args = parse_args()

    preprocess_csv(
        args.input,
        args.output,
        args.val_ratio,
        args.max_samples,
        args.chunk_size
    )


if __name__ == '__main__':
    main()
