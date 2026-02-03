#!/usr/bin/env python3
"""
Supervised Training Script for Gumbel AlphaZero

Trains the neural network on expert game data using supervised learning.
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network.alphanet import AlphaZeroNet, save_checkpoint, load_checkpoint
from src.training.dataset import create_data_loader, StreamingCSVDataset
from src.training.loss import AlphaZeroLoss, compute_policy_entropy
from src.utils.logger import get_logger

logger = get_logger("train_supervised")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AlphaZero network with supervised learning"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/training.yaml',
        help='Path to training configuration file'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/chess.csv',
        help='Path to training data (CSV or HDF5)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='models/final/',
        help='Output directory for models'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--background',
        action='store_true',
        help='Run training in background'
    )

    parser.add_argument(
        '--task-id',
        type=str,
        default=None,
        help='Task ID for background training'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )

    return parser.parse_args()


def load_config(config_path):
    """Load training configuration from YAML file."""
    default_config = {
        'training': {
            'epochs': 50,
            'batch_size': 128,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'lr_schedule': 'cosine',
            'policy_weight': 1.0,
            'value_weight': 1.0,
            'grad_clip': 1.0
        },
        'network': {
            'in_channels': 14,
            'hidden_channels': 256,
            'num_residual_blocks': 15,
            'action_space_size': 2086
        },
        'validation': {
            'ratio': 0.1,
            'frequency': 1
        },
        'checkpointing': {
            'frequency': 5,
            'keep_best': 3
        }
    }

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Merge with defaults
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
        logger.info(f"Loaded config from {config_path}")
    else:
        config = default_config
        logger.warning(f"Config file not found: {config_path}, using defaults")

    return config


def create_optimizer(model, config):
    """Create optimizer with learning rate scheduler."""
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler_type = config['training'].get('lr_schedule', 'none')

    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
    else:
        scheduler = None

    return optimizer, scheduler


def train_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    epoch,
    config,
    writer=None,
    progress_callback=None
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0

    num_batches = len(train_loader)

    for batch_idx, (boards, actions, values) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        boards = boards.to(device)
        actions = actions.to(device)
        values = values.to(device)

        # Forward pass
        policy_logits, value = model(boards)

        # Compute loss
        loss, policy_loss, value_loss = loss_fn(
            policy_logits, value, actions, values
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_clip = config['training'].get('grad_clip')
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

        # Update progress
        if progress_callback:
            overall_progress = (epoch - 1 + batch_idx / num_batches) / config['training']['epochs']
            progress_callback(overall_progress)

    avg_loss = total_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches

    # Log to tensorboard
    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/policy_loss', avg_policy_loss, epoch)
        writer.add_scalar('train/value_loss', avg_value_loss, epoch)

    return avg_loss, avg_policy_loss, avg_value_loss


def validate(model, val_loader, loss_fn, device, epoch, writer=None):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0

    with torch.no_grad():
        for boards, actions, values in val_loader:
            boards = boards.to(device)
            actions = actions.to(device)
            values = values.to(device)

            policy_logits, value = model(boards)
            loss, policy_loss, value_loss = loss_fn(
                policy_logits, value, actions, values
            )

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches

    # Log to tensorboard
    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/policy_loss', avg_policy_loss, epoch)
        writer.add_scalar('val/value_loss', avg_value_loss, epoch)

    return avg_loss, avg_policy_loss, avg_value_loss


def train_supervised(config, args, progress_callback=None):
    """Main training function."""
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create model
    net_config = config['network']
    model = AlphaZeroNet(
        in_channels=net_config['in_channels'],
        hidden_channels=net_config['hidden_channels'],
        num_residual_blocks=net_config['num_residual_blocks'],
        action_space_size=net_config['action_space_size']
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        model = load_checkpoint(args.resume, model)
        logger.info(f"Resumed from checkpoint: {args.resume}")

    # Create optimizer
    optimizer, scheduler = create_optimizer(model, config)

    # Create loss function
    loss_fn = AlphaZeroLoss(
        policy_weight=config['training']['policy_weight'],
        value_weight=config['training']['value_weight']
    )

    # Create data loaders
    logger.info(f"Loading data from: {args.data}")

    # Check if data is HDF5 or CSV
    if args.data.endswith('.h5') or args.data.endswith('.hdf5'):
        train_loader = create_data_loader(
            args.data,
            group='train',
            batch_size=args.batch_size or config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )

        val_loader = create_data_loader(
            args.data,
            group='val',
            batch_size=args.batch_size or config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )
    else:
        # Use streaming CSV dataset
        from torch.utils.data import random_split

        full_dataset = StreamingCSVDataset(args.data, max_samples=100000)

        # Split into train/val
        val_size = int(len(full_dataset) * config['validation']['ratio'])
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size or config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size or config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Setup tensorboard
    writer = SummaryWriter('logs/tensorboard')

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Training loop
    epochs = args.epochs or config['training']['epochs']
    best_val_loss = float('inf')

    logger.info(f"Starting training for {epochs} epochs")

    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_loss, train_p_loss, train_v_loss = train_epoch(
            model, train_loader, optimizer, loss_fn,
            device, epoch, config, writer,
            progress_callback
        )

        # Validate
        if epoch % config['validation']['frequency'] == 0:
            val_loss, val_p_loss, val_v_loss = validate(
                model, val_loader, loss_fn, device, epoch, writer
            )

            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

            # Save checkpoint
            if epoch % config['checkpointing']['frequency'] == 0:
                checkpoint_path = os.path.join(
                    args.output,
                    f"checkpoint_epoch_{epoch}.pth"
                )
                save_checkpoint(model, checkpoint_path, optimizer, epoch)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.output, "best_model.pth")
                save_checkpoint(model, best_path, optimizer, epoch)
                logger.info(f"Saved best model: {best_path}")

        # Update learning rate
        if scheduler:
            scheduler.step()

    # Save final model
    final_path = os.path.join(args.output, "final_model.pth")
    save_checkpoint(model, final_path, optimizer, epochs)
    logger.info(f"Saved final model: {final_path}")

    writer.close()

    return model


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)

    if args.background:
        # Run in background
        from src.training.background_trainer import train_model_background

        # Note: This is a simplified version - you'd need to adapt the data loading
        logger.info("Starting background training...")
        logger.warning("Background training mode needs additional setup")
    else:
        # Run in foreground
        model = train_supervised(config, args)
        logger.info("Training completed!")


if __name__ == '__main__':
    main()
