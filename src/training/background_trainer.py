"""
Background Training Support for Gumbel AlphaZero

Allows training to run in background threads with progress tracking.
"""

import sys
from pathlib import Path
from typing import Optional, Callable

# Add parent directory to import from other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.background import get_background_manager
from utils.logger import get_logger

logger = get_logger("background_training")


def train_model_background(
    network,
    train_loader,
    val_loader,
    task_id: str,
    epochs: int = 10,
    progress_callback: Optional[Callable[[float], None]] = None,
    **training_kwargs
) -> str:
    """
    Train model in background.

    Args:
        network: AlphaZero network to train
        train_loader: Training data loader
        val_loader: Validation data loader
        task_id: Unique task identifier
        epochs: Number of training epochs
        progress_callback: Callback for progress updates (0.0 to 1.0)
        **training_kwargs: Additional training arguments

    Returns:
        Task ID
    """
    def training_wrapper():
        logger.info(f"Background training task started: {task_id}")
        logger.info(f"Epochs: {epochs}")

        import torch
        import torch.optim as optim
        from .loss import AlphaZeroLoss

        device = training_kwargs.get('device', 'cpu')
        network.to(device)

        # Setup optimizer
        lr = training_kwargs.get('learning_rate', 0.001)
        optimizer = optim.Adam(network.parameters(), lr=lr)

        # Setup loss
        loss_fn = AlphaZeroLoss(
            policy_weight=training_kwargs.get('policy_weight', 1.0),
            value_weight=training_kwargs.get('value_weight', 1.0)
        )

        # Training loop
        network.train()
        total_batches = epochs * len(train_loader)
        current_batch = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (boards, actions, values) in enumerate(train_loader):
                boards = boards.to(device)
                actions = actions.to(device)
                values = values.to(device)

                # Forward pass
                policy_logits, value = network(boards)

                # Compute loss
                loss, p_loss, v_loss = loss_fn(
                    policy_logits, value, actions, values
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Update progress
                current_batch += 1
                if progress_callback:
                    progress = current_batch / total_batches
                    progress_callback(progress)

            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        logger.info(f"Background training task completed: {task_id}")
        return network

    # Submit to background manager
    manager = get_background_manager()
    manager.submit_task(
        task_id=task_id,
        func=training_wrapper,
        progress_callback=progress_callback
    )

    logger.info(f"Background training submitted: {task_id}")
    return task_id


def check_training_status(task_id: str):
    """
    Check status of background training.

    Args:
        task_id: Task identifier

    Returns:
        TaskResult or None if not found
    """
    manager = get_background_manager()
    task = manager.get_task_status(task_id)

    if task:
        logger.info(
            f"Task {task_id}: status={task.status.value}, "
            f"progress={task.progress:.1%}, elapsed={task.elapsed_time():.1f}s"
        )
        return task
    else:
        logger.warning(f"Task not found: {task_id}")
        return None


def wait_for_training(task_id: str, timeout: Optional[float] = None):
    """
    Wait for background training to complete.

    Args:
        task_id: Task identifier
        timeout: Optional timeout in seconds

    Returns:
        Trained network or None
    """
    manager = get_background_manager()
    task = manager.wait_for_task(task_id, timeout=timeout)

    if task.status.value == 'completed':
        logger.info(f"Training completed successfully: {task_id}")
        return task.result
    elif task.status.value == 'failed':
        logger.error(f"Training failed: {task.error}")
        raise Exception(task.error)
    elif task.status.value == 'cancelled':
        logger.warning(f"Training was cancelled: {task_id}")
        return None
    else:
        logger.warning(f"Training timeout or still running: {task_id}")
        return None


def list_training_tasks():
    """List all training tasks."""
    manager = get_background_manager()
    tasks = manager.list_tasks()
    return tasks
