#!/usr/bin/env python3
"""
Check status of background tasks.

Displays task status, progress, and results for Gumbel AlphaZero training.
"""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.background_trainer import check_training_status, wait_for_training
from src.utils.background import get_background_manager


def main():
    parser = argparse.ArgumentParser(
        description="Check background task status for Gumbel AlphaZero"
    )

    parser.add_argument('--task-id', required=True, help='Task ID to check')
    parser.add_argument('--wait', action='store_true', help='Wait for completion')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    manager = get_background_manager()
    task = manager.get_task_status(args.task_id)

    if task:
        if args.json:
            output = {
                'task_id': task.task_id,
                'status': task.status.value,
                'progress': float(task.progress),
                'elapsed_time': task.elapsed_time(),
            }
            if task.result is not None:
                output['result'] = str(type(task.result))
            if task.error:
                output['error'] = task.error

            print(json.dumps(output, indent=2))
        else:
            print(f"Task: {task.task_id}")
            print(f"Status: {task.status.value}")
            print(f"Progress: {task.progress:.1%}")
            print(f"Elapsed: {task.elapsed_time():.1f}s")

            if task.start_time:
                from datetime import datetime
                start = datetime.fromtimestamp(task.start_time)
                print(f"Started: {start}")

            if task.end_time:
                end = datetime.fromtimestamp(task.end_time)
                print(f"Completed: {end}")

            if task.error:
                print(f"Error: {task.error}")

        if args.wait and task.status.value == 'running':
            print("\nWaiting for completion...")
            task = manager.wait_for_task(args.task_id)
            print(f"Final status: {task.status.value}")

            if task.status.value == 'completed':
                print(f"Progress: {task.progress:.1%}")
                print(f"Elapsed: {task.elapsed_time():.1f}s")

        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
