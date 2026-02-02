"""
Background task manager for Chinese Chess engine.

Provides thread-based task execution with progress tracking and status management.
"""

import threading
import time
from typing import Callable, Optional, Any, Dict
from dataclasses import dataclass, field
from enum import Enum

from .logger import get_logger

logger = get_logger("background")


class TaskStatus(Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a background task execution."""
    task_id: str
    status: TaskStatus
    progress: float  # 0.0 to 1.0
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


class BackgroundTaskManager:
    """
    Manager for background task execution.

    Features:
    - Thread-based task execution
    - Progress tracking with callbacks
    - Task result storage
    - Error handling
    - Task status querying
    """

    def __init__(self):
        """Initialize task manager."""
        self.tasks: Dict[str, TaskResult] = {}
        self.task_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self.logger = logger

    def submit_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        Submit a task to run in background.

        Args:
            task_id: Unique task identifier
            func: Function to execute
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            Task ID
        """
        if kwargs is None:
            kwargs = {}

        with self._lock:
            # Check if task already exists
            if task_id in self.tasks:
                self.logger.warning(f"Task {task_id} already exists, replacing")

            # Create task result
            self.tasks[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING,
                progress=0.0,
                start_time=time.time()
            )

        # Create and start thread
        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, func, args, kwargs, progress_callback),
            name=f"Task-{task_id}",
            daemon=True
        )

        with self._lock:
            self.task_threads[task_id] = thread

        thread.start()

        self.logger.info(f"Task submitted: {task_id}")
        return task_id

    def _run_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple,
        kwargs: dict,
        progress_callback: Optional[Callable[[float], None]]
    ) -> None:
        """
        Internal method to run task in thread.

        Args:
            task_id: Task identifier
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            progress_callback: Optional progress callback
        """
        task = self.tasks.get(task_id)
        if not task:
            self.logger.error(f"Task not found: {task_id}")
            return

        task.status = TaskStatus.RUNNING
        self.logger.info(f"Task started: {task_id}")

        try:
            # Run function
            result = func(*args, **kwargs)

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.progress = 1.0

            self.logger.info(f"Task completed: {task_id}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.logger.error(f"Task failed: {task_id}, error={e}", exc_info=True)

        finally:
            task.end_time = time.time()

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """
        Get status of a task.

        Args:
            task_id: Task identifier

        Returns:
            TaskResult or None if task not found
        """
        with self._lock:
            return self.tasks.get(task_id)

    def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Optional[TaskResult]:
        """
        Wait for task completion.

        Args:
            task_id: Task identifier
            timeout: Optional timeout in seconds

        Returns:
            TaskResult or None if timeout
        """
        thread = self.task_threads.get(task_id)
        if thread:
            thread.join(timeout=timeout)

        return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Note: Thread cancellation is limited in Python. Tasks should
        check for cancellation themselves.

        Args:
            task_id: Task identifier

        Returns:
            True if task was marked as cancelled
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
                self.logger.info(f"Task cancelled: {task_id}")
                return True
        return False

    def list_tasks(self) -> Dict[str, TaskResult]:
        """List all tasks."""
        with self._lock:
            return self.tasks.copy()

    def cleanup_task(self, task_id: str) -> bool:
        """
        Clean up completed task from memory.

        Args:
            task_id: Task identifier

        Returns:
            True if task was removed
        """
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    # Remove task
                    del self.tasks[task_id]
                    if task_id in self.task_threads:
                        del self.task_threads[task_id]
                    self.logger.info(f"Task cleaned up: {task_id}")
                    return True
        return False


# Global background task manager
_background_manager: Optional[BackgroundTaskManager] = None


def get_background_manager() -> BackgroundTaskManager:
    """Get the global background task manager."""
    global _background_manager
    if _background_manager is None:
        _background_manager = BackgroundTaskManager()
    return _background_manager
