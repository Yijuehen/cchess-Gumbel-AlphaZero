"""
Logging module for Chinese Chess engine.

Provides structured logging with file rotation and multiple handlers.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_obj = {
            'timestamp': self.formatTime(record, '%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


class ChessEngineLogger:
    """
    Centralized logger configuration for the chess engine.

    Features:
    - Multiple handlers (console, file with rotation)
    - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - JSON format option for structured logging
    - Module-specific loggers
    """

    def __init__(
        self,
        name: str = "chess_engine",
        log_dir: str = "logs",
        level: str = "INFO",
        console: bool = True,
        file: bool = True,
        json_format: bool = False
    ):
        """
        Initialize logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console: Enable console output
            file: Enable file output
            json_format: Use JSON format for logs
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        self.logger.propagate = False

        # Create formatter
        if json_format:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        # Console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, level))
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_path / f"{name}.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, level))
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger

    def set_level(self, level: str) -> None:
        """Change logging level."""
        self.logger.setLevel(getattr(logging, level))
        for handler in self.logger.handlers:
            handler.setLevel(getattr(logging, level))


# Global logger instance
_global_logger: Optional[ChessEngineLogger] = None


def get_logger(name: str = "chess_engine") -> logging.Logger:
    """
    Get or create a logger.

    Args:
        name: Logger name (module-specific names are recommended)

    Returns:
        Logger instance
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = ChessEngineLogger()

    return logging.getLogger(name)


def setup_logging(
    name: str = "chess_engine",
    log_dir: str = "logs",
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
    json_format: bool = False
) -> ChessEngineLogger:
    """
    Setup logging for the application.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Enable console output
        file: Enable file output
        json_format: Use JSON format

    Returns:
        Configured ChessEngineLogger instance
    """
    global _global_logger
    _global_logger = ChessEngineLogger(
        name=name,
        log_dir=log_dir,
        level=level,
        console=console,
        file=file,
        json_format=json_format
    )
    return _global_logger
