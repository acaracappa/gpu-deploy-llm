"""Structured JSON logging for debugging cloud-gpu-shopper issues.

Provides structured logging with:
- Timestamps in ISO format
- Log levels
- Context fields (session_id, provider, etc.)
- JSON output for machine parsing
"""

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: str
    level: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class StructuredLogger:
    """Structured logger for diagnostic output.

    Usage:
        logger = StructuredLogger(session_id="sess-123")
        logger.info("Starting deployment", model="tinyllama")
        logger.error("Deployment failed", error="Container crash")

        # Get all entries for diagnostics
        entries = logger.get_entries()
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        provider: Optional[str] = None,
        output_json: bool = False,
    ):
        """Initialize structured logger.

        Args:
            session_id: Session ID for context
            provider: Provider name for context
            output_json: If True, output JSON to stderr
        """
        self.session_id = session_id
        self.provider = provider
        self.output_json = output_json
        self._entries: List[LogEntry] = []
        self._python_logger = logging.getLogger("gpu_deploy_llm")

    def _log(
        self,
        level: LogLevel,
        message: str,
        **kwargs,
    ) -> LogEntry:
        """Create and store a log entry."""
        context = {}

        # Add standard context
        if self.session_id:
            context["session_id"] = self.session_id
        if self.provider:
            context["provider"] = self.provider

        # Add additional context
        context.update(kwargs)

        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level.value,
            message=message,
            context=context,
        )

        self._entries.append(entry)

        # Output to Python logger
        log_func = getattr(self._python_logger, level.value)
        if context:
            log_func(f"{message} | {context}")
        else:
            log_func(message)

        # Output JSON if enabled
        if self.output_json:
            print(entry.to_json(), file=sys.stderr)

        return entry

    def debug(self, message: str, **kwargs) -> LogEntry:
        """Log debug message."""
        return self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> LogEntry:
        """Log info message."""
        return self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> LogEntry:
        """Log warning message."""
        return self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> LogEntry:
        """Log error message."""
        return self._log(LogLevel.ERROR, message, **kwargs)

    def get_entries(self) -> List[LogEntry]:
        """Get all log entries."""
        return self._entries.copy()

    def get_entries_json(self) -> str:
        """Get all entries as JSON array."""
        return json.dumps([e.to_dict() for e in self._entries], indent=2, default=str)

    def clear(self) -> None:
        """Clear all stored entries."""
        self._entries.clear()


def setup_logging(
    level: str = "INFO",
    debug_shopper: bool = False,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        debug_shopper: Enable verbose shopper API logging
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Set httpx logging level
    if debug_shopper:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Set asyncssh logging
    logging.getLogger("asyncssh").setLevel(logging.WARNING)
