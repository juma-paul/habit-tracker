"""Logging configurations using loguru."""

import sys

from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    """Configure loguru for the application."""
    logger.remove()

    log_format = (
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG" if settings.environment == "development" else "INFO",
        colorize=True,
    )

    if settings.environment == "production":
        logger.add(
            "logs/habits.log",
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="INFO",
        )


def log_agent_run(
    user_id: int,
    message: str,
    response: str,
    elapsed_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_calls: int = 0,
) -> None:
    """Log agent run with metrics."""
    cost = round(
        (input_tokens / 1_000_000) * 2.50 + (output_tokens / 1_000_000) * 10.00, 6
    )

    logger.info(
        "Agent run completed",
        user_id=user_id,
        message_preview=message[:50] + "..." if len(message) > 50 else message,
        response_preview=response[:50] + "..." if len(response) > 50 else response,
        elapsed_ms=round(elapsed_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
        cost_usd=cost,
    )

    # LLM calls typically take 1–5s; warn above 10s (likely a timeout or hung request).
    if elapsed_ms > 10_000:
        logger.warning(
            f"High latency: {elapsed_ms:.2f}ms (target: <=10000ms)",
            elapsed_ms=round(elapsed_ms, 2),
        )
