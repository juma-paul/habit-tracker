"""Logging configurations using loguru."""

import sys
from functools import wraps
from time import perf_counter
from typing import Any, Callable

from loguru import logger

from app.core.config import settings

def setup_logging() -> None:
    """Configure loguru for the application."""

    # Remove default handler
    logger.remove()

    # Console format for development
    log_format = (
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG" if settings.environment == "development" else "INFO",
        colorize=True
    )

    # Add file handler for production
    if settings.environment == "production":
        logger.add(
            "logs/habits.log",
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="INFO",
        )

# Cost estimates per 1M tokens (GPT-4o)
COST_PER_1M_INPUT = 2.50
COST_PER_1M_OUTPUT = 10.00

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost in USD."""
    input_cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT
    output_cost = (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT

    return round(input_cost + output_cost, 6)

def log_tool_call(func: Callable) -> Callable:
    """Decorator to log tool calls with timing."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        tool_name = func.__name__
        start = perf_counter()

        logger.info(f"Tool call: {tool_name}", tool=tool_name, args=kwargs)

        try:
            result = await func(*args, **kwargs)
            elapsed_ms = (perf_counter() - start) * 1000

            # Check for errors in result
            if isinstance(result, dict) and "error" in result:
                logger.warning(
                    f"Tool {tool_name} returned error: {result["error"]}",
                    tool=tool_name,
                    elapsed_ms=round(elapsed_ms, 2)
                )
            else:
                logger.info(
                    f"Tool {tool_name} completed in {elapsed_ms:.2f}ms",
                    tool=tool_name,
                    elapsed_ms=round(elapsed_ms, 2),
                )

            return result
        
        except Exception as e:
            elapsed_ms = (perf_counter() - start) * 1000
            logger.error(
                f"Tool {tool_name} failed: {e}",
                tool=tool_name,
                elapsed_ms=round(elapsed_ms, 2),
                error=str(e),
            )


def log_agent_run(
        user_id: int,
        message: str,
        response: str,
        elapsed_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0
        ) -> None:
    """Log agent run with metrics."""
    cost = calculate_cost(input_tokens, output_tokens)

    logger.info(
        f"Agent run completed",
        user_id=user_id,
        message_preview=message[:50] + "..." if len(message) > 50 else message,
        response_preview=response[:50] + "..." if len(response) > 50 else response,
        elapsed_ms=round(elapsed_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
        cost_usd=cost,
    )

    # Warn if latency exceeds target
    if elapsed_ms > 200:
        logger.warning(
            f"High latency: {elapsed_ms:.2f}ms (target: <=200ms)",
            elapsed_ms=round(elapsed_ms, 2)
        )