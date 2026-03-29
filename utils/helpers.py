"""
Quantix — Shared Utilities (v2)
"""
from __future__ import annotations
import functools
import json
import logging
import time
from datetime import datetime
from typing import Any, Coroutine

import asyncio
import threading


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def run_async(coro: Coroutine) -> Any:
    """Runs an async coroutine in a synchronous context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # In a running loop (like FastAPI), we need to run in a separate thread
        from concurrent.futures import ThreadPoolExecutor
        def _run():
            return asyncio.run(coro)
        with ThreadPoolExecutor() as executor:
            future = executor.submit(_run)
            return future.result()
    else:
        return loop.run_until_complete(coro)


def score_to_risk_level(score: float) -> str:
    if score < 0.25:
        return "LOW"
    elif score < 0.50:
        return "MEDIUM"
    elif score < 0.75:
        return "HIGH"
    return "CRITICAL"


def timed(agent_name: str):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            logger = get_logger(agent_name)
            t0 = time.time()
            result = await fn(*args, **kwargs)
            elapsed = round((time.time() - t0) * 1000, 1)
            logger.info(f"[{agent_name}] Completed in {elapsed}ms")
            return result
        return wrapper
    return decorator


def log_agent_output(agent_name: str, output: Any):
    logger = get_logger(agent_name)
    try:
        summary = output.model_dump() if hasattr(output, "model_dump") else str(output)
        logger.info(f"[{agent_name}] Output: {json.dumps(summary, default=str)[:500]}")
    except Exception:
        logger.info(f"[{agent_name}] Output logged (serialization skipped)")


def create_log_entry(agent_name: str, status: str, data: Any) -> dict:
    return {
        "agent": agent_name,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": data if isinstance(data, dict) else str(data)[:200],
    }
