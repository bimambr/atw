import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Awaitable, TypeVar, cast

import aiohttp

LOGGER = logging.getLogger("lib")

T = TypeVar("T")


class Bail(Exception): ...


async def run_inference(
    client: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    seed: int,
    timeout: int,
    grammar: str | None = None,
) -> str:
    LOGGER.info("Hitting %s with temp=%f, seed=%d", endpoint, temperature, seed)
    try:
        payload = {
            "model": model,
            "stream": False,
            "temperature": temperature,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "cache_prompt": False,
        }

        if grammar is not None:
            payload["grammar"] = grammar

        async with client.post(endpoint, json=payload, timeout=timeout) as response:
            LOGGER.info(
                "Received response with status code: %d, data: %s",
                response.status,
                await response.text(),
            )
            response.raise_for_status()
            full_response = (
                (await response.json())
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            return full_response
    except aiohttp.ClientError as e:
        LOGGER.error("API request failed: %s", e, exc_info=e)
        return "API request failed"
    except json.JSONDecodeError:
        LOGGER.error("Failed to decode JSON from response.")
        return "Failed to decode JSON from response."


async def wait(awaitable: Awaitable[T], event: asyncio.Event) -> T:
    async def _wrap(awaitable: Awaitable[T]) -> T:
        return await awaitable

    done, pending = await asyncio.wait(
        [
            asyncio.Task(_wrap(awaitable), name="coro"),
            asyncio.Task(event.wait(), name="event"),
        ],
        return_when=asyncio.FIRST_COMPLETED,
    )
    task = done.pop()

    try:
        for future in pending:
            future.cancel()
            await future
    except asyncio.CancelledError:
        pass

    if task.get_name() == "event":
        raise Bail

    return cast("T", task.result())


def signal_handler(event: asyncio.Event) -> None:
    LOGGER.info("Received CTRL+C")
    asyncio.get_running_loop().call_soon_threadsafe(lambda: event.set())


def get_next_available_path(path: Path) -> Path:
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    pattern = f"{stem}_*{suffix}"
    existing = parent.glob(pattern)

    max_index = 0
    for f in existing:
        f_match = re.match(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$", f.name)
        if f_match:
            idx = int(f_match.group(1))
            max_index = max(max_index, idx)

    next_index = max_index + 1
    return parent / f"{stem}_{next_index}{suffix}"
