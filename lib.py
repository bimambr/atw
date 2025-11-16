"""
Copyright 2025 Muhammad Bima Ramadhan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Awaitable, TypedDict, TypeVar, cast

import aiohttp

LOGGER = logging.getLogger("lib")

T = TypeVar("T")


class Bail(Exception): ...


ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "gemma-3n-E4B-it-GGUF"
DEFAULT_N_ITERATIONS = 5
MAX_N_ITERATIONS = 10
DEFAULT_REFINEMENT_ITERATIONS = 3
MAX_REFINEMENT_ITERATIONS = 5
# provide up to MAX_N_ITERATIONS iterations worth of seeds
TIMEOUT = 240


class Payload(TypedDict, total=False):
    model: str
    stream: bool
    temperature: float
    seed: int
    messages: list[dict[str, str]]
    cache_prompt: bool
    grammar: str


async def stream_response(response: aiohttp.ClientResponse) -> str:
    json_data = ""
    full_response = ""
    chunk = ""

    LOGGER.info("Streaming response...")
    async for line in response.content:
        decoded_line = line.decode("utf-8").strip()
        if not decoded_line.startswith("data: "):
            continue
        data = decoded_line[len("data: ") :].strip()
        if data == "[DONE]":
            break
        try:
            json_data = json.loads(data)
            chunk = (
                json_data.get("choices", [{}])[0].get("delta", {}).get("content")
            ) or ""
            full_response += chunk
            print(chunk, end="", flush=True)
        except json.JSONDecodeError:
            LOGGER.error("Failed to decode JSON chunk: %s", data)

    if not chunk.endswith("\n"):
        print()

    LOGGER.info("Completed streaming response. Last chunk: %s", json_data)
    return full_response.strip()


async def run_inference(
    client: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    temperature: float,
    seed: int,
    timeout: float,
    grammar: str | None = None,
    cache_prompt: bool = False,
    messages: list[tuple[str, str, str]] | None = None,
) -> str:
    LOGGER.info("Hitting %s with temp=%f, seed=%d", endpoint, temperature, seed)
    for i in range(3):
        LOGGER.debug("Trying attempt %d...", i + 1)
        try:
            formatted_messages = [
                {"role": role, "content": content, "name": name}
                for role, content, name in messages or []
            ]
            if not formatted_messages:
                raise ValueError("Messages must be provided for inference.")

            payload: Payload = {
                "model": model,
                "stream": True,
                "temperature": temperature,
                "seed": seed,
                "messages": formatted_messages,
                "cache_prompt": cache_prompt,
            }

            if grammar is not None:
                payload["grammar"] = grammar

            async with client.post(
                endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response.raise_for_status()
                return await stream_response(response)
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as e:
            LOGGER.error("API request failed: %s", e, exc_info=e)
            LOGGER.info("Retrying in 1 second...")
            await asyncio.sleep(1)
            continue
        except json.JSONDecodeError:
            LOGGER.error("Failed to decode JSON from response.")
            return "Failed to decode JSON from response."

    return "API request failed"


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


class CLIArgs(argparse.Namespace):
    endpoint: str
    model: str
    iterations: int
    input: str
    timeout: int
    refinement_iterations: int
    simulate_thinking: bool
    simple_evaluator: bool
    cache_prompt: bool
    omit_roles: bool
    preserve_history: bool
    save_output: bool


def get_parsed_args() -> type[CLIArgs]:
    parser = argparse.ArgumentParser(description="llama.cpp Translation Experiment")
    parser.add_argument(
        "--endpoint",
        default=ENDPOINT,
        help=f"OpenAI-like API endpoint URL (default: {ENDPOINT})",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name to use (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--iterations",
        type=lambda x: min(int(x), MAX_N_ITERATIONS),
        default=DEFAULT_N_ITERATIONS,
        help=f"Number of iterations per temperature (default: {DEFAULT_N_ITERATIONS}, cap: {MAX_N_ITERATIONS})",
    )
    parser.add_argument(
        "--refinement-iterations",
        type=lambda x: min(int(x), MAX_REFINEMENT_ITERATIONS),
        default=DEFAULT_REFINEMENT_ITERATIONS,
        help=f"Number of refinement iterations (default: {DEFAULT_REFINEMENT_ITERATIONS}, cap: {MAX_REFINEMENT_ITERATIONS})",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file(s) containing the source text to translate. Use `,` to separate multiple files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT,
        help=f"Timeout for API requests in seconds (default: {TIMEOUT})",
    )
    parser.add_argument(
        "--simulate-thinking",
        action="store_true",
        default=False,
        help="Simulate 'thinking' time by asking the model to analyse the text before attempting translation",
    )
    parser.add_argument(
        "--simple-evaluator",
        action="store_true",
        default=False,
        help="Use a simpler evaluation prompt",
    )
    parser.add_argument(
        "--cache-prompt",
        action="store_true",
        default=False,
        help="Cache the prompt for faster subsequent requests",
    )
    parser.add_argument(
        "--omit-roles",
        action="store_true",
        default=False,
        help="Omit roles in system prompts",
    )
    parser.add_argument(
        "--preserve-history",
        action="store_true",
        default=False,
        help="Preserve the full interaction history when optimizing translations.",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        default=True,
        help="Do not save the output to a file.",
        dest="save_output",
    )
    parsed = parser.parse_args(namespace=CLIArgs)

    LOGGER.info("Using endpoint: %s", parsed.endpoint)
    LOGGER.info("Model: %s", parsed.model)
    LOGGER.info("Iterations per seed: %d", parsed.iterations)
    LOGGER.info("Refinement iterations: %d", parsed.refinement_iterations)
    LOGGER.info("Input files: %s", parsed.input)
    LOGGER.info("Timeout: %d seconds", parsed.timeout)
    LOGGER.info("Simulate thinking: %s", parsed.simulate_thinking)
    LOGGER.info("Simple evaluator: %s", parsed.simple_evaluator)
    LOGGER.info("Cache prompt: %s", parsed.cache_prompt)
    LOGGER.info("Omit roles: %s", parsed.omit_roles)
    LOGGER.info("Preserve history: %s", parsed.preserve_history)
    LOGGER.info("Save output: %s", parsed.save_output)

    return parsed
