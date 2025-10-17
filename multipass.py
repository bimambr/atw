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
import csv
import json
import logging
import signal
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar, cast

import aiohttp

# --- DEFAULT CONFIGURATION ---
ENDPOINT = "http://localhost:8000/api/v1/chat/completions"
MODEL_NAME = "user.gemma-3n-E4B-it-GGUF"
REFINEMENT_TEMP = 0
DRAFT_TEMP = 0.7
DEFAULT_N_ITERATIONS = 5
MAX_N_ITERATIONS = 10
# provide up to MAX_N_ITERATIONS + 1 iterations worth of seeds
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111]
TIMEOUT = 240
SYSTEM_PROMPT = """You are a professional linguistic translator with a specialization in formal, legal, and literary texts. Your primary directive is to provide a precise and faithful translation from {SOURCE_LANG} to {TARGET_LANG}. Adhere strictly to the following principles:

1.  **Prioritize Semantic Equivalence:** Your goal is to translate the meaning, intent, and nuance of the source text, not just the literal words. Find the most natural and accurate phrasing in the {TARGET_LANG} that preserves the original message. Remember that sometimes a direct translation is not the most desirable method if it compromises meaning or readability.

2.  **Maintain Register and Tone:** The tone of your translation must be neutral, formal, and academic. Avoid colloquialisms, slang, or overly casual language unless it is a direct and intentional translation of such language in the source text.

3.  **Handle Idiomatic and Cultural Nuances:** When encountering an idiom or culturally specific reference, provide the closest functional equivalent in the {TARGET_LANG}. If a direct equivalent does not exist, provide a translation that preserves the intended meaning rather than a literal, and potentially nonsensical, rendering.

4.  **Do Not Omit Information:** Do not trim anything from the source unless it is a direct translation. You may add more information in the event that there is a cultural item that does not exist in the target language. You may format it as a parenthetical explanation or as a translator note.

5.  **Output Format is Absolute:** Your response must contain ONLY the translated text and nothing else. Do not include introductory phrases like "Here is the translation:" or any other conversational filler.
"""
DRAFT_PROMPT = """Translate the following {SOURCE_LANG} text to {TARGET_LANG}. Provide only the translated text, without any additional explanations or introductions.

**Text type:**
{TEXT_TYPE}

**Source text:**
{SOURCE_TEXT}

**{TARGET_LANG} translation:**
"""
REFINEMENT_PROMPT = """Your task is to review and refine a draft translation. The original source text is the ultimate source of truth. The provided draft is a suggestion that may contain errors in tone, accuracy, or grammar.

Critically evaluate the draft against the original source text and provide a final, improved translation that strictly adheres to all of your system prompt's principles.

**Text type:**
{TEXT_TYPE}

**Original Source Text:**
{SOURCE_TEXT}

**Draft Translation to Review:**
{DRAFT_TRANSLATION}

**Your Final, Improved Translation:**
"""
# --- END OF CONFIGURATION ---

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

T = TypeVar("T")


class Bail(Exception): ...


class CLIArgs(argparse.Namespace):
    endpoint: str
    model: str
    iterations: int
    input: str
    timeout: int


parser = argparse.ArgumentParser(description="Ollama Translation Experiment")
parser.add_argument(
    "--endpoint",
    default=ENDPOINT,
    help=f"Ollama API endpoint URL (default: {ENDPOINT})",
)
parser.add_argument(
    "--model",
    default=MODEL_NAME,
    help=f"Ollama model name (default: {MODEL_NAME})",
)
parser.add_argument(
    "--iterations",
    type=lambda x: min(int(x), MAX_N_ITERATIONS),
    default=DEFAULT_N_ITERATIONS,
    help=f"Number of iterations per temperature (default: {DEFAULT_N_ITERATIONS}, cap: {MAX_N_ITERATIONS})",
)
parser.add_argument(
    "--input",
    default="*",
    required=True,
    help="Path to the input JSON file(s) containing the source text to translate. Use `,` to separate multiple files (default: '*')",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=TIMEOUT,
    help=f"Timeout for API requests in seconds (default: {TIMEOUT})",
)


async def generate_translation(
    client: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    seed: int,
    timeout: int,
) -> str:
    LOGGER.info("Generating translation with temp=%f, seed=%d", temperature, seed)
    try:
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

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


async def main():
    LOGGER.info("Starting Lemonade translation experiment...")

    args = parser.parse_args(namespace=CLIArgs())
    input_files = [Path(p) for p in args.input.split(",")]
    output_files = [
        (
            p.parent
            / "multipass_attempts"
            / f"{p.stem}_translated_{args.model}_attempt.csv"
        )
        for p in input_files
    ]

    LOGGER.info("Model: %s", args.model)
    LOGGER.info("Iterations per temperature: %d", args.iterations)
    LOGGER.info("Input files: %s", args.input)

    event = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_args: signal_handler(event))

    async with aiohttp.ClientSession() as client:
        for file_idx, (input_file, output_file) in enumerate(
            zip(input_files, output_files)
        ):
            if not input_file.exists():
                LOGGER.error("Input file '%s' does not exist.", input_file)
                continue

            LOGGER.info(
                "--- Processing file %d out of %d ---", file_idx + 1, len(input_files)
            )
            LOGGER.info("Processing input file: %s", input_file)
            LOGGER.info("Output will be saved to: %s", output_file)
            try:
                with output_file.open("w", newline="", encoding="utf-8") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(
                        [
                            "iteration",
                            "draft_seed",
                            "draft_temp",
                            "refinement_seed",
                            "refinement_temp",
                            "source_text",
                            "draft_translation",
                            "final_translation",
                            "timestamp",
                        ]
                    )

                    input_json = json.loads(input_file.read_text("utf-8").strip())

                    for text_idx, text in enumerate(input_json["texts"]):
                        LOGGER.info(
                            "--- Translating text %d out of %d ---",
                            text_idx + 1,
                            len(input_json["texts"]),
                        )
                        draft_prompt = DRAFT_PROMPT.format(
                            SOURCE_TEXT=text,
                            SOURCE_LANG=input_json["source_lang"],
                            TARGET_LANG=input_json["target_lang"],
                            TEXT_TYPE=input_json.get("type", "general"),
                        )
                        filled_system_prompt = SYSTEM_PROMPT.format(
                            SOURCE_LANG=input_json["source_lang"],
                            TARGET_LANG=input_json["target_lang"],
                        )

                        for i in range(args.iterations):
                            iteration_num = i + 1
                            current_seed = SEEDS[i]
                            LOGGER.info(
                                "--- Iteration %d/%d, Seed: %d ---",
                                iteration_num,
                                args.iterations,
                                current_seed,
                            )

                            LOGGER.info("Generating draft translation...")
                            draft_translation = await wait(
                                generate_translation(
                                    client,
                                    args.endpoint,
                                    args.model,
                                    draft_prompt,
                                    filled_system_prompt,
                                    DRAFT_TEMP,
                                    current_seed,
                                    timeout=args.timeout,
                                ),
                                event,
                            )
                            refinement_prompt = REFINEMENT_PROMPT.format(
                                SOURCE_TEXT=text,
                                SOURCE_LANG=input_json["source_lang"],
                                TARGET_LANG=input_json["target_lang"],
                                TEXT_TYPE=input_json.get("type", "general"),
                                DRAFT_TRANSLATION=draft_translation,
                            )
                            refinement_seed = SEEDS[i + 1]
                            LOGGER.info(
                                "Generating final refined translation using seed %d...",
                                refinement_seed,
                            )
                            final_translation = await wait(
                                generate_translation(
                                    client,
                                    args.endpoint,
                                    args.model,
                                    refinement_prompt,
                                    filled_system_prompt,
                                    REFINEMENT_TEMP,
                                    refinement_seed,
                                    timeout=args.timeout,
                                ),
                                event,
                            )
                            csv_writer.writerow(
                                [
                                    iteration_num,
                                    current_seed,
                                    DRAFT_TEMP,
                                    refinement_seed,
                                    REFINEMENT_TEMP,
                                    text,
                                    draft_translation,
                                    final_translation,
                                    time.ctime(),
                                ]
                            )
                            csvfile.flush()
            except IOError as e:
                LOGGER.error(
                    "Could not write to file %s. Reason: %s", output_file, e, exc_info=e
                )
                return
            except Bail:
                LOGGER.info("Experiment interrupted by user.")
                return

    LOGGER.info(
        "Experiment complete. Results saved to %s.",
        ", ".join([str(i) for i in output_files]),
    )


if __name__ == "__main__":
    asyncio.run(main())
