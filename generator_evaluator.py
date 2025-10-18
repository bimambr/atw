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
from pathlib import Path
from typing import TypedDict

import aiohttp

from lib import Bail, run_inference, signal_handler, wait

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "gemma-3n-E4B-it-GGUF"
EVALUATOR_TEMP = 0.01
GENERATOR_TEMP = 1.4
GENERATOR_ALT_TEMP = 0.7  # used when applying suggestions
EVALUATOR_SEED = 727
DEFAULT_N_ITERATIONS = 5
MAX_N_ITERATIONS = 10
DEFAULT_REFINEMENT_ITERATIONS = 3
MAX_REFINEMENT_ITERATIONS = 5
# provide up to MAX_N_ITERATIONS iterations worth of seeds
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
TIMEOUT = 240
GENERATOR_SYSTEM_PROMPT = """<ROLE>
You are a professional linguistic translator. Your primary directive is to provide a precise and faithful translation from {SOURCE_LANG} to {TARGET_LANG}. If a Revision Plan is provided, then you must implement every point from the plan in your translation. Failure to do so will be considered a critical error.
</ROLE>

<PRINCIPLES>
You must adhere strictly to the following principles:

1.  **Semantic Equivalence:** Your goal is to translate the meaning, intent, and nuance of the source text, not just the literal words. Find the most natural phrasing in the {TARGET_LANG} that preserves the original message.
2.  **Register and Tone:** The tone of your translation must be neutral, formal, and academic. Avoid colloquialisms unless they are a direct and intentional translation of such language in the source text.
3.  **Cultural Nuances:** When encountering an idiom or culturally specific reference, provide the closest functional equivalent in the {TARGET_LANG}.
4.  **Information Fidelity:** Do not add or omit information. If a cultural item requires explanation, you may add a brief parenthetical note.
5.  **Text Type Adaptation:** Adapt your translation style to fit the specified text type (e.g., literary, legal, general).
</PRINCIPLES>

<FINAL_INSTRUCTION>
Your response must contain ONLY the translated text. Do not include any introductory phrases like "Here is the translation:" or any other conversational filler.
</FINAL_INSTRUCTION>
"""
GENERATOR_INIT_USER_PROMPT = """<CONTEXT>
**Text type:**
{TEXT_TYPE}

**Source text:**
{SOURCE_TEXT}
</CONTEXT>

<TASK>
Provide the {TARGET_LANG} translation:
</TASK>
"""
EVALUATOR_SYSTEM_PROMPT = """<ROLE>
You are a senior quality assurance editor. Your reputation depends on your ability to find flaws. Your default assumption is that every translation can be improved.
</ROLE>

<INTERNAL_THOUGHT_PROCESS>
(Do not write this in your output)
First, you must assess the text's context (audience, purpose, tone). Second, using that context, you must rigorously compare the translation to the source text to identify any and all potential flaws.
</INTERNAL_THOUGHT_PROCESS>

<OUTPUT_COMPONENTS>
Your output will consist of two potential parts:
1.  **A Revision Plan:** A numbered list of specific changes. Each item must follow this exact format:
    -   **Quote:** The exact phrase from the translation that needs to be changed.
    -   **Suggestion:** Your direct, improved replacement (clean and unformatted).
    -   **Reason:** A brief explanation for the change.

2.  **A Final Grade:** A single word: 'acceptable' or 'needs_revision'.
</OUTPUT_COMPONENTS>

<REQUIRED_OUTPUT_STRUCTURE>
Your entire response must ONLY contain the content described above.
- **Do not include any headings, titles, or introductory text whatsoever.** For example, do not write "Revision Plan:" or "Final Grade:".
- If revisions are needed, provide the numbered list of changes first.
- The final grade (`acceptable` or `needs_revision`) **must** be the very last thing in your response, on its own, separate line.
</REQUIRED_OUTPUT_STRUCTURE>

<EXAMPLE_OF_GOOD_OUTPUT>
1.  **Quote:** "Saya kangen sama Anda."
    **Suggestion:** "Aku kangen sama kamu."
    **Reason:** "Aku" and "kamu" sound more natural and intimate; "saya" and "Anda" are too formal for this context.
2.  **Quote:** "Yang menarik gerbong ke atas dan bawah jalan."
    **Suggestion:** "Yang menarik gerbong bolak balik."
    **Reason:** The phrase 'up and down the street' does not refer to vertical motion, but rather to the back-and-forth movement along the street.
needs_revision
</EXAMPLE_OF_GOOD_OUTPUT>

<CRUCIAL_RULE>
- If you find no flaws after your rigorous internal analysis, your entire response must be the single word: `acceptable`.
- In all other cases, you must provide both the revision plan and the 'needs_revision' grade according to the specified structure.
</CRUCIAL_RULE>
"""
EVALUATOR_USER_PROMPT = """<CONTEXT>
**Text type:**
{TEXT_TYPE}

**Original Source Text ({SOURCE_LANG}):**
{SOURCE_TEXT}

**Translation to Evaluate ({TARGET_LANG}):**
{TRANSLATION_ATTEMPT}
</CONTEXT>

<TASK>
Please provide your critical evaluation based on the four-step process defined in your system instructions.
</TASK>
"""
RETRY_PROMPT = """<CONTEXT>
**Text type:**
{TEXT_TYPE}

**Original Source Text:**
{SOURCE_TEXT}

**Your Previous Attempt (Contains errors):**
{TRANSLATION_ATTEMPT}
</CONTEXT>

<REVISION_PLAN>
{FEEDBACK}
</REVISION_PLAN>

<FINAL_INSTRUCTION>
Your single most important task is to generate a new translation that **implements every point** from the `REVISION_PLAN`. You must not skip any suggestions. Failure to implement all suggestions will result in a failed task. Produce only the single, clean, final block of text in {TARGET_LANG}.
</FINAL_INSTRUCTION>

<TASK>
Provide your new, final, and improved translation in {TARGET_LANG}:
</TASK>
"""

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TranslationAttempt(TypedDict):
    translation: str
    grade: str
    feedback: str


class SourceTextEntry(TypedDict):
    source_lang: str
    target_lang: str
    text: str
    type: str


class State(TypedDict):
    iteration_id: int
    source_text: SourceTextEntry
    next_state: str
    max_attempt: int
    attempt: int
    last_attempt: TranslationAttempt
    generator_seed: int
    evaluator_seed: int
    client: aiohttp.ClientSession
    csv_writer: csv.writer


class CLIArgs(argparse.Namespace):
    endpoint: str
    model: str
    iterations: int
    input: str
    timeout: int
    refinement_iterations: int


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
args = parser.parse_args(namespace=CLIArgs())


async def handle_draft_state(state: State) -> None:
    state["next_state"] = "evaluation"
    state["attempt"] += 1

    LOGGER.info(
        "Starting draft generation for iteration %d/%d, attempt %d/%d",
        state["iteration_id"],
        args.iterations,
        state["attempt"],
        state["max_attempt"],
    )

    draft_prompt = GENERATOR_INIT_USER_PROMPT.format(
        SOURCE_TEXT=state["source_text"]["text"],
        SOURCE_LANG=state["source_text"]["source_lang"],
        TARGET_LANG=state["source_text"]["target_lang"],
        TEXT_TYPE=state["source_text"]["type"],
    )
    system_prompt = GENERATOR_SYSTEM_PROMPT.format(
        SOURCE_LANG=state["source_text"]["source_lang"],
        TARGET_LANG=state["source_text"]["target_lang"],
    )
    draft_translation = await run_inference(
        state["client"],
        args.endpoint,
        args.model,
        draft_prompt,
        system_prompt,
        GENERATOR_TEMP,
        state["generator_seed"],
        timeout=args.timeout,
    )
    state["last_attempt"] = {
        "translation": draft_translation,
        "grade": "",
        "feedback": "",
    }


async def handle_evaluation_state(state: State) -> None:
    LOGGER.info(
        "Starting evaluation for iteration %d/%d, attempt %d/%d",
        state["iteration_id"],
        args.iterations,
        state["attempt"],
        state["max_attempt"],
    )
    last_attempt = state["last_attempt"]
    evaluation_prompt = EVALUATOR_USER_PROMPT.format(
        SOURCE_TEXT=state["source_text"]["text"],
        SOURCE_LANG=state["source_text"]["source_lang"],
        TARGET_LANG=state["source_text"]["target_lang"],
        TRANSLATION_ATTEMPT=last_attempt["translation"],
        TEXT_TYPE=state["source_text"]["type"],
    )
    output = await run_inference(
        state["client"],
        args.endpoint,
        args.model,
        evaluation_prompt,
        EVALUATOR_SYSTEM_PROMPT,
        EVALUATOR_TEMP,
        state["evaluator_seed"],
        timeout=args.timeout,
    )
    grade_raw = output.rsplit(maxsplit=1)[-1].lower()
    last_attempt["feedback"] = output[: len(grade_raw)].strip()
    last_attempt["grade"] = grade_raw.strip("\n *")

    state["csv_writer"].writerow(
        (
            state["iteration_id"],
            state["attempt"],
            state["generator_seed"],
            state["evaluator_seed"],
            GENERATOR_TEMP,
            EVALUATOR_TEMP,
            state["source_text"]["text"],
            last_attempt["translation"],
            last_attempt["grade"],
            last_attempt["feedback"],
            time.ctime(),
        )
    )

    if last_attempt["grade"] not in ("acceptable", "needs_revision"):
        LOGGER.warning("Unexpected grade '%s' received!", last_attempt["grade"])

    state["next_state"] = "refinement"
    if (
        "acceptable" in last_attempt["grade"]
        or state["attempt"] >= state["max_attempt"]
    ):
        state["next_state"] = "done"
        return


async def handle_refinement_state(state: State) -> None:
    state["next_state"] = "evaluation"
    state["attempt"] += 1

    LOGGER.info(
        "Starting refinement for iteration %d/%d, attempt %d/%d",
        state["iteration_id"],
        args.iterations,
        state["attempt"],
        state["max_attempt"],
    )

    last_attempt = state["last_attempt"]
    refinement_prompt = RETRY_PROMPT.format(
        SOURCE_TEXT=state["source_text"]["text"],
        TRANSLATION_ATTEMPT=last_attempt["translation"],
        FEEDBACK=last_attempt["feedback"],
        TEXT_TYPE=state["source_text"]["type"],
        TARGET_LANG=state["source_text"]["target_lang"],
    )
    system_prompt = GENERATOR_SYSTEM_PROMPT.format(
        SOURCE_LANG=state["source_text"]["source_lang"],
        TARGET_LANG=state["source_text"]["target_lang"],
    )
    refinement_translation = await run_inference(
        state["client"],
        args.endpoint,
        args.model,
        refinement_prompt,
        system_prompt,
        GENERATOR_ALT_TEMP,
        state["generator_seed"],
        timeout=args.timeout,
    )
    state["last_attempt"] = {
        "translation": refinement_translation,
        "grade": "",
        "feedback": "",
    }


async def main():
    LOGGER.info("Starting translation experiment...")
    input_files = [Path(p) for p in args.input.split(",")]
    output_files = [
        (
            p.parent
            / "generator_evaluator_attempts"
            / f"{p.stem}_translated_{args.model}_attempt.csv"
        )
        for p in input_files
    ]

    LOGGER.info("Model: %s", args.model)
    LOGGER.info("Iterations per seed: %d", args.iterations)
    LOGGER.info("Refinement iterations: %d", args.refinement_iterations)
    LOGGER.info("Generator temperature: %f", GENERATOR_TEMP)
    LOGGER.info("Evaluator temperature: %f", EVALUATOR_TEMP)
    LOGGER.info("Input files: %s", args.input)

    event = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_args: signal_handler(event))

    state_handlers = {
        "draft": handle_draft_state,
        "evaluation": handle_evaluation_state,
        "refinement": handle_refinement_state,
    }

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
                            "iteration_id",
                            "attempt",
                            "generator_seed",
                            "evaluator_seed",
                            "generator_temp",
                            "evaluator_temp",
                            "source_text",
                            "translation_attempt",
                            "grade",
                            "feedback",
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
                        source_text: SourceTextEntry = {
                            "source_lang": input_json["source_lang"],
                            "target_lang": input_json["target_lang"],
                            "text": text,
                            "type": input_json.get("type", "general"),
                        }

                        for i in range(args.iterations):
                            iteration_num = i + 1
                            LOGGER.info(
                                "=== Iteration %d out of %d ===",
                                iteration_num,
                                args.iterations,
                            )

                            state = State(
                                iteration_id=iteration_num,
                                source_text=source_text,
                                next_state="draft",
                                max_attempt=args.refinement_iterations,
                                attempt=0,
                                last_attempt={},
                                generator_seed=SEEDS[i],
                                evaluator_seed=EVALUATOR_SEED,
                                client=client,
                                csv_writer=csv_writer,
                            )

                            while handler := state_handlers.get(state["next_state"]):
                                await wait(handler(state), event)
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
