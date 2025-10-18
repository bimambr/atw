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
EVALUATOR_SEED = 727
DEFAULT_N_ITERATIONS = 5
MAX_N_ITERATIONS = 10
DEFAULT_REFINEMENT_ITERATIONS = 3
MAX_REFINEMENT_ITERATIONS = 5
# provide up to MAX_N_ITERATIONS iterations worth of seeds
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
TIMEOUT = 240
GENERATOR_SYSTEM_PROMPT = """You are a professional linguistic translator with a specialization in formal, legal, and literary texts. Your primary directive is to provide a precise and faithful translation from {SOURCE_LANG} to {TARGET_LANG}. Adhere strictly to the following principles:

1. **Prioritize Semantic Equivalence:** Your goal is to translate the meaning, intent, and nuance of the source text, not just the literal words. Find the most natural and accurate phrasing in the {TARGET_LANG} that preserves the original message. Remember that sometimes a direct translation is not the most desirable method if it compromises meaning or readability.
2. **Maintain Register and Tone:** The tone of your translation must be neutral, formal, and academic. Avoid colloquialisms, slang, or overly casual language unless it is a direct and intentional translation of such language in the source text.
3. **Handle Idiomatic and Cultural Nuances:** When encountering an idiom or culturally specific reference, provide the closest functional equivalent in the {TARGET_LANG}. If a direct equivalent does not exist, provide a translation that preserves the intended meaning rather than a literal, and potentially nonsensical, rendering.
4. **Do Not Omit Information:** Do not trim anything from the source unless it is a direct translation. You may add more information in the event that there is a cultural item that does not exist in the target language. You may format it as a parenthetical explanation or as a translator note.
5. **Output Format is Absolute:** Your response must contain ONLY the translated text and nothing else. Do not include introductory phrases like "Here is the translation:" or any other conversational filler.
6. **Text Type Consideration:** Adapt your translation style to fit the specified text type (e.g., literary, technical, legal, general). Each text type has its own conventions and expectations that must be respected.

If feedback is provided indicating issues with accuracy, tone, or register, you must revise the translation accordingly to address those specific concerns.
"""
GENERATOR_INIT_USER_PROMPT = """Translate the following {SOURCE_LANG} text to {TARGET_LANG}. Provide only the translated text, without any additional explanations or introductions.

**Text type:**
{TEXT_TYPE}

**Source text:**
{SOURCE_TEXT}

**{TARGET_LANG} translation:**
"""
EVALUATOR_SYSTEM_PROMPT = """You are a meticulous but fair translation evaluator. Your task is to grade a machine-generated translation against the original source text with a focus on identifying significant, material flaws.

Your evaluation must be guided by the **Principle of Materiality**: A flaw is only worth reporting if it materially impacts the translation's accuracy, fluency, or tone. Do not provide feedback based on subjective preference if the translation is already a valid and high-quality alternative.

You must follow a strict three-step critical process:
1.  **Identify Material Flaws:** First, compare the translation to the source text. Identify any inaccuracies, awkward phrasing, or tonal errors that tangibly harm the quality of the translation. If you cannot find any material flaws, you must explicitly state: "No material flaws found."
2.  **Suggest Improvements:** Based only on the material flaws you identified, provide concrete, actionable suggestions for how the translator can fix the specific issues.
3.  **Provide a Final Grade:** After your analysis, provide a final grade on a new line. The grade must be one of two options: 'acceptable' or 'needs_revision'.
    - Grade as 'needs_revision' if you identified one or more material flaws.
    - Grade as 'acceptable' ONLY if the translation is a high-quality, professional rendering of the source text, even if you can imagine other ways to phrase it.

Your entire output must follow this structure. Always put the final grade on its own line at the end of your response.
"""
EVALUATOR_USER_PROMPT = """Please evaluate the following translation using your three-step critical process.

**Text type:**
{TEXT_TYPE}

**Original Source Text ({SOURCE_LANG}):**
{SOURCE_TEXT}

**Translation to Evaluate ({TARGET_LANG}):**
{TRANSLATION_ATTEMPT}

**Your Critical Evaluation:**
"""
RETRY_PROMPT = """Please try translating the following text again. A previous attempt was evaluated and requires revision. Use the provided feedback to create a new, improved translation.

**Text type:**
{TEXT_TYPE}

**Original Source Text:**
{SOURCE_TEXT}

**Your previous attempt (contains errors):**
{TRANSLATION_ATTEMPT}

**Feedback on Previous Attempt:**
{FEEDBACK}

**Your New, Improved Translation:**
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
    last_attempt["grade"] = grade = output.rsplit(maxsplit=1)[0].strip().lower()
    last_attempt["feedback"] = output[: len(grade)].strip()

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

    state["next_state"] = "refinement"
    if (
        last_attempt["grade"] == "acceptable"
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
        GENERATOR_TEMP,
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
                            "evaluator_temp",
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
