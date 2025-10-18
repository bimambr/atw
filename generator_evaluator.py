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
EVALUATOR_TEMP = 0.1
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
GENERATOR_SYSTEM_PROMPT = """You are a professional linguistic translator. Your primary directive is to provide a fluent, accurate, and contextually appropriate translation from {SOURCE_LANG} to {TARGET_LANG}.

--- CORE PRINCIPLES ---
1.  **Meaning and Nuance:** Translate the core meaning, intent, and nuance of the source text.
2.  **Tone and Register:** Match the tone of the source text (e.g., formal, literary, technical).
3.  **Fluency:** The final translation must read naturally in the {TARGET_LANG}.

--- OUTPUT FORMAT ---
Your response must contain ONLY the translated text. Do not include any introductory phrases or explanations.
"""
GENERATOR_INIT_USER_PROMPT = """--- CONTEXT ---
Text type: {TEXT_TYPE}
Source Language: {SOURCE_LANG}
Target Language: {TARGET_LANG}

--- SOURCE TEXT ---
{SOURCE_TEXT}

--- TASK ---
Provide the translation in {TARGET_LANG}:
"""
EVALUATOR_SYSTEM_PROMPT = """You are a Quality Assurance Gatekeeper for a prestigious publishing house. Your sole purpose is to protect the company's reputation by rejecting any translation that is not of the absolute highest quality. You are known for being extremely strict, fair, and having an eye for detail.

--- MANDATORY EVALUATION RUBRIC ---
You MUST first evaluate the translation against the following four criteria. For each criterion, you must assign a grade of **PASS** or **FAIL**.

1.  **Semantic Accuracy:** Does it perfectly preserve the meaning, including all subtext and implications? Make sure to check for any mistranslations or omissions and see if there is any phrase that is translated without considering the full, broader context.
2.  **Tonal Fidelity:** Does it match the source text's tone (e.g., literary, formal, informal) precisely?
3.  **Natural Fluency:** Does it read like a native speaker wrote it, with no awkward phrasing or grammatical errors?
4.  **Nuance Preservation:** Are subtle cultural references, wordplay, or literary devices handled effectively?

--- REQUIRED OUTPUT STRUCTURE ---
Your entire response MUST follow this structure in this exact order:

1.  **The Rubric Scorecard:** First, list your grades for the four criteria.
2.  **The Final Grade:** On a new line, provide the overall grade: `acceptable` or `needs_revision`.
3.  **The Critique:** Following the final grade, provide your detailed analysis explaining the reasoning behind your scorecard and final grade.

--- CRUCIAL RULE ---
If even ONE criterion in your scorecard is marked as **FAIL**, the final grade MUST be `needs_revision`. You can only give a grade of `acceptable` if all four criteria are a **PASS**.
"""
EVALUATOR_USER_PROMPT = """--- CONTEXT ---
Text type: {TEXT_TYPE}
Source Language: {SOURCE_LANG}
Target Language: {TARGET_LANG}

--- SOURCE TEXT ---
{SOURCE_TEXT}

--- TRANSLATION TO EVALUATE ---
{TRANSLATION_ATTEMPT}

--- TASK ---
Provide your grade and critique based on your system instructions.
"""
# https://github.com/ggml-org/llama.cpp/tree/master/grammars
EVALUATOR_JSON_GRAMMAR = r"""boolean ::= ("true" | "false") space
char ::= [^"\\\x7f\x00-\x1f] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
feedback-kv ::= "\"feedback\"" space ":" space string
grade ::= ("\"acceptable\"" | "\"needs_revision\"") space
grade-kv ::= "\"grade\"" space ":" space grade
root ::= "{" space rubric-kv "," space grade-kv "," space feedback-kv "}" space
rubric ::= "{" space rubric-semantic-accuracy-kv "," space rubric-tonal-fidelity-kv "," space rubric-natural-fluency-kv "," space rubric-nuance-preservation-kv "}" space
rubric-kv ::= "\"rubric\"" space ":" space rubric
rubric-natural-fluency-kv ::= "\"natural_fluency\"" space ":" space boolean
rubric-nuance-preservation-kv ::= "\"nuance_preservation\"" space ":" space boolean
rubric-semantic-accuracy-kv ::= "\"semantic_accuracy\"" space ":" space boolean
rubric-tonal-fidelity-kv ::= "\"tonal_fidelity\"" space ":" space boolean
space ::= | " " | "\n"{1,2} [ \t]{0,20}
string ::= "\"" char* "\"" space
"""
JSON_FORMATTER_SYSTEM_PROMPT = """You are a highly efficient text-parsing robot. Your only function is to extract structured data from a given text and format it as a JSON object. You do not re-interpret, evaluate, or change the information. You only extract and format.
"""
JSON_FORMATTER_USER_PROMPT = """Please parse the following evaluation text and convert it into a valid JSON object with three keys: "rubric" (an object with four boolean keys), "grade" (a string), and "feedback" (a string).

--- TEXT TO PARSE ---
{EVALUATION_TEXT}

--- JSON OUTPUT ---
"""
RETRY_PROMPT = """A previous translation attempt was evaluated and requires a complete rewrite. Your task is to deeply consider the editor's critique and generate a completely new version of the translation that addresses the identified problems.

**Start again from scratch, keeping the feedback in mind.**

--- CONTEXT ---
Text type: {TEXT_TYPE}
Source Language: {SOURCE_LANG}
Target Language: {TARGET_LANG}

--- SOURCE TEXT ---
{SOURCE_TEXT}

--- EDITOR'S CRITIQUE TO ADDRESS ---
{FEEDBACK}

--- FINAL TASK ---
Generate a new, final, and improved translation in {TARGET_LANG}. Your output must be only the clean, final text.
"""

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Rubric(TypedDict):
    semantic_accuracy: bool
    tonal_fidelity: bool
    natural_fluency: bool
    nuance_preservation: bool


class TranslationAttempt(TypedDict):
    translation: str
    rubric: Rubric
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
        "rubric": {},
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
    free_form_output = await run_inference(
        state["client"],
        args.endpoint,
        args.model,
        evaluation_prompt,
        EVALUATOR_SYSTEM_PROMPT,
        EVALUATOR_TEMP,
        state["evaluator_seed"],
        timeout=args.timeout,
    )
    json_formatter_prompt = JSON_FORMATTER_USER_PROMPT.format(
        EVALUATION_TEXT=free_form_output
    )
    json_output_str = await run_inference(
        state["client"],
        args.endpoint,
        args.model,
        json_formatter_prompt,
        JSON_FORMATTER_SYSTEM_PROMPT,
        temperature=0.0,
        seed=1,
        timeout=args.timeout,
        grammar=EVALUATOR_JSON_GRAMMAR,
    )
    json_output = json.loads(json_output_str)
    last_attempt.update(json_output)

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
            "\n".join(f"{k}: {v}" for k, v in last_attempt["rubric"].items()),
            last_attempt["grade"],
            last_attempt["feedback"],
            time.ctime(),
        )
    )

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
        FEEDBACK=last_attempt["feedback"],
        TEXT_TYPE=state["source_text"]["type"],
        SOURCE_LANG=state["source_text"]["source_lang"],
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
        "rubric": {},
        "grade": "",
        "feedback": "",
    }


1


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
                            "rubric",
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
