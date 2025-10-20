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

import asyncio
import csv
import json
import logging
import re
import signal
import time
from pathlib import Path
from typing import TypedDict

import aiohttp

from lib import (
    Bail,
    get_next_available_path,
    get_parsed_args,
    run_inference,
    signal_handler,
    wait,
)

EVALUATOR_TEMP = 0.1
OPTIMIZER_TEMP = 1.4
OPTIMIZER_ALT_TEMP = 0.7
EVALUATOR_SEED = 727
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
ARGS = get_parsed_args()
OPTIMIZER_SYSTEM_PROMPT = f"""{
    "You are a professional linguistic translator. " * (not ARGS.omit_roles)
}Your primary directive is to provide a fluent, accurate, and contextually appropriate translation from {{SOURCE_LANG}} to {{TARGET_LANG}}.

--- REQUIREMENTS ---
1.  Translate the core meaning, intent, and nuance of the source text. The translation must be following conventions and idioms of the target language and text type. You may change the structure of sentences as needed, or use equivalent expressions in {{TARGET_LANG}} to best convey the original meaning.
2.  Match the tone of the source text (e.g., formal, literary, technical) and use register appropriate for the text type.
3.  The final translation must read naturally in the {{TARGET_LANG}}. If it means changing phrases or sentence structures (e.g., merging two sentences into a single one) to achieve fluency, do so.

--- OUTPUT FORMAT ---
{
    '''
The output is divided under two headers with the following structure:
1.  --- ANALYSIS ---: You must first provide a detailed analysis of each sentence/phrase in the source text, highlighting potential challenges and methods you will use to address them in your translation.
2.  --- FINAL TRANSLATION ---: Implement your analysis, providing the clean, final translation in {{TARGET_LANG}}. Your output must be only the clean, final text.
    '''
    if ARGS.simulate_thinking
    else "The output is the clean, final translation in {{TARGET_LANG}}. You must not include any additional commentary or analysis."
}

"""
OPTIMIZER_INIT_USER_PROMPT = """--- CONTEXT ---
Text type: {TEXT_TYPE}
Source Language: {SOURCE_LANG}
Target Language: {TARGET_LANG}

--- SOURCE TEXT ---
{SOURCE_TEXT}

--- TASK ---
Provide the translation in {TARGET_LANG}:
"""
EVALUATOR_COMPLEX_SYSTEM_PROMPT = f"""{"You are a Quality Assurance Gatekeeper for a prestigious publishing house. Your sole purpose is to protect the company's reputation by rejecting any translation that is not of the absolute highest quality. " * (not ARGS.omit_roles)}You are known for being extremely strict, fair, and having an eye for detail.

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
EVALUATOR_SIMPLE_SYSTEM_PROMPT = f"""{"You are a meticulous and highly critical linguistic editor tasked with evaluating translations. " * (not ARGS.omit_roles)}Your goal is to ensure that every translation meets the highest standards of quality.

--- REQUIREMENTS ---
1. The translation must be faithful to the original text in meaning, tone, and style.
2. The translation must read naturally in the target language.
3. You must provide constructive feedback highlighting any issues or areas for improvement.
4. Do not sugarcoat your assessment; be direct and precise.

--- OUTPUT FORMAT ---
Your response must include the following sections in order:
1. Analyse the tone, style, and meaning of the source text under the `--- ANALYSIS ---`.
2. Evaluate the translation attempt against the source text under the `--- EVALUATION ---`.
3. Provide your final assessment under the `--- VERDICT ---` header, without the subticks:
    - Respond only with "acceptable" if you find the translation meets all quality standards, free of ANY issues.
    - Otherwise, respond with "needs_revision", followed by a comprehensive feedback on how to improve the translation and maintain the meaning, tone, and style of its original source text. Give specific examples for each problematic phrase. If you find a phrase with multiple meanings depending on the context, explore all the possible meanings in differing contexts for the translator to decide. Do not emphasize your changes with formatting.
"""
EVALUATOR_USER_PROMPT = """--- CONTEXT ---
Text type: {TEXT_TYPE}
Source Language: {SOURCE_LANG}
Target Language: {TARGET_LANG}

--- SOURCE TEXT ---
{SOURCE_TEXT}

--- TRANSLATION TO EVALUATE ---
{TRANSLATION_ATTEMPT}

--- OUTPUT ---
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
JSON_FORMATTER_SYSTEM_PROMPT = """You are a highly efficient text-parsing robot. Your only function is to extract structured data from a given text and format it as a JSON object. You do not re-interpret, evaluate, or change the information. You only extract and format."""
JSON_FORMATTER_USER_PROMPT = """Please parse the following evaluation text and convert it into a valid JSON object with three keys: "rubric" (an object with four boolean keys), "grade" (a string), and "feedback" (a string).

--- TEXT TO PARSE ---
{EVALUATION_TEXT}

--- JSON OUTPUT ---
"""
OPTIMIZER_RETRY_PROMPT = """A previous translation attempt was evaluated and requires a complete rewrite. Your task is to deeply consider the editor's feedback and generate a completely new version of the translation that addresses the identified problems.

Start again from scratch, keeping the feedback in mind.

--- CONTEXT ---
Text type: {TEXT_TYPE}
Source Language: {SOURCE_LANG}
Target Language: {TARGET_LANG}

--- SOURCE TEXT ---
{SOURCE_TEXT}

--- EDITOR'S FEEDBACK ---
{FEEDBACK}

--- FINAL TASK ---
Generate a new, final, and improved translation in {TARGET_LANG}:
"""
VERIFIER_SYSTEM_PROMPT = """You are a robotic and literal Quality Assurance Verifier. Your only function is to check if a revised text has correctly implemented a set of required changes. You do not have opinions or creative ideas.

--- REQUIRED OUTPUT ---
Your entire response must be a single word: `pass` or `fail`.
-   Output `pass` if the New Translation successfully fixed the problems described in the Original Critique.
-   Output `fail` if it did not.
"""
VERIFIER_USER_PROMPT = """--- ORIGINAL CRITIQUE (The Requirements) ---
{ORIGINAL_FEEDBACK}

--- NEW TRANSLATION (The Implementation) ---
{NEW_TRANSLATION_ATTEMPT}

--- TASK ---
Did the New Translation successfully implement the changes described in the Original Critique? Respond with only `pass` or `fail`.
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
    prompt: str
    system_prompt: str
    seed: int
    temp: float


class SourceTextEntry(TypedDict):
    source_lang: str
    target_lang: str
    text: str
    type: str
    id: int


class State(TypedDict):
    iteration_id: int
    source_text: SourceTextEntry
    next_state: str
    max_attempt: int
    attempt: int
    last_attempt: TranslationAttempt
    optimizer_seed: int
    evaluator_seed: int
    client: aiohttp.ClientSession
    csv_writer: csv.writer


async def handle_optimization_state(state: State) -> None:
    state["attempt"] += 1

    is_draft = state["attempt"] == 1

    LOGGER.info(
        "Starting %s for text %d, iteration %d/%d, attempt %d/%d",
        "draft generation" if is_draft else "refinement",
        state["source_text"]["id"],
        state["iteration_id"],
        ARGS.iterations,
        state["attempt"],
        state["max_attempt"],
    )

    if is_draft:
        prompt = OPTIMIZER_INIT_USER_PROMPT.format(
            SOURCE_TEXT=state["source_text"]["text"],
            SOURCE_LANG=state["source_text"]["source_lang"],
            TARGET_LANG=state["source_text"]["target_lang"],
            TEXT_TYPE=state["source_text"]["type"],
        )
        system_prompt = OPTIMIZER_SYSTEM_PROMPT.format(
            SOURCE_LANG=state["source_text"]["source_lang"],
            TARGET_LANG=state["source_text"]["target_lang"],
        )
        state["next_state"] = "evaluation"
    else:
        last_attempt = state["last_attempt"]
        prompt = OPTIMIZER_RETRY_PROMPT.format(
            SOURCE_TEXT=state["source_text"]["text"],
            FEEDBACK=last_attempt["feedback"],
            TEXT_TYPE=state["source_text"]["type"],
            SOURCE_LANG=state["source_text"]["source_lang"],
            TARGET_LANG=state["source_text"]["target_lang"],
        )
        system_prompt = OPTIMIZER_SYSTEM_PROMPT.format(
            SOURCE_LANG=state["source_text"]["source_lang"],
            TARGET_LANG=state["source_text"]["target_lang"],
        )
        state["next_state"] = "verification"

    temp = OPTIMIZER_TEMP if is_draft else OPTIMIZER_ALT_TEMP
    seed = state["optimizer_seed"] * 10 + state["attempt"]
    translation = await run_inference(
        state["client"],
        ARGS.endpoint,
        ARGS.model,
        prompt,
        system_prompt,
        temp,
        seed,
        timeout=ARGS.timeout,
        cache_prompt=ARGS.cache_prompt,
    )

    if ARGS.simulate_thinking:
        match = re.search("--- FINAL TRANSLATION ---\s*:?", translation, re.IGNORECASE)
        if not match:
            LOGGER.error(
                "Could not find final translation section in output: %s", translation
            )
            translation = "Parsing error"
        else:
            translation = translation[match.end() :].strip()

    state["last_attempt"]["translation"] = translation
    state["last_attempt"]["prompt"] = prompt
    state["last_attempt"]["system_prompt"] = system_prompt
    state["last_attempt"]["seed"] = seed
    state["last_attempt"]["temp"] = temp


async def handle_evaluation_state(state: State) -> None:
    LOGGER.info(
        "Starting evaluation for text %d, iteration %d/%d, attempt %d/%d",
        state["source_text"]["id"],
        state["iteration_id"],
        ARGS.iterations,
        state["attempt"],
        state["max_attempt"],
    )
    # do not mutate the original evaluator seed
    seed = state["evaluator_seed"] + state["iteration_id"] * 100
    last_attempt = state["last_attempt"]
    system_prompt = (
        EVALUATOR_SIMPLE_SYSTEM_PROMPT
        if ARGS.simple_evaluator
        else EVALUATOR_COMPLEX_SYSTEM_PROMPT
    )
    prompt = EVALUATOR_USER_PROMPT.format(
        SOURCE_TEXT=state["source_text"]["text"],
        SOURCE_LANG=state["source_text"]["source_lang"],
        TARGET_LANG=state["source_text"]["target_lang"],
        TRANSLATION_ATTEMPT=last_attempt["translation"],
        TEXT_TYPE=state["source_text"]["type"],
    )
    free_form_output = await run_inference(
        state["client"],
        ARGS.endpoint,
        ARGS.model,
        prompt,
        system_prompt,
        EVALUATOR_TEMP,
        seed,
        timeout=ARGS.timeout,
        cache_prompt=ARGS.cache_prompt,
    )

    if ARGS.simple_evaluator:
        if not (
            match := re.search(r"--- VERDICT ---\s*:?", free_form_output, re.IGNORECASE)
        ):
            last_attempt["rubric"] = {}
            last_attempt["grade"] = "N/A"
            last_attempt["feedback"] = "Failed to find verdict section."
        else:
            verdict = free_form_output[match.end() :].strip()
            grade = verdict.split(maxsplit=1)[0].strip().lower()
            last_attempt["grade"] = grade
            last_attempt["feedback"] = free_form_output.strip()
            last_attempt["rubric"] = {}
    else:
        json_formatter_prompt = JSON_FORMATTER_USER_PROMPT.format(
            EVALUATION_TEXT=free_form_output
        )
        json_output_str = await run_inference(
            state["client"],
            ARGS.endpoint,
            ARGS.model,
            json_formatter_prompt,
            JSON_FORMATTER_SYSTEM_PROMPT,
            temperature=0.0,
            seed=1,
            timeout=ARGS.timeout,
            grammar=EVALUATOR_JSON_GRAMMAR,
            cache_prompt=ARGS.cache_prompt,
        )

        try:
            json_output = json.loads(json_output_str)
            last_attempt.update(json_output)
        except json.JSONDecodeError:
            LOGGER.error(
                "Failed to parse JSON from evaluator output: %s", json_output_str
            )
            last_attempt["rubric"] = {
                "semantic_accuracy": False,
                "tonal_fidelity": False,
                "natural_fluency": False,
                "nuance_preservation": False,
            }
            last_attempt["grade"] = "N/A"
            last_attempt["feedback"] = "Failed to parse evaluator output."

    state["csv_writer"].writerow(
        (
            state["source_text"]["id"],
            state["iteration_id"],
            state["attempt"],
            state["last_attempt"]["seed"],
            state["last_attempt"]["temp"],
            seed,
            EVALUATOR_TEMP,
            state["source_text"]["text"],
            last_attempt["translation"],
            "evaluator",
            "\n".join(f"{k}: {v}" for k, v in last_attempt["rubric"].items()) or "N/A",
            last_attempt["grade"],
            last_attempt["feedback"],
            time.ctime(),
            last_attempt["system_prompt"],
            last_attempt["prompt"],
            system_prompt,
            prompt,
        )
    )

    state["next_state"] = "optimization"
    if (
        "acceptable" in last_attempt["grade"]
        or state["attempt"] >= state["max_attempt"]
    ):
        state["next_state"] = ""


async def handle_verification_state(state: State) -> None:
    LOGGER.info(
        "Starting verification for text %d, iteration %d/%d, attempt %d/%d",
        state["source_text"]["id"],
        state["iteration_id"],
        ARGS.iterations,
        state["attempt"],
        state["max_attempt"],
    )
    last_attempt = state["last_attempt"]
    temp = 0.0
    # do not mutate the original evaluator seed
    seed = state["evaluator_seed"] + state["iteration_id"] * 200 + state["attempt"]
    verification_prompt = VERIFIER_USER_PROMPT.format(
        ORIGINAL_FEEDBACK=last_attempt["feedback"],
        NEW_TRANSLATION_ATTEMPT=last_attempt["translation"],
    )
    verification_output = await run_inference(
        state["client"],
        ARGS.endpoint,
        ARGS.model,
        verification_prompt,
        VERIFIER_SYSTEM_PROMPT,
        temperature=temp,
        seed=seed,
        timeout=ARGS.timeout,
        cache_prompt=ARGS.cache_prompt,
    )
    verification_result = verification_output.strip().lower()
    state["next_state"] = (
        ""
        if state["attempt"] >= state["max_attempt"] or verification_result == "pass"
        else "optimization"
    )
    state["csv_writer"].writerow(
        (
            state["source_text"]["id"],
            state["iteration_id"],
            state["attempt"],
            state["last_attempt"]["seed"],
            state["last_attempt"]["temp"],
            seed,
            temp,
            state["source_text"]["text"],
            last_attempt["translation"],
            "verifier",
            "N/A",
            verification_result,
            "N/A",
            time.ctime(),
            last_attempt["system_prompt"],
            last_attempt["prompt"],
            VERIFIER_SYSTEM_PROMPT,
            verification_prompt,
        )
    )
    # this will either get overridden by the next iteration
    # or get used in the next optimization state
    state["optimizer_seed"] += 1


async def main():
    LOGGER.info("Starting translation experiment...")
    input_files = [Path(p) for p in ARGS.input.split(",")]
    root = Path(__file__).parent
    output_files = [
        get_next_available_path(
            root
            / "evaluator_optimizer_attempts"
            / f"{p.stem}_translated_{ARGS.model}_attempt.csv"
        )
        for p in input_files
    ]

    LOGGER.info("Model: %s", ARGS.model)
    LOGGER.info("Iterations per seed: %d", ARGS.iterations)
    LOGGER.info("Refinement iterations: %d", ARGS.refinement_iterations)
    LOGGER.info("Optimizer temperature: %f", OPTIMIZER_TEMP)
    LOGGER.info("Evaluator temperature: %f", EVALUATOR_TEMP)
    LOGGER.info("Input files: %s", ARGS.input)
    LOGGER.info("Simulate thinking: %s", ARGS.simulate_thinking)
    LOGGER.info("Simple evaluator: %s", ARGS.simple_evaluator)

    event = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_args: signal_handler(event))

    state_handlers = {
        "optimization": handle_optimization_state,
        "evaluation": handle_evaluation_state,
        "verification": handle_verification_state,
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
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with output_file.open("w", newline="", encoding="utf-8") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(
                        [
                            "text_id",
                            "iteration_id",
                            "attempt",
                            "optimizer_seed",
                            "optimizer_temp",
                            "evaluator_seed",
                            "evaluator_temp",
                            "source_text",
                            "translation_attempt",
                            "evaluator_type",
                            "rubric",
                            "grade",
                            "feedback",
                            "timestamp",
                            "optimizer_system_prompt",
                            "optimizer_user_prompt",
                            "evaluator_system_prompt",
                            "evaluator_user_prompt",
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
                            "id": text_idx + 1,
                        }

                        for i in range(ARGS.iterations):
                            iteration_num = i + 1
                            LOGGER.info(
                                "=== Iteration %d out of %d ===",
                                iteration_num,
                                ARGS.iterations,
                            )

                            state = State(
                                iteration_id=iteration_num,
                                source_text=source_text,
                                next_state="optimization",
                                max_attempt=ARGS.refinement_iterations,
                                attempt=0,
                                last_attempt={},
                                optimizer_seed=SEEDS[i],
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
