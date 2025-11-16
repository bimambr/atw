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
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol, TypedDict

import aiohttp

from lib import (
    Bail,
    get_next_available_path,
    get_parsed_args,
    run_inference,
    signal_handler,
    wait,
)

EVALUATOR_TEMP = 0.7
OPTIMIZER_TEMP = 1.4
OPTIMIZER_ALT_TEMP = 0.01
EVALUATOR_SEED = 727
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
ARGS = get_parsed_args()


OPTIMIZER_SYSTEM_PROMPT = f"""{
    "<role>You are a professional linguistic translator.</role>" * (not ARGS.omit_roles)
}
<goal>Your primary directive is to provide a fluent, accurate, and contextually appropriate translation.</goal>.
<requirements>
  <item>Pay attention to any markedness, whether syntactical or lexical, that should be considered in the translation.</item>
  <item>Preserve the tone, style, and intent of the source text.</item>
  <item>Ensure the translation reads naturally in the target language.</item>
  <item>Adhere to the context provided, as it may contain important information that affects your translation choices.</item>
</requirements>
<output_format>
  <item>First, parahrase the source text in your own words to demonstrate your understanding of its meaning in the paraphrase tag.</item>
  <item>Then, provide the translation in the translation tag.</item>
</output_format>
<example>
  <input>This is an example source text that needs to be translated.</input>
  <output>
    <paraphrase>This is a paraphrase of the input.</paraphrase>
    <translation>Ini adalah contoh terjemahan dari input.</translation>
  </output>
</example>
""".strip()


OPTIMIZER_INIT_USER_PROMPT = """{CONTEXT}
<source_text>{SOURCE_TEXT}</source_text>
<task>Provide only the translation following the required output format.</task>
""".strip()


EVALUATOR_SYSTEM_PROMPT = f"""
{"<role>You are a meticulous and highly critical linguistic editor tasked with evaluating translations.</role>" * (not ARGS.omit_roles)}
<goal>Your goal is to ensure that every translation meets the highest standards of quality.<goal>
<requirements>
  <item>The translation must read naturally in the target language.</item>
  <item>It must accurately convey the meaning of the source text.</item>
  <item>Pay close attention to tone, style, any cultural nuances, and markedness of the source text.</item>
</requirements>
<output_format>
  Your response must include the following sections in order:
  <item>Analysis of the source text in terms of its meaning, tone, style, and any particular challenges or nuances in the analysis tag.</item>
  <item>Evaluation of the translation attempt under the evaluation tag, identifying specific issues, errors, or awkward phrasings, along with multiple alternatives or suggestions for each.</item>
  <item>The grade tag must contain 'fail' if you have any suggestions (even minor ones). Respond with 'pass' only if it requires no changes whatsoever and you have nothing to add.</item>
</output_format>
<example>
    <analysis>The source text is a novel excerpt. One markedness observed is the characters are friends with each other, suggested by the use of informal language. This may or may not be translated using the same informality depending on how the target culture depicts closeness</analysis>
    <evaluation>The translation is generally accurate but has some awkward phrasings. For example, "X" could be better translated as "Y" to sound more natural. Additionally, "A" might be interpreted differently in the target culture; consider using "B" instead.</evaluation>
    <grade>fail</grade>
</example>
""".strip()


EVALUATOR_USER_PROMPT = f"""
{
    '''{CONTEXT}
<source_text>{SOURCE_TEXT}</source_text>
<attempt>{TRANSLATION_ATTEMPT}</attempt>
    '''.strip()
    * (not ARGS.preserve_history)
}
<task>Evaluate the content in the attempt tag based on the source text provided. Provide only the evaluation following the required output format.</task>
""".strip()


OPTIMIZER_RETRY_PROMPT = f"""
<context>A previous translation attempt was evaluated.</context>
{
    '''{CONTEXT}
<source_text>{SOURCE_TEXT}</source_text>
<feedback>{FEEDBACK}</feedback>
    '''.strip()
    * (not ARGS.preserve_history)
}
<task>{
    "Your task is to consider the editor's feedback and then revise the translation accordingly."
    if ARGS.preserve_history
    else "Your task is to deeply consider the editor's feedback and generate a completely new version of the translation that addresses the identified problems. Start again from scratch, keeping the feedback in mind."
} Follow the output format</task>
""".strip()


class Rubric(TypedDict, total=False):
    semantic_accuracy: bool
    tonal_fidelity: bool
    natural_fluency: bool
    nuance_preservation: bool


class TranslationAttempt(TypedDict, total=False):
    type: Literal["attempt"]
    translation: str
    raw_output: str
    prompt: str
    system_prompt: str
    seed: int
    temp: float


class TranslationFeedback(TypedDict, total=False):
    type: Literal["feedback"]
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
    external_knowledge: list[str]


class State(TypedDict):
    iteration_id: int
    source_text: SourceTextEntry
    next_state: str
    max_attempt: int
    attempt: int
    history: list[TranslationAttempt | TranslationFeedback]
    optimizer_seed: int
    evaluator_seed: int
    client: aiohttp.ClientSession
    csv_writer: "CSVWriter | None"


class CSVWriter(Protocol):
    def writerow(self, row: Iterable[Any]) -> Any: ...

    def writerows(self, rows: Iterable[Iterable[Any]]) -> None: ...


def get_last_feedback(state: State) -> TranslationFeedback | None:
    for entry in state["history"][::-1]:
        assert "type" in entry
        if entry["type"] == "feedback":
            return entry  # type: ignore
    return None


def fill_in_messages(state: State, messages: list[tuple[str, str, str]]) -> None:
    if not ARGS.preserve_history:
        return

    getter: dict[
        str, Callable[[TranslationAttempt | TranslationFeedback], tuple[str, str]]
    ] = {
        "attempt": lambda e: (e.get("translation", ""), "optimizer"),
        "feedback": lambda e: (e.get("feedback", ""), "evaluator"),
    }

    for entry in state["history"]:
        assert "type" in entry
        assert "prompt" in entry

        messages.append(("user", entry["prompt"], "user"))
        messages.append(("assistant",) + getter[entry["type"]](entry))


def build_messages(
    state: State, system_prompt: str, user_prompt: str
) -> list[tuple[str, str, str]]:
    messages = [("system", system_prompt, "system")]
    fill_in_messages(state, messages)
    messages.append(("user", user_prompt, "user"))
    return messages


def format_context(state: State) -> str:
    nl = "\n"
    return f"""
<context>
  <text_type>{state["source_text"]["type"]}</text_type>
  <source_lang>{state["source_text"]["source_lang"]}</source_lang>
  <target_lang>{state["source_text"]["target_lang"]}</target_lang>
  <external_knowledge>
{nl.join([f"    <item>{i}</item>" for i in state["source_text"].get("external_knowledge", [])])}
  </external_knowledge>
</context>
""".strip()


async def handle_optimization_state(state: State) -> None:
    state["attempt"] += 1
    state["next_state"] = "evaluation"

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

    context = format_context(state)
    system_prompt = OPTIMIZER_SYSTEM_PROMPT

    if is_draft:
        prompt = OPTIMIZER_INIT_USER_PROMPT.format(
            SOURCE_TEXT=state["source_text"]["text"],
            CONTEXT=context,
        )
    else:
        # walk backwards to find the last attempt with feedback
        last_feedback = get_last_feedback(state)

        # THIS SHOULD NEVER HAPPEN
        if not last_feedback:
            LOGGER.error(
                "No feedback found from previous attempts for text %d, iteration %d/%d, attempt %d/%d. Cannot proceed with refinement.",
                state["source_text"]["id"],
                state["iteration_id"],
                ARGS.iterations,
                state["attempt"],
                state["max_attempt"],
            )
            state["next_state"] = ""
            return

        prompt = OPTIMIZER_RETRY_PROMPT.format(
            SOURCE_TEXT=state["source_text"]["text"],
            FEEDBACK=last_feedback.get("feedback", "Not available."),
            CONTEXT=context,
        )

    temp = OPTIMIZER_TEMP if is_draft else OPTIMIZER_ALT_TEMP
    seed = state["optimizer_seed"] * 10 + state["attempt"]
    messages = build_messages(state, system_prompt, prompt)
    output = (
        await run_inference(
            state["client"],
            ARGS.endpoint,
            ARGS.model,
            temp,
            seed,
            timeout=ARGS.timeout,
            cache_prompt=ARGS.cache_prompt,
            messages=messages,
        )
    ).strip()

    translation = (
        output
        if not (
            match := re.search(r"<translation>(.*?)</translation>", output, re.DOTALL)
        )
        else match.group(1).strip()
    )
    state["history"].append(
        {
            "type": "attempt",
            "translation": translation,
            "raw_output": output,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "seed": seed,
            "temp": temp,
        }
    )


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
    last_attempt = state["history"][-1]
    assert last_attempt.get("type") == "attempt"

    system_prompt = EVALUATOR_SYSTEM_PROMPT
    prompt = EVALUATOR_USER_PROMPT.format(
        SOURCE_TEXT=state["source_text"]["text"],
        TRANSLATION_ATTEMPT=last_attempt.get("translation", "Not available."),
        CONTEXT=format_context(state),
    )
    messages = build_messages(state, system_prompt, prompt)
    output = (
        await run_inference(
            state["client"],
            ARGS.endpoint,
            ARGS.model,
            EVALUATOR_TEMP,
            seed,
            timeout=ARGS.timeout,
            cache_prompt=ARGS.cache_prompt,
            messages=messages,
        )
    ).strip()

    feedback: TranslationFeedback = {
        "type": "feedback",
        "prompt": prompt,
        "system_prompt": system_prompt,
        "seed": seed,
        "temp": EVALUATOR_TEMP,
        "grade": "fail"
        if not (match := re.search(r"<grade>(.*?)</grade>", output, re.DOTALL))
        else match.group(1).strip().lower(),
        "feedback": output,
    }
    state["history"].append(feedback)

    if csv_writer := state.get("csv_writer"):
        csv_writer.writerow(
            (
                state["source_text"]["id"],
                state["iteration_id"],
                state["attempt"],
                last_attempt.get("seed", -1),
                last_attempt.get("temp", -1),
                seed,
                EVALUATOR_TEMP,
                state["source_text"]["text"],
                last_attempt.get("translation", "Not available."),
                "evaluator",
                "\n".join(f"{k}: {v}" for k, v in feedback.get("rubric", {}).items())
                or "N/A",
                feedback.get("grade", "N/A"),
                feedback.get("feedback", "Not available."),
                time.ctime(),
                last_attempt.get("system_prompt", "Not available."),
                last_attempt.get("prompt", "Not available."),
                system_prompt,
                prompt,
            )
        )

    state["next_state"] = "optimization"
    if "pass" in feedback["grade"] or state["attempt"] >= state["max_attempt"]:
        state["next_state"] = ""


class FileProcessor:
    CSV_HEADER: tuple[str, ...] = (
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
    )

    STATE_HANDLERS = {
        "optimization": handle_optimization_state,
        "evaluation": handle_evaluation_state,
    }

    def __init__(
        self,
        id: int,
        input_file: Path,
        output_file: Path,
        client: aiohttp.ClientSession,
    ) -> None:
        self.id = id
        self.input_file = input_file
        self.output_file = output_file

        self.csv_file: TextIOWrapper | None = None
        self.csv_writer: CSVWriter | None = None
        self.log_file: TextIOWrapper | None = None

        self.client = client

    def open(self) -> None:
        if not ARGS.save_output:
            return

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_file:
            self.csv_file = open(self.output_file, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(self.CSV_HEADER)
            LOGGER.info("Output will be saved to: %s", self.output_file)

        if not self.log_file:
            self.log_file = open(
                self.output_file.with_suffix(".jsonl"), "w", encoding="utf-8"
            )

    def __del__(self) -> None:
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.csv_writer = None

    async def process(self) -> None:
        if not self.input_file.exists():
            LOGGER.error("Input file '%s' does not exist.", self.input_file)
            return

        LOGGER.info("Processing input file: %s", self.input_file)

        self.open()

        input_json = json.loads(self.input_file.read_text("utf-8").strip())

        for text_idx, text in enumerate(input_json["texts"]):
            LOGGER.info(
                "--- Translating text %d out of %d ---",
                text_idx + 1,
                len(input_json["texts"]),
            )

            source_text: SourceTextEntry = {
                "source_lang": input_json["source_lang"],
                "target_lang": input_json["target_lang"],
                "text": text["content"],
                "type": input_json.get("type", "general"),
                "id": text_idx + 1,
                "external_knowledge": input_json.get("external_knowledge", [])
                + text.get("external_knowledge", []),
            }

            await self._process_text(source_text)

    async def _process_text(self, source_text: SourceTextEntry) -> None:
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
                history=[],
                optimizer_seed=SEEDS[i],
                evaluator_seed=EVALUATOR_SEED,
                client=self.client,
                csv_writer=self.csv_writer,
            )

            while handler := self.STATE_HANDLERS.get(state["next_state"]):
                await handler(state)
                _ = self.csv_file and self.csv_file.flush()

                # let llama-server disconnect the previous connection
                await asyncio.sleep(0.1)

            if self.log_file:
                self.log_file.write(
                    json.dumps(state["history"], ensure_ascii=False, indent=4) + "\n"
                )
                self.log_file.flush()


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

    event = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_args: signal_handler(event))  # type: ignore

    async with aiohttp.ClientSession() as client:
        for file_idx, (input_file, output_file) in enumerate(
            zip(input_files, output_files)
        ):
            try:
                processor = FileProcessor(
                    id=file_idx,
                    input_file=input_file,
                    output_file=output_file,
                    client=client,
                )

                await wait(
                    processor.process(),
                    event,
                )

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
