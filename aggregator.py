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
import logging
import re
from pathlib import Path

import aiohttp
import pandas as pd

from lib import run_inference

AGGREGATOR_TEMP = 0.1
ENDPOINT = "http://localhost:8000/v1/chat/completions"
TIMEOUT = 0

AGGREGATOR_SYSTEM_PROMPT = """Your task is to synthesize a single, definitive, publishable translation from a set of candidates.

--- REQUIREMENTS ---
1. The synthesized translation must accurately convey the meaning of the source text. It does not mean a word-for-word translation, but the overall message, tone, and intent must be preserved. Any nuances, idioms, or cultural references in the source text must be appropriately adapted to the target language.
2. The synthesized translation must read naturally and fluently in the target language. It should feel like it was originally written in that language, without awkward phrasing or unnatural constructions.
3. The synthesized translation must maintain an appropriate tone and style for the content. If the source text is formal, the translation should be formal; if it's casual, the translation should reflect that.
4. If it is a novel excerpt, maintain the literary style and voice of the original author, and each character's unique manner of speaking.
5. Do not mistake a common occurrence of a phrase in the candidates as a sign of correctness. There may be multiple candidates with the same error. Likewise, only one candidate may have the correct phrasing.

--- OUTPUT FORMAT ---
1. Under the `--- SOURCE TEXT ANALYSIS ---` header, you must provide a detailed anaylsis of the source text in terms of its meaning, tone, and style.
2. Then, under the `--- CANDIDATE REVIEW ---` header, review each candidate translation one by one. For each candidate, highlight its strengths and weaknesses, noting any errors or particularly well-translated phrases.
3. Pick three best candidate(s) that you think are closest to the ideal translation, and explain why. List them under the `--- TOP CANDIDATES ---` header.
4. Then finally, under the `--- SYNTHESIZED TRANSLATION ---` header, produce a single, polished translation. You may combine elements from multiple top candidates as needed to create the best possible version. Remember not to mix personal pronouns or verb forms incorrectly, as different candidates may have used different tenses, perspectives, levels of formality, or styles.

The `--- SYNTHESIZED TRANSLATION ---` is a final section, that should contain only the final translation text, without any additional commentary or formatting. If you want to include notes or explanations, put them in the previous sections.
"""

AGGREGATOR_USER_PROMPT = """Review all the translation candidates below. They represent various attempts and may include errors. Your goal is to synthesize them into one final version that represents the absolute best of all options.

--- CONTEXT ---
Source Language: {SOURCE_LANG}
Target Language: {TARGET_LANG}

--- SOURCE TEXT ---
{SOURCE_TEXT}

{CANDIDATE_BLOCK}

--- OUTPUT ---
"""

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Translation Aggregator Experiment")
parser.add_argument(
    "--input",
    required=True,
    type=Path,
    help="Path to the input CSV file from the main experiment.",
)
parser.add_argument(
    "--output",
    default=None,
    help="Path to save the final synthesized translations.",
)
parser.add_argument(
    "--source-lang",
    default="English",
    help="Source language of the texts.",
)
parser.add_argument(
    "--target-lang",
    default="Indonesian",
    help="Target language for the synthesized translations.",
)
ARGS = parser.parse_args()
if ARGS.output is None:
    ARGS.output = (
        Path(ARGS.input).parent
        / "aggregated"
        / ("aggregated_" + ARGS.input.stem + ".csv")
    )
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)


async def main():
    LOGGER.info("Starting aggregator script...")

    try:
        df = pd.read_csv(ARGS.input)
    except FileNotFoundError:
        LOGGER.error(f"Input file not found: {ARGS.input}")
        return

    grouped_by_text = df.groupby("text_id")

    LOGGER.info(f"Found {len(grouped_by_text)} unique texts to process.")

    output_path = Path(ARGS.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as client:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ["text_id", "source_text", "synthesized_translation", "notes"]
            )

            for text_id, group in grouped_by_text:
                LOGGER.info(
                    f"Processing text_id: {text_id} with {len(group)} total attempts as candidates."
                )

                source_text = group["source_text"].iloc[0]
                candidates = group["translation_attempt"].tolist()

                candidate_block = ""
                for i, candidate_text in enumerate(set(candidates)):
                    candidate_block += f"--- CANDIDATE {i} ---\n{candidate_text}\n\n"

                user_prompt = AGGREGATOR_USER_PROMPT.format(
                    SOURCE_TEXT=source_text,
                    CANDIDATE_BLOCK=candidate_block.strip(),
                    SOURCE_LANG=ARGS.source_lang,
                    TARGET_LANG=ARGS.target_lang,
                )

                LOGGER.info(f"Generating synthesized version for text_id: {text_id}")
                synthesized_translation = await run_inference(
                    client,
                    ENDPOINT,
                    "",
                    AGGREGATOR_TEMP,
                    seed=int(text_id) * 337,
                    timeout=TIMEOUT,
                    messages=[
                        ("system", AGGREGATOR_SYSTEM_PROMPT),
                        ("user", user_prompt),
                    ],
                )
                if not (
                    match := re.search(
                        r"--- SYNTHESIZED TRANSLATION ---\s*:?", synthesized_translation
                    )
                ):
                    LOGGER.error(
                        f"Failed to find synthesized translation section for text_id: {text_id}"
                    )
                    notes = ""
                    synthesized_translation = "Parsing error"
                else:
                    notes = synthesized_translation[: match.start()].strip()
                    synthesized_translation = synthesized_translation[
                        match.end() :
                    ].strip()

                csv_writer.writerow(
                    [text_id, source_text, synthesized_translation, notes]
                )
                csvfile.flush()
                LOGGER.info(
                    f"Successfully synthesized translation for text_id: {text_id}"
                )
                await asyncio.sleep(1)

    LOGGER.info(f"Aggregation complete. Results saved to {ARGS.output}")


if __name__ == "__main__":
    asyncio.run(main())
