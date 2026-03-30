from dataclasses import dataclass
from typing import TypedDict


class Corpus(TypedDict):
    source_lang: str
    target_lang: str
    type: str
    external_knowledge: list[str]
    texts: list["TextEntry"]


class TextEntry(TypedDict):
    content: str
    external_knowledge: list[str]


class IdiomDefinitionEntry(TypedDict):
    senses: list[str]
    translations: dict[str, str]


class IdiomMatchResult(IdiomDefinitionEntry):
    idiom: str
    matched_chunk: str
    score: float


class Payload(TypedDict, total=False):
    model: str
    stream: bool
    temperature: float
    seed: int
    messages: list[dict[str, str]]
    cache_prompt: bool
    grammar: str


@dataclass
class CLIArgs:
    endpoint: str
    model: str
    iterations: int
    input: str
    timeout: int
    refinement_iterations: int
    cache_prompt: bool
    omit_roles: bool
    keep_n_messages: int
    save_output: bool
