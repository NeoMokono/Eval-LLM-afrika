"""Versioned prompt templates. The template id and a hash of its text (plus its answer
options) are part of every fingerprint, so editing a template reruns exactly the benchmarks
that use it.

Where lm-evaluation-harness ships a task for a benchmark, the wording and answer options are
ported from it at commit f4d4b3de (the commit the top40-gold benchmark blocklist was built
against); those ids end in `@lm-eval-f4d4b3d`. Translation and slot filling have no usable
lm-eval task (its FLORES task targets facebook/flores with a completion-style prompt), so they
use instruction-style prompts suited to chat models.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from eval.registry import BenchmarkSpec

SIB_CATEGORIES = (
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
)

INJONGO_INTENTS = (
    "alarm", "balance", "bill_balance", "book_flight", "book_hotel", "calendar_update",
    "cancel_reservation", "car_rental", "confirm_reservation", "cook_time", "exchange_rate",
    "food_last", "freeze_account", "ingredients_list", "interest_rate", "international_visa",
    "make_call", "meal_suggestion", "min_payment", "pay_bill", "pin_change", "play_music",
    "plug_type", "recipe", "restaurant_reservation", "restaurant_reviews",
    "restaurant_suggestion", "share_location", "shopping_list_update", "spending_history",
    "text", "time", "timezone", "transactions", "transfer", "translate", "travel_notification",
    "travel_suggestion", "update_playlist", "weather",
)  # fmt: skip

INJONGO_SLOTS = (
    "ACCOUNT_TYPE", "ARTIST_NAME", "BANK_NAME", "BILL_TYPE", "CALENDAR_EVENT",
    "CITY_OR_PROVINCE", "COUNTRY", "CURRENCY", "DATE", "DISH_OR_FOOD", "HOTEL_NAME",
    "LANGUAGE_NAME", "MEAL_PERIOD", "MONEY", "MUSIC_GENRE", "NUMBER", "PERSONAL_NAME",
    "PLACE_NAME", "RESTAURANT_NAME", "SHOPPING_ITEM", "SONG_NAME", "TIME",
)  # fmt: skip

LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    text: str
    options: tuple[str, ...] = ()  # fixed answer options scored for this prompt, if any

    @property
    def sha256(self) -> str:
        payload = "\x00".join([self.text, *self.options])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def format(self, **fields: object) -> str:
        return self.text.format(**fields)


_MMLU = PromptTemplate(
    id="mmlu-letters@lm-eval-f4d4b3d",
    text="{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:",
    options=LETTERS,
)
_TRANSLATION = PromptTemplate(
    id="translate-instruction-v1",
    text=(
        "Translate the following text from English to {language}. "
        "Provide only the translation.\n\n{text}"
    ),
)

TEMPLATES: dict[str, PromptTemplate] = {
    "afrixnli": PromptTemplate(
        id="afrixnli-en-direct@lm-eval-f4d4b3d",
        text="{premise}\nQuestion: {hypothesis} True, False, or Neither?\nAnswer:",
        options=("True", "Neither", "False"),  # label 0 entailment, 1 neutral, 2 contradiction
    ),
    # lm-eval's afrimmlu prompt scores the full answer texts; the MMLU letter format is used
    # instead so AfriMMLU and English MMLU are prompted identically.
    "afrimmlu": _MMLU,
    "english_mmlu": _MMLU,
    "afrimgsm": PromptTemplate(
        id="afrimgsm-direct-prompt_1@lm-eval-f4d4b3d",
        text="Question: {question}\nAnswer:",
    ),
    "sib200": PromptTemplate(
        id="sib-prompt_1@lm-eval-f4d4b3d",
        text=(
            "Given the categories science/technology, travel, politics, sports, health, "
            "entertainment, or geography; what category does the text: '{text}' belong to: \n\n"
        ),
        options=SIB_CATEGORIES,
    ),
    "belebele": PromptTemplate(
        id="belebele-prompt_1@lm-eval-f4d4b3d",
        text=(
            "P: {passage}\nQ: {question}\nA: {a}\nB: {b}\nC: {c}\nD: {d}\n"
            "Please choose the correct answer from the options above:"
        ),
        options=LETTERS,
    ),
    "injongo": PromptTemplate(
        id="injongointent-prompt_1@lm-eval-f4d4b3d",
        text=(
            "Given the text: '{text}', determine the correct intent from the following list: ["
            + ", ".join(INJONGO_INTENTS)
            + "]. Only output one intent from the list."
        ),
        options=INJONGO_INTENTS,
    ),
    "injongo_slots": PromptTemplate(
        id="injongo-slots-v1",
        text=(
            "Extract the slot values from the text below. The possible slot labels are: "
            + ", ".join(INJONGO_SLOTS)
            + ". Ensure the output strictly follows the format: LABEL: value $$ LABEL: value, "
            "copying each value exactly as it appears in the text. If the text contains no "
            "slots, output: none. Return only the output.\n\nText: {text}"
        ),
    ),
    "flores_plus": _TRANSLATION,
    "afriscience_mt": _TRANSLATION,
    # Perplexity scores raw text; the template is a marker and is not part of its fingerprint.
    "english_perplexity": PromptTemplate(id="raw-text", text="{text}"),
}


def template_for(benchmark: BenchmarkSpec) -> PromptTemplate:
    try:
        return TEMPLATES[benchmark.id]
    except KeyError:
        raise KeyError(
            f"no prompt template for benchmark {benchmark.id!r}; add one to eval/prompts"
        ) from None
