"""The single language-code normaliser.

Every language code that enters the suite (config files, dataset config names, training-data
`language` fields) goes through `normalize()`, so one language can never be counted under two
codes (`swa`/`swh`, `nso`/`nso_Latn`, `orm`/`gaz`, ...).

Canonical codes are ISO 639-3, matching the AfriGemma training data. Unknown codes raise
instead of passing through, so a typo or an unplanned language surfaces at config-load time.
To support a new language, add it to `LANGUAGE_NAMES` (and an alias if datasets spell it
differently).
"""

from __future__ import annotations

import re

LANGUAGE_NAMES: dict[str, str] = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "bam": "Bambara",
    "bbj": "Ghomálá'",
    "eng": "English",
    "ewe": "Ewe",
    "fon": "Fon",
    "fra": "French",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lin": "Lingala",
    "lug": "Luganda",
    "luo": "Dholuo",
    "mos": "Mossi",
    "nbl": "South Ndebele",
    "nso": "Sepedi",
    "nya": "Chichewa",
    "orm": "Oromo",
    "pcm": "Nigerian Pidgin",
    "plt": "Malagasy",
    "run": "Kirundi",
    "sna": "Shona",
    "som": "Somali",
    "sot": "Sesotho",
    "ssw": "Siswati",
    "swa": "Swahili",
    "tir": "Tigrinya",
    "tsn": "Setswana",
    "tso": "Xitsonga",
    "twi": "Twi",
    "ven": "Tshivenda",
    "wol": "Wolof",
    "xho": "isiXhosa",
    "yor": "Yoruba",
    "zul": "isiZulu",
}

SOUTH_AFRICAN: frozenset[str] = frozenset(
    {"afr", "nbl", "nso", "sot", "ssw", "tsn", "tso", "ven", "xho", "zul"}
)

# Alternative spellings -> canonical code. Individual-language codes fold into the macrolanguage
# code the training data uses. `aka` is deliberately *not* folded into `twi`: datasets such as
# SIB-200 ship both as separate subsets, and merging them would double-count Twi.
_ALIASES: dict[str, str] = {
    # ISO 639-3 individual / variant codes
    "swh": "swa",
    "gaz": "orm",
    "mlg": "plt",
    # ISO 639-1
    "af": "afr",
    "am": "amh",
    "bm": "bam",
    "ee": "ewe",
    "en": "eng",
    "fr": "fra",
    "ha": "hau",
    "ig": "ibo",
    "lg": "lug",
    "ln": "lin",
    "mg": "plt",
    "nr": "nbl",
    "ny": "nya",
    "om": "orm",
    "rn": "run",
    "rw": "kin",
    "sn": "sna",
    "so": "som",
    "ss": "ssw",
    "st": "sot",
    "sw": "swa",
    "ti": "tir",
    "tn": "tsn",
    "ts": "tso",
    "tw": "twi",
    "ve": "ven",
    "wo": "wol",
    "xh": "xho",
    "yo": "yor",
    "zu": "zul",
}

# `zul_Latn`, `zul-Latn`, `twi_Latn_akua1239` (FLORES+ glottocode-suffixed variants).
_TAGGED = re.compile(r"^([A-Za-z]{2,3})(?:[_-][A-Za-z]{4})?(?:[_-][a-z]{4}\d{4})?$")


class UnknownLanguageError(ValueError):
    pass


def normalize(code: str) -> str:
    """Return the canonical ISO 639-3 code for `code`, or raise UnknownLanguageError."""
    match = _TAGGED.match(code.strip())
    if match is None:
        raise UnknownLanguageError(f"unrecognised language code {code!r}")
    base = match.group(1).lower()
    canonical = _ALIASES.get(base, base)
    if canonical not in LANGUAGE_NAMES:
        raise UnknownLanguageError(
            f"language {code!r} (-> {canonical!r}) is not in eval/languages.py; "
            "add it there if it should be evaluated"
        )
    return canonical


def display_name(code: str) -> str:
    return LANGUAGE_NAMES[normalize(code)]
