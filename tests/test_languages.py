import pytest

from eval.languages import SOUTH_AFRICAN, UnknownLanguageError, display_name, normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("zul", "zul"),
        ("zul_Latn", "zul"),
        ("ZUL", "zul"),
        ("nso_Latn", "nso"),
        ("swh", "swa"),
        ("swh_Latn", "swa"),
        ("sw", "swa"),
        ("gaz_Latn", "orm"),
        ("amh_Ethi", "amh"),
        ("ary_Arab", "ary"),
        ("twi_Latn_akua1239", "twi"),
        ("twi_Latn_asan1239", "twi"),
        ("zu", "zul"),
        ("en", "eng"),
        ("xho-Latn", "xho"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_swahili_spellings_collapse_to_one_language() -> None:
    assert len({normalize(c) for c in ["swa", "swh", "sw", "swh_Latn"]}) == 1


@pytest.mark.parametrize("raw", ["", "unknown", "aka", "aka_Latn", "kik", "zzz", "zul_latn_x"])
def test_unknown_codes_raise(raw: str) -> None:
    with pytest.raises(UnknownLanguageError):
        normalize(raw)


def test_south_african_codes_are_canonical() -> None:
    assert all(normalize(c) == c for c in SOUTH_AFRICAN)


def test_display_name() -> None:
    assert display_name("swh_Latn") == "Swahili"
