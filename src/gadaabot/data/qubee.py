"""Qubee (Afaan Oromoo Latin script) text normalization and language heuristics.

Design notes
------------
* Orthography. Qubee marks vowel length and gemination by doubling letters and the
  glottal stop (hudhaa) with an apostrophe. Web text writes the hudhaa with at least
  six different code points (U+0027, U+2019, U+2018, U+02BC, U+0060, U+00B4, ...).
  Subword tokenizers treat each variant as a different symbol, so the same word
  fragments differently depending on keyboard. We canonicalize all of them to
  U+0027 (ASCII apostrophe), the form most common in Oromo web text.
* Script. Oromo is occasionally written in Ge'ez (Ethiopic) script; such documents are
  counted and excluded from the Qubee corpus rather than transliterated.
* Language heuristic. Upstream corpora already ran language ID (FineWeb-2 uses GlotLID).
  We add a cheap function-word check as a second line of defense against English and
  Somali leakage (Somali shares Latin script and double vowels with Oromo).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

HUDHAA_VARIANTS: tuple[str, ...] = (
    "’",  # right single quotation mark
    "‘",  # left single quotation mark
    "ʼ",  # modifier letter apostrophe
    "ʻ",  # modifier letter turned comma
    "`",  # grave accent
    "´",  # acute accent
    "′",  # prime
    "＇",  # fullwidth apostrophe
)
_HUDHAA_TABLE = str.maketrans({c: "'" for c in HUDHAA_VARIANTS})
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t  -   　]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)

OROMO_FUNCTION_WORDS: frozenset[str] = frozenset(
    "fi kan akka irratti keessatti ni isaa kana sana jedhe ta'e ture jira garuu yookaan "
    "yeroo waan hin isaanii kanaaf waliin gara kun ammoo immoo akkasumas itti irraa dha "
    "ti kanneen isaan nu nuti ani ati inni ishee kanarratti booda dura osoo".split()
)
SOMALI_FUNCTION_WORDS: frozenset[str] = frozenset(
    "iyo oo ka ku waa ayaa ee la u ah aan uu ay soo kale sida kuwa waxaa".split()
)
ENGLISH_FUNCTION_WORDS: frozenset[str] = frozenset(
    "the and of to in is that for with as was on are by this be from".split()
)


def is_ethiopic(ch: str) -> bool:
    """Return True for characters in the Ethiopic Unicode blocks."""
    cp = ord(ch)
    return 0x1200 <= cp <= 0x139F or 0x2D80 <= cp <= 0x2DDF or 0xAB00 <= cp <= 0xAB2F


def normalize_qubee(text: str) -> str:
    """Canonicalize Qubee text without changing its linguistic content.

    Steps: NFC composition, hudhaa unification to U+0027, removal of zero-width and
    control characters, whitespace collapsing (newlines kept, at most two in a row).
    """
    t = unicodedata.normalize("NFC", text)
    t = t.translate(_HUDHAA_TABLE)
    t = _ZERO_WIDTH.sub("", t)
    t = _CONTROL.sub("", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(_SPACES.sub(" ", line).strip() for line in t.split("\n"))
    t = _MANY_NEWLINES.sub("\n\n", t)
    return t.strip()


def words(text: str) -> list[str]:
    """Lower-cased word tokens; hudhaa stays inside the word (ta'e, mirkanaa'e)."""
    return [w.lower() for w in _WORD.findall(text)]


@dataclass(frozen=True)
class TextProfile:
    """Cheap per-document statistics used for filtering and reporting."""

    n_chars: int
    n_words: int
    alpha_ratio: float
    ethiopic_ratio: float
    oromo_fw: float
    somali_fw: float
    english_fw: float

    @property
    def looks_oromo(self) -> bool:
        return self.oromo_fw > max(self.somali_fw, self.english_fw)


def profile(text: str) -> TextProfile:
    """Compute a TextProfile for normalized text."""
    n = len(text)
    letters = sum(ch.isalpha() for ch in text)
    eth = sum(is_ethiopic(ch) for ch in text)
    ws = words(text)
    nw = len(ws) or 1
    return TextProfile(
        n_chars=n,
        n_words=len(ws),
        alpha_ratio=letters / n if n else 0.0,
        ethiopic_ratio=eth / letters if letters else 0.0,
        oromo_fw=sum(w in OROMO_FUNCTION_WORDS for w in ws) / nw,
        somali_fw=sum(w in SOMALI_FUNCTION_WORDS for w in ws) / nw,
        english_fw=sum(w in ENGLISH_FUNCTION_WORDS for w in ws) / nw,
    )


def count_hudhaa_variants(text: str) -> dict[str, int]:
    """Count each apostrophe-like code point in raw text (for the data statement)."""
    out: dict[str, int] = {}
    for ch in ("'",) + HUDHAA_VARIANTS:
        c = text.count(ch)
        if c:
            out[f"U+{ord(ch):04X}"] = c
    return out


# ---------------------------------------------------------------------------
# v3 filter helpers (D26: line-level script cleaning, Qubee alphabet rule)
# ---------------------------------------------------------------------------
def has_non_ascii_letter(word: str) -> bool:
    """True if a word contains a letter outside ASCII. Qubee needs only A-Z plus
    the hudhaa (already canonical U+0027), so such words are foreign or Ge'ez."""
    return any(ord(ch) > 127 and ch.isalpha() for ch in word)


def non_qubee_word_ratio(ws: list[str]) -> float:
    """Fraction of words that contain a non-ASCII letter."""
    return sum(has_non_ascii_letter(w) for w in ws) / len(ws) if ws else 0.0


def line_is_foreign_script(line: str, max_ethiopic: float = 0.3, max_non_qubee: float = 0.5) -> bool:
    """Line-level script test: Ge'ez-dominant or mostly non-Qubee-alphabet words."""
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False
    if sum(is_ethiopic(ch) for ch in letters) / len(letters) > max_ethiopic:
        return True
    return non_qubee_word_ratio(words(line)) > max_non_qubee


def function_word_counts(ws: list[str]) -> tuple[int, int, int]:
    """Counts of (Oromo, Somali, English) function words in a word list."""
    return (sum(w in OROMO_FUNCTION_WORDS for w in ws),
            sum(w in SOMALI_FUNCTION_WORDS for w in ws),
            sum(w in ENGLISH_FUNCTION_WORDS for w in ws))
