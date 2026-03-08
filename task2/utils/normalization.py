# task2/utils/normalization.py
from __future__ import annotations

import re
from typing import Optional

# Canonical set for your Task2 classes (keep in sync with CNN class_names.json)
CANONICAL_ANIMALS = {
    "dog",
    "cat",
    "horse",
    "spider",
    "butterfly",
    "chicken",
    "sheep",
    "cow",
    "squirrel",
    "elephant",
}

# Common variations / typos / plurals / synonyms -> canonical label
# Add more as you see errors in your demo.
NORMALIZATION_MAP = {
    # plural forms
    "dogs": "dog",
    "cats": "cat",
    "horses": "horse",
    "spiders": "spider",
    "butterflies": "butterfly",
    "chickens": "chicken",
    "cows": "cow",
    "squirrels": "squirrel",
    "elephants": "elephant",

    # typos / variants
    "spyder": "spider",
    "squirel": "squirrel",
    "sqirrel": "squirrel",
    "buterfly": "butterfly",
    "butterflie": "butterfly",

    # simple synonyms (optional)
    "hen": "chicken",
    "rooster": "chicken",
    "calf": "cow",
    "kitten": "cat",
    "puppy": "dog",
}

# Simple tokens you might get from HuggingFace NER ("\u2581cow", "##cow", punctuation etc.)
_LEADING_SUBWORD_RE = re.compile(r"^(?:##|\\u2581)+")
_NON_LETTERS_RE = re.compile(r"[^a-z]+")


def clean_token(text: str) -> str:
    """
    Normalize raw token-ish strings:
    - strip spaces
    - remove HF subword markers (##, \u2581)
    - lowercase
    - keep only a-z letters
    """
    s = text.strip().lower()
    s = _LEADING_SUBWORD_RE.sub("", s)
    s = _NON_LETTERS_RE.sub("", s)
    return s


def naive_singularize(word: str) -> str:
    """
    Very lightweight singularization for English plurals.
    We keep this intentionally simple (no external deps).
    """
    if len(word) <= 3:
        return word

    # butterflies -> butterfly, puppies -> puppy
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # boxes -> box, classes -> class (approx)
    if word.endswith("es") and not word.endswith(("ses", "xes", "zes", "ches", "shes")):
        # e.g. "horses" -> "horse" actually works better with just removing 's',
        # so we don't overdo 'es' stripping. Keep conservative.
        pass

    # dogs -> dog, cats -> cat
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


def normalize_animal_name(name: Optional[str]) -> Optional[str]:
    """
    Convert input (possibly messy) animal mention to a canonical class label.

    Returns:
        canonical label (e.g. "cow") or None if cannot normalize to known animal.
    """
    if name is None:
        return None

    token = clean_token(name)
    if not token:
        return None

    # direct map
    token = NORMALIZATION_MAP.get(token, token)

    # try singular form if needed
    if token not in CANONICAL_ANIMALS:
        token2 = naive_singularize(token)
        token2 = NORMALIZATION_MAP.get(token2, token2)
        token = token2

    return token if token in CANONICAL_ANIMALS else None


def normalize_animal_list(names: list[str]) -> list[str]:
    """
    Normalize multiple extracted mentions, drop unknowns, keep unique in order.
    """
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        canon = normalize_animal_name(n)
        if canon and canon not in seen:
            out.append(canon)
            seen.add(canon)
    return out


def is_supported_animal(name: Optional[str]) -> bool:
    """True if name normalizes to one of the canonical animals."""
    return normalize_animal_name(name) is not None
