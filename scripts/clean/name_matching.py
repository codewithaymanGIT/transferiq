"""
Name/identity normalization -- the single biggest practical time sink when
joining FBref (accented names, "Bukayo Saka") against Transfermarkt/FPL,
per the blueprint's risk section. Centralized here so every cleaning script
uses the same rules instead of drifting.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_name(raw_name: str | None) -> str:
    """
    Produce a stable join key from a player name:
    - Unicode-normalize and strip accents (Öskarsson -> Oskarsson)
    - Collapse whitespace, strip punctuation used inconsistently across sources
    - Lowercase

    This is a JOIN KEY, not a display name -- never show `normalize_name`
    output to a user; keep the original `player_name` column for display.
    """
    if raw_name is None:
        return ""

    text = unicodedata.normalize("NFKD", raw_name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[.\-']", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def name_similarity(a: str, b: str) -> float:
    """
    Cheap, dependency-free similarity score (0..1) for flagging fuzzy-match
    candidates for manual review. Token-overlap based (handles "Bukayo Saka"
    vs "Saka" partial-name mismatches better than pure edit distance) --
    not a replacement for a real fuzzy-matching library (e.g. rapidfuzz) if
    higher precision is needed at scale, but adequate to flag review cases.
    """
    tokens_a = set(normalize_name(a).split())
    tokens_b = set(normalize_name(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
