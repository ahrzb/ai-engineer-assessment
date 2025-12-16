"""
Extraction utilities for address-like strings.

This module unifies:
- Country extraction (via alias matching)
- Postal code extraction (simple heuristic)
- House number extraction (simple heuristic)

Primary API:
- extract_country(address) -> (rest, inferred_country)
- extract_postal_code(text) -> (rest, postal_code)
- extract_house_number(text) -> (rest, house_number)

All extractors remove the extracted part from the returned `rest` and try to keep
the rest readable by normalizing whitespace and trimming commas/spaces at ends.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple


# Country aliases dictionary with NFD normalization applied
COUNTRIES: Dict[str, List[str]] = {
    "Netherlands": ["Netherlands", "Holland", "NL", "The Netherlands"],
    "Germany": ["Germany", "Deutschland", "DE"],
    "France": ["France", "FR"],
    "United Kingdom": [
        "United Kingdom",
        "UK",
        "Great Britain",
        "GB",
        "England",
        "Scotland",
        "Wales",
        "Northern Ireland",
    ],
    "Spain": ["Spain", "España", "ES"],
    "Italy": ["Italy", "Italia", "IT"],
    "Belgium": ["Belgium", "België", "Belgique", "BE"],
    "Austria": ["Austria", "Österreich", "AT"],
    "Portugal": ["Portugal", "PT"],
    "Ireland": ["Ireland", "Éire", "IE", "Republic of Ireland"],
    "Hungary": ["Hungary", "Magyarország", "HU", "Hongarije"],
    "Switzerland": ["Switzerland", "Schweiz", "Suisse", "Svizzera", "CH"],
    "Sweden": ["Sweden", "Sverige", "SE"],
    "Norway": ["Norway", "Norge", "NO"],
}

# Normalize keys + aliases to NFD to make matching robust to accents
COUNTRIES = {
    unicodedata.normalize("NFD", k): [unicodedata.normalize("NFD", v) for v in aliases]
    for k, aliases in COUNTRIES.items()
}


def _strip_part(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = str(x).strip(" ,")
    return x or None


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,")


def _find_country_alias(address: str, countries_dict: Optional[Dict[str, List[str]]] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (inferred_country, matched_alias) using earliest alias position in the alias list.
    """
    if not address:
        return None, None
    if countries_dict is None:
        countries_dict = COUNTRIES

    hay = address.lower()
    best: Tuple[int, str, str] | None = None  # (alias_index, country, alias)

    for country, aliases in countries_dict.items():
        for idx, alias in enumerate(aliases):
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, hay):
                cand = (idx, country, alias)
                if best is None or cand[0] < best[0]:
                    best = cand
                break

    if best is None:
        return None, None
    _, inferred, matched_alias = best
    return inferred, matched_alias


def extract_country(address: str, countries_dict: Optional[Dict[str, List[str]]] = None) -> Tuple[str, Optional[str]]:
    """
    Extract country from an address string.

    Returns:
        (rest_of_address, inferred_country)
    """
    if not address:
        return "", None

    inferred, matched_alias = _find_country_alias(address, countries_dict)
    if not inferred or not matched_alias:
        return _normalize_spaces(address), None

    # remove matched alias (case-insensitive)
    pattern = r"\b" + re.escape(matched_alias) + r"\b"
    m = re.search(pattern, address, re.IGNORECASE)
    if not m:
        return _normalize_spaces(address), inferred

    before = address[: m.start()].rstrip()
    after = address[m.end() :].lstrip()
    before = re.sub(r"[,\s]+$", "", before)
    after = re.sub(r"^[,\s]+", "", after)
    rest = _normalize_spaces((before + " " + after).strip())
    return rest, _strip_part(inferred)


def extract_postal_code(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract a postal code token and remove it from text.

    Heuristic supports:
    - digits-only: 12345
    - digits + 2-letter suffix: 1234AB / 1234 AB
    - digits + 2-digit extension (only if separated): 12345-01 / 123456 01

    Rule: choose last occurrence.
    """
    if not text:
        return "", None

    # IMPORTANT: require a clear terminator after the token to avoid "1234 something" -> "1234so"
    suffix = r"(?:[\s-]?[A-Za-z]{2}|[\s-][0-9]{2})?"
    pattern = rf"\b\d{{4,}}{suffix}(?=(?:\s|,|$|[\.;:]))"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return _normalize_spaces(text), None
    m = matches[-1]
    token = _strip_part(m.group(0))
    rest = _normalize_spaces((text[: m.start()] + " " + text[m.end() :]).strip())
    return rest, token


def extract_house_number(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract a house number token and remove it from text.

    Heuristic:
    - 1-3 digits, optionally followed by a single letter (with or without space): 29, 29a, 29 a
    Rule: choose last occurrence.
    """
    if not text:
        return "", None

    matches = list(re.finditer(r"\b(\d{1,3})(?:\s*([A-Za-z]))?\b", text))
    if not matches:
        return _normalize_spaces(text), None
    m = matches[-1]
    num = m.group(1)
    letter = m.group(2)
    token = _strip_part(f"{num}{letter.lower()}" if letter else num)
    rest = _normalize_spaces((text[: m.start()] + " " + text[m.end() :]).strip())
    return rest, token


def parse_address_components(address: str) -> dict:
    """
    Convenience parser returning a dict with unified keys:
    {country, postal_code, house_number, rest}
    """
    rest, country = extract_country(address)
    rest, postal = extract_postal_code(rest)
    rest, house = extract_house_number(rest)
    return {"country": country, "postal_code": postal, "house_number": house, "rest": rest}
