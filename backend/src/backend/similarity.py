import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from backend import extractors


@runtime_checkable
class AddressSimilarity(Protocol):
    """Common interface for address similarity engines."""

    def score(self, a: str, b: str) -> float: ...

    def __call__(self, a: str, b: str) -> float: ...


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return " ".join(text.strip().lower().split())


@dataclass(frozen=True, slots=True)
class BaselineAddressSimilarity(AddressSimilarity):
    """Simple character-level baseline similarity."""

    def score(self, a: str, b: str) -> float:
        a_norm = _normalize(a)
        b_norm = _normalize(b)
        if not a_norm or not b_norm:
            return 0.0
        return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

    def __call__(self, a: str, b: str) -> float:
        return float(self.score(a, b))


class GLiNERAddressSimilarity(AddressSimilarity):
    """Hybrid similarity: extractors for structured bits + GLiNER for street/city."""

    def __init__(
        self,
        model_name: str = "fastino/gliner2-base-v1",
        street_weight: float = 0.35,
        city_weight: float = 0.25,
        postal_weight: float = 0.25,
        house_weight: float = 0.10,
        country_mismatch_cap: float = 0.25,
    ) -> None:
        import gliner2  # local import so baseline users don't need the heavy dependency at import-time

        self.model = gliner2.GLiNER2.from_pretrained(model_name)

        self.labels: Dict[str, Dict[str, str]] = {
            "city": {
                "description": "The city name, examples: Amsterdam, Rotterdam, The Hague",
                "dtype": "str",
            },
            "street_name": {
                "description": "The name of a street or a road of the address",
                "dtype": "str",
            },
        }

        self.street_weight = street_weight
        self.city_weight = city_weight
        self.postal_weight = postal_weight
        self.house_weight = house_weight
        self.country_mismatch_cap = country_mismatch_cap

    def _normalize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        return _normalize(str(text))

    def _extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        if not text:
            return {"street_name": None, "city": None}

        out: Any = self.model.extract_entities(text, self.labels)
        ents = out.get("entities", {}) if isinstance(out, dict) else {}

        def _get_first(label: str) -> Optional[str]:
            v = ents.get(label)
            if not v:
                return None
            return str(v).strip(" ,") or None

        return {
            "street_name": _get_first("street_name"),
            "city": _get_first("city"),
        }

    def _parse(self, address: str) -> Dict[str, Optional[str]]:
        base = extractors.parse_address_components(address)
        ents = self._extract_entities(base["rest"])
        base["street_name"] = ents.get("street_name")
        base["city"] = ents.get("city")
        return base

    def _sim(self, a: Optional[str], b: Optional[str]) -> float:
        na = self._normalize_text(a)
        nb = self._normalize_text(b)
        if not na and not nb:
            return 1.0
        if not na or not nb:
            return 0.0
        return difflib.SequenceMatcher(None, na, nb).ratio()

    def score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0

        pa = self._parse(a)
        pb = self._parse(b)

        country_a, country_b = pa.get("country"), pb.get("country")
        postal_a, postal_b = pa.get("postal_code"), pb.get("postal_code")
        house_a, house_b = pa.get("house_number"), pb.get("house_number")

        def _norm_postal(x: Optional[str]) -> str:
            return re.sub(r"[\s-]+", "", x).upper() if x else ""

        def _norm_house(x: Optional[str]) -> str:
            return re.sub(r"\s+", "", x).lower() if x else ""

        ca = self._normalize_text(country_a) or None
        cb = self._normalize_text(country_b) or None
        pa_n = _norm_postal(postal_a) or None
        pb_n = _norm_postal(postal_b) or None
        ha_n = _norm_house(house_a) or None
        hb_n = _norm_house(house_b) or None

        country_known = (ca is not None) and (cb is not None)
        postal_known = (pa_n is not None) and (pb_n is not None)
        house_known = (ha_n is not None) and (hb_n is not None)

        country_match = (ca == cb) if country_known else None
        postal_match = (pa_n == pb_n) if postal_known else None
        house_match = (ha_n == hb_n) if house_known else None

        street_sim = self._sim(pa.get("street_name"), pb.get("street_name"))
        city_sim = self._sim(pa.get("city"), pb.get("city"))
        rest_sim = self._sim(pa.get("rest"), pb.get("rest"))

        semantic_weight = self.street_weight + self.city_weight
        if semantic_weight > 0:
            semantic_sim = (
                self.street_weight * street_sim + self.city_weight * city_sim
            ) / semantic_weight
        else:
            semantic_sim = 0.0

        within_sim = max(semantic_sim, rest_sim)

        base = 0.0
        max_score = 1.0

        if country_known and country_match is False:
            max_score = min(max_score, self.country_mismatch_cap)

        if postal_known and postal_match is True:
            base = max(base, 0.85)
        elif postal_known and postal_match is False:
            max_score = min(max_score, 0.55)

        street_or_rest_strong = (street_sim > 0.80) or (rest_sim > 0.85)
        if house_known and house_match is True and street_or_rest_strong:
            base = max(base, 0.92)
        elif house_known and house_match is False and street_or_rest_strong:
            max_score = min(max_score, 0.85)

        if max_score < base:
            max_score = base

        score = base + (max_score - base) * within_sim
        return float(max(0.0, min(1.0, score)))

    def __call__(self, a: str, b: str) -> float:
        return float(self.score(a, b))


def build_default_similarity() -> AddressSimilarity:
    """
    Prefers GLiNER-backed similarity when available; falls back to baseline.
    """
    try:
        return GLiNERAddressSimilarity()
    except Exception:
        return BaselineAddressSimilarity()
