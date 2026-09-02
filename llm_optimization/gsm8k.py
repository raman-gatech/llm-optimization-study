"""Shared GSM8K prompting and numeric-answer normalization.

Keeping these functions in one dependency-light module prevents training,
teacher-trace generation, and evaluation from silently using different prompts
or scoring rules.
"""

from __future__ import annotations

import math
import re

_NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_FRACTION = rf"(?P<numerator>{_NUMBER})\s*/\s*(?P<denominator>{_NUMBER})"
_NUMBER_RE = re.compile(_NUMBER)
_FRACTION_RE = re.compile(_FRACTION)
_MARKER_RE = re.compile(r"####\s*(?P<answer>[^\n\r]+)")
_BOXED_RE = re.compile(r"\\boxed\{(?P<answer>[^{}]+)\}")


def format_reasoning_prompt(question: str) -> str:
    """Return the canonical chain-of-thought prompt used by HF workflows."""

    return f"{question.strip()}\n\nLet's think step by step.\n"


def _parse_candidate(candidate: str, *, prefer_last: bool = True) -> float | None:
    candidate = candidate.replace("$", "").replace("%", "").strip()
    fraction_matches = list(_FRACTION_RE.finditer(candidate))
    if fraction_matches:
        match = fraction_matches[-1 if prefer_last else 0]
        numerator = float(match.group("numerator").replace(",", ""))
        denominator = float(match.group("denominator").replace(",", ""))
        if denominator == 0:
            return None
        value = numerator / denominator
        return value if math.isfinite(value) else None

    number_matches = list(_NUMBER_RE.finditer(candidate))
    if not number_matches:
        return None
    match = number_matches[-1 if prefer_last else 0]
    value = float(match.group(0).replace(",", ""))
    return value if math.isfinite(value) else None


def extract_numeric_answer(text: str) -> float | None:
    """Extract a final numeric answer from GSM8K-style text.

    Priority is given to the GSM8K ``####`` marker, then a final ``\boxed{}``
    expression, and finally the last numeric expression in the text. Commas,
    decimals, scientific notation, currency symbols, percentages, and simple
    fractions are normalized.
    """

    if not text:
        return None

    marker_matches = list(_MARKER_RE.finditer(text))
    if marker_matches:
        value = _parse_candidate(
            marker_matches[-1].group("answer"),
            prefer_last=False,
        )
        if value is not None:
            return value

    boxed_matches = list(_BOXED_RE.finditer(text))
    if boxed_matches:
        value = _parse_candidate(
            boxed_matches[-1].group("answer"),
            prefer_last=False,
        )
        if value is not None:
            return value

    return _parse_candidate(text)


def numeric_answers_match(
    prediction: float | None,
    reference: float | None,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Compare normalized numeric answers with an absolute tolerance."""

    if prediction is None or reference is None:
        return False
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    return math.isclose(prediction, reference, rel_tol=0.0, abs_tol=tolerance)
