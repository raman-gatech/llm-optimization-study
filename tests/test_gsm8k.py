from __future__ import annotations

import pytest

from llm_optimization.gsm8k import (
    extract_numeric_answer,
    format_reasoning_prompt,
    numeric_answers_match,
)


def test_prompt_uses_real_newlines() -> None:
    prompt = format_reasoning_prompt("  What is 2 + 2?  ")
    assert prompt == "What is 2 + 2?\n\nLet's think step by step.\n"
    assert "\\n" not in prompt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work #### 1,234", 1234.0),
        ("answer #### -12.5", -12.5),
        ("answer #### 1.2e3", 1200.0),
        (r"therefore \boxed{42}", 42.0),
        ("the ratio is 3/4", 0.75),
        ("first 10, finally 22", 22.0),
        ("cost #### $2,500.50", 2500.5),
    ],
)
def test_extract_numeric_answer(text: str, expected: float) -> None:
    assert extract_numeric_answer(text) == pytest.approx(expected)


def test_marker_takes_priority() -> None:
    assert extract_numeric_answer("reasoning 100 #### 42 trailing 99") == 42.0
    assert extract_numeric_answer("reasoning 100 #### 42\ntrailing 99") == 42.0


def test_missing_and_invalid_values() -> None:
    assert extract_numeric_answer("") is None
    assert extract_numeric_answer("no numeric answer") is None
    assert extract_numeric_answer("division 1/0") is None


def test_numeric_match() -> None:
    assert numeric_answers_match(1.0, 1.0 + 1e-7)
    assert not numeric_answers_match(1.0, 1.1)
    assert not numeric_answers_match(None, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        numeric_answers_match(1.0, 1.0, tolerance=-1)
