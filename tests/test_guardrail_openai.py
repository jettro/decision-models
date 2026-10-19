import pytest

from decision_models.samples.guardrail_openai import verdict


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.85, (True, 0.7)),
        (0.36, (False, 0.28)),
        (0.5, (True, 0.0)),
        (0.0, (False, 1.0)),
        (1.0, (True, 1.0)),
    ],
)
def test_verdict_matches_pydantic_ai(probability, expected):
    assert verdict(probability) == expected


def test_verdict_with_a_custom_threshold():
    assert verdict(0.85, threshold=0.9) == (False, pytest.approx(0.055556))
