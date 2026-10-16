from enum import IntEnum

from decision_models.priority import Priority, assess_priority, escalate, priority_for


class Urgency(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class Impact(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


def test_escalate_when_chance_of_higher_level_reaches_threshold():
    # Unsubscribe message: the score 1.49 rounds to MEDIUM, but HIGH has a 0.50 chance.
    probabilities = {"0": 0.02, "1": 0.48, "2": 0.50, "3": 0.00}
    assert escalate(Urgency.MEDIUM, probabilities, 0.3) is Urgency.HIGH


def test_escalate_keeps_level_when_higher_levels_are_unlikely():
    # Logout question: only 0.13 chance of HIGH or higher.
    probabilities = {"0": 0.57, "1": 0.29, "2": 0.13, "3": 0.00}
    assert escalate(Urgency.MEDIUM, probabilities, 0.3) is Urgency.MEDIUM


def test_escalate_adds_up_the_tail():
    # Neither HIGH nor CRITICAL reaches 0.3 alone, together they do: escalate to HIGH, not CRITICAL.
    probabilities = {"0": 0.50, "1": 0.20, "2": 0.18, "3": 0.12}
    assert escalate(Urgency.LOW, probabilities, 0.3) is Urgency.HIGH


def test_escalate_never_lowers_a_level():
    assert escalate(Urgency.HIGH, {"0": 1.0}, 0.3) is Urgency.HIGH
    assert escalate(Urgency.HIGH, None, 0.3) is Urgency.HIGH


def test_priority_for():
    assert priority_for(Urgency.CRITICAL, Impact.HIGH) is Priority.P1
    assert priority_for(Urgency.HIGH, Impact.MEDIUM) is Priority.P2
    assert priority_for(Urgency.MEDIUM, Impact.MEDIUM) is Priority.P3
    assert priority_for(Urgency.LOW, Impact.MEDIUM) is Priority.P4


def test_assess_priority_escalates_and_flags_low_confidence():
    details = {
        "confidence": {"urgency": 0.28, "impact": 0.88},
        "probabilities": {
            "urgency": {"0": 0.02, "1": 0.48, "2": 0.50, "3": 0.00},
            "impact": {"0": 0.00, "1": 0.03, "2": 0.97},
        },
    }
    assessment = assess_priority(Urgency.MEDIUM, Impact.HIGH, details)
    assert assessment.urgency is Urgency.HIGH
    assert assessment.impact is Impact.HIGH
    assert assessment.baseline is Priority.P2
    assert assessment.priority is Priority.P1
    assert assessment.needs_review
    assert len(assessment.reasons) == 2


def test_assess_priority_without_details_keeps_levels():
    assessment = assess_priority(Urgency.LOW, Impact.LOW, None)
    assert assessment.priority is assessment.baseline is Priority.P4
    assert not assessment.needs_review
    assert assessment.reasons == ()
