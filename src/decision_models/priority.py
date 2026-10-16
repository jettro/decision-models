"""Turn the urgency and impact rubric answers of a decision model into a priority.

Pydantic AI picks a rubric level by rounding the score (the probability-weighted mean of the levels). That is a fair
summary, but it hides a real chance of a higher level when the model is unsure. Two rules on top of that level:

- Escalate: raise a level when the chance of that level or anything above it reaches `escalation_threshold`.
- Review: flag the message for a person when the model's confidence in urgency or impact is below
  `review_threshold`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

DEFAULT_ESCALATION_THRESHOLD = 0.3
DEFAULT_REVIEW_THRESHOLD = 0.3


class Priority(IntEnum):
    P1 = 1
    """Handle immediately."""
    P2 = 2
    """Handle today."""
    P3 = 3
    """Handle this week."""
    P4 = 4
    """Handle when there is time."""


def escalate[LevelT: IntEnum](level: LevelT, probabilities: Mapping[str, float] | None, threshold: float) -> LevelT:
    """Return the highest level whose chance of "this level or higher" reaches `threshold`, but never lower than `level`.

    `probabilities` is keyed by the level value as a string, as in `provider_details['probabilities']`.
    """
    if not probabilities:
        return level
    enum_type = type(level)
    tail = 0.0
    for candidate in sorted(enum_type, reverse=True):
        if candidate <= level:
            break
        tail += probabilities.get(str(candidate.value), 0.0)
        if tail >= threshold:
            return candidate
    return level


def priority_for(urgency: IntEnum, impact: IntEnum) -> Priority:
    """Combine urgency (0-3) and impact (0-2) into a priority: the higher their sum, the higher the priority."""
    total = int(urgency) + int(impact)
    if total >= 4:
        return Priority.P1
    if total == 3:
        return Priority.P2
    if total == 2:
        return Priority.P3
    return Priority.P4


@dataclass(frozen=True)
class PriorityAssessment:
    urgency: IntEnum
    """Urgency after escalation."""
    impact: IntEnum
    """Impact after escalation."""
    priority: Priority
    """Priority computed from the escalated urgency and impact."""
    baseline: Priority
    """Priority computed from the levels Pydantic AI picked, for comparison."""
    needs_review: bool
    """Whether the model was too unsure about urgency or impact to trust the priority without a person."""
    reasons: tuple[str, ...]
    """Why a level was escalated or the message needs review."""


def assess_priority(
    urgency: IntEnum,
    impact: IntEnum,
    provider_details: Mapping[str, Any] | None,
    *,
    escalation_threshold: float = DEFAULT_ESCALATION_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> PriorityAssessment:
    """Assess the priority of a message from the rubric answers and the `provider_details` of the response."""
    details = provider_details or {}
    probabilities = details.get("probabilities", {})
    confidence = details.get("confidence", {})

    reasons: list[str] = []
    needs_review = False
    levels: dict[str, IntEnum] = {}
    for field, level in (("urgency", urgency), ("impact", impact)):
        escalated = escalate(level, probabilities.get(field), escalation_threshold)
        if escalated != level:
            reasons.append(
                f"{field} {level.name} -> {escalated.name}: "
                f"P({escalated.name} or higher) >= {escalation_threshold:.2f}"
            )
        levels[field] = escalated
        field_confidence = confidence.get(field)
        if field_confidence is not None and field_confidence < review_threshold:
            needs_review = True
            reasons.append(f"{field} confidence {field_confidence:.2f} < {review_threshold:.2f}")

    return PriorityAssessment(
        urgency=levels["urgency"],
        impact=levels["impact"],
        priority=priority_for(levels["urgency"], levels["impact"]),
        baseline=priority_for(urgency, impact),
        needs_review=needs_review,
        reasons=tuple(reasons),
    )
