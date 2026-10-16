import asyncio
import json
from enum import IntEnum

import httpx2
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, UseEnumMemberDocstrings
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider

from decision_models.system_one import RoundingTransport, round_score_answers

# A real answer from Ollama `nimble` that Pydantic AI rejects because of float noise in the last bit.
PROBABILITIES = {"0": 0.0483716639315193, "1": 0.694482817577544, "2": 0.2560127781040187, "3": 0.0011327403869179724}
SCORE = 1.2099065949463355


class Urgency(UseEnumMemberDocstrings, IntEnum):
    LOW = 0
    """The message can wait."""
    MEDIUM = 1
    """The message should be addressed soon."""
    HIGH = 2
    """The message needs to be addressed immediately."""
    CRITICAL = 3
    """The message is a matter of life and death."""


class Classification(BaseModel):
    urgency: Urgency


def _ollama_response(request: httpx2.Request) -> httpx2.Response:
    names = json.loads(request.content)["questions"].keys()
    answer = {"type": "score", "score": SCORE, "confidence": 0.45, "probabilities": PROBABILITIES}
    body = {"model": "nimble", "answers": {name: answer for name in names}, "usage": {}}
    return httpx2.Response(200, json=body)


def _run(transport: httpx2.AsyncBaseTransport) -> Classification:
    provider = SystemOneProvider(base_url="http://ollama.test/v1", http_client=httpx2.AsyncClient(transport=transport))
    agent = Agent(SystemOneModel("nimble", provider=provider), output_type=Classification, instructions="Classify.")
    return asyncio.run(agent.run("I keep being logged out.")).output


def test_raw_answer_fails_score_check():
    with pytest.raises(UnexpectedModelBehavior, match="does not match its question"):
        _run(httpx2.MockTransport(_ollama_response))


def test_rounded_answer_passes_score_check():
    output = _run(RoundingTransport(transport=httpx2.MockTransport(_ollama_response)))
    assert output.urgency is Urgency.MEDIUM


def test_round_score_answers_leaves_other_answers_alone():
    body = {
        "answers": {
            "a": {"type": "score", "score": SCORE, "probabilities": dict(PROBABILITIES)},
            "b": {"type": "choice", "choice": "x", "probabilities": {"x": 0.123456789}},
        }
    }
    rounded = round_score_answers(body, decimals=6)
    assert rounded["answers"]["a"]["score"] == 1.209907
    assert rounded["answers"]["a"]["probabilities"]["1"] == 0.694483
    assert rounded["answers"]["b"]["probabilities"]["x"] == 0.123456789
