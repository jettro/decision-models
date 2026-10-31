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

# A real Ollama `nimble` answer rejected before Pydantic AI 2.60.0 (PR #11092).
# Keep the original floats: rounding them would hide the regression.
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


def _run(handler=_ollama_response):
    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
            provider = SystemOneProvider(base_url="http://ollama.test/v1", http_client=client)
            agent = Agent(SystemOneModel("nimble", provider=provider), output_type=Classification, instructions="Classify.")
            return await agent.run("I keep being logged out.")

    return asyncio.run(run())


def test_raw_answer_passes_score_check_without_rounding():
    result = _run()
    assert result.output.urgency is Urgency.MEDIUM
    # The native validator preserves the backend's full precision in provider metadata.
    assert result.response.provider_details["scores"]["urgency"] == SCORE
    assert result.response.provider_details["probabilities"]["urgency"] == PROBABILITIES


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"score": SCORE + 0.001}, id="inconsistent-score"),
        pytest.param(
            {"probabilities": PROBABILITIES | {"1": PROBABILITIES["1"] + 0.0001}},
            id="probabilities-do-not-sum-to-one",
        ),
    ],
)
def test_inconsistent_raw_answer_still_fails_score_check(change):
    def handler(request):
        body = json.loads(_ollama_response(request).content)
        for answer in body["answers"].values():
            answer.update(change)
        return httpx2.Response(200, json=body)

    with pytest.raises(UnexpectedModelBehavior, match="does not match its question"):
        _run(handler)


@pytest.mark.parametrize("decimals", [2, 6])
def test_rounded_backend_answer_still_passes_score_check(decimals):
    def handler(request):
        body = json.loads(_ollama_response(request).content)
        for answer in body["answers"].values():
            answer["score"] = round(SCORE, decimals)
            answer["probabilities"] = {level: round(p, decimals) for level, p in PROBABILITIES.items()}
        return httpx2.Response(200, json=body)

    assert _run(handler).output.urgency is Urgency.MEDIUM
