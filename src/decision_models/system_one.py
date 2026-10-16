"""Workaround for float precision in System One rubric (score) answers.

Pydantic AI checks that a rubric answer's `score` is the weighted mean of its `probabilities`, allowing a rounding
margin based on the number of decimals it sees. Backends such as Ollama return full-precision floats, which shrinks
that margin to ~1e-16, so float noise in the last bit makes a valid answer fail. Rounding the answer to a fixed
number of decimals before Pydantic AI reads it gives the check a realistic margin again.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider

DEFAULT_DECIMALS = 6


def round_score_answers(body: dict[str, Any], decimals: int = DEFAULT_DECIMALS) -> dict[str, Any]:
    """Round the `score` and `probabilities` of every score answer in a `/v1/systemone` response body."""
    for answer in (body.get("answers") or {}).values():
        if not isinstance(answer, dict) or answer.get("type") != "score":
            continue
        if isinstance(answer.get("score"), (int, float)):
            answer["score"] = round(answer["score"], decimals)
        probabilities = answer.get("probabilities")
        if isinstance(probabilities, dict):
            answer["probabilities"] = {
                level: round(p, decimals) if isinstance(p, (int, float)) else p for level, p in probabilities.items()
            }
    return body


class RoundingTransport(httpx2.AsyncBaseTransport):
    """Transport that rounds score answers in successful `/systemone` responses."""

    def __init__(self, decimals: int = DEFAULT_DECIMALS, transport: httpx2.AsyncBaseTransport | None = None):
        self._decimals = decimals
        self._transport = transport or httpx2.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await self._transport.handle_async_request(request)
        if not request.url.path.endswith("/systemone") or response.status_code >= 400:
            return response

        content = await response.aread()
        await response.aclose()
        try:
            body = json.loads(content)
        except ValueError:
            body = None
        if isinstance(body, dict):
            content = json.dumps(round_score_answers(body, self._decimals)).encode()

        # The content is already decoded, so drop headers that describe the original encoding and length.
        headers = [
            (name, value)
            for name, value in response.headers.multi_items()
            if name.lower() not in ("content-encoding", "content-length", "transfer-encoding")
        ]
        return httpx2.Response(response.status_code, headers=headers, content=content, request=request)

    async def aclose(self) -> None:
        await self._transport.aclose()


def build_system_one_model(
    model_name: str,
    *,
    base_url: str | None = None,
    decimals: int = DEFAULT_DECIMALS,
) -> SystemOneModel:
    """Build a `SystemOneModel` whose score answers are rounded to `decimals` before validation.

    `base_url` defaults to the `SYSTEM_ONE_BASE_URL` environment variable, like `SystemOneProvider`.
    """
    client = httpx2.AsyncClient(
        transport=RoundingTransport(decimals),
        timeout=httpx2.Timeout(600, connect=5),
    )
    provider = SystemOneProvider(base_url=base_url, http_client=client)
    return SystemOneModel(model_name, provider=provider)
