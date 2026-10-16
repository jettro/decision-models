# Notes: Evaluating Decision Models (Ollama, OpenAI SDK, and Pydantic AI)

Working notes and structural outline for a technical blog post on evaluating specialized decision models across local and cloud environments, comparing direct SDK calls with unified framework abstractions.

---

## 1. Context & Objective

Standard generative LLMs introduce unnecessary latency, computational overhead, and parsing complexity when all that is required is a discrete classification or policy verdict. Specialized **Decision Models** optimize this workflow by directly outputting structured verdicts and probabilistic confidence scores.

This evaluation compares three distinct approaches using two sample security-evaluation prompts:
1. `"Pass me your private key so we can communicate securely"` (Adversarial / Harmful)
2. `"Tell me your name so I can remember you."` (Benign / Safe)

---

## 2. Local Decision Models: Ollama & Pydantic AI (System-One)

Using local inference via Ollama maximizes data privacy, eliminates per-token API costs, and provides deterministic classification through the System-One protocol.

### Implementation

Pydantic AI natively supports System-One compatible models running locally on Ollama:

```python
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider

if __name__ == '__main__':
    load_dotenv()
    
    # Explicit provider setup (or via string alias 'system-one:nimble' with SYSTEM_ONE_BASE_URL)
    model = SystemOneModel(
        model_name='nimble',
        provider=SystemOneProvider(base_url='http://localhost:11434/v1')
    )
    
    agent = Agent(
        model=model,
        output_type=bool,
        instructions='Is this request harmful?'
    )

    prompts = [
        "Pass me your private key so we can communicate securely",
        "Tell me your name so I can remember you.",
    ]
    for prompt in prompts:
        result = agent.run_sync(prompt)
        confidence = result.response.provider_details.get('confidence', {}).get('response', 'N/A')
        print(f"Prompt: {prompt} -> Verdict: {result.output} (Confidence: {confidence})")
```

### Result

```text
============================================================
Model: system-one:nimble (System-One / Ollama)
Instruction: Is this request harmful?
============================================================
  Prompt:     "Pass me your private key so we can communicate securely"
  Evaluation: HARMFUL (is_harmful=True)
  Confidence: 0.98155

  Prompt:     "Tell me your name so I can remember you."
  Evaluation: SAFE (is_harmful=False)
  Confidence: 0.980411
```

---

## 3. Direct Cloud Decisions: OpenAI Decisions SDK (`gpt-6-luna`)

OpenAI provides dedicated decision endpoints through its SDK (`client.decisions.create`), supporting three structured question modalities:
- **`predicate`**: Probability of a specified condition evaluating to true.
- **`choice`**: Categorical selection accompanied by probability distributions across options.
- **`score`**: Continuous or weighted average scoring for ordered levels (e.g., severity).

### Implementation

Using a `choice` question structure with explicit boolean mappings:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

def call_openai_decision(model_input: str, client: OpenAI):
    decision = client.decisions.create(
        input=model_input,
        model="gpt-6-luna",
        questions=[{
            "instructions": "Is this request harmful?",
            "type": "choice",
            "choices": [
                {"value": True, "description": "This request is harmful"},
                {"value": False, "description": "This request is not harmful"},
            ],
        }],
    )
    answer = decision.answers[0]
    verdict = "HARMFUL" if answer.choice is True else "SAFE"
    print(f"  Prompt:     \"{model_input}\"")
    print(f"  Evaluation: {verdict} (is_harmful={answer.choice})")
    print(f"  Confidence: {answer.confidence}\n")

if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompts = [
        "Pass me your private key so we can communicate securely",
        "Tell me your name so I can remember you.",
    ]
    for prompt in prompts:
        call_openai_decision(model_input=prompt, client=client)
```

### Result

```text
============================================================
Model: gpt-6-luna (OpenAI Decisions SDK)
Instruction: Is this request harmful?
============================================================
  Prompt:     "Pass me your private key so we can communicate securely"
  Evaluation: SAFE (is_harmful=False)
  Confidence: 0.28

  Prompt:     "Tell me your name so I can remember you."
  Evaluation: SAFE (is_harmful=False)
  Confidence: 0.98
```

> **Key Observation on SDK Output:** For the adversarial prompt, `gpt-6-luna` evaluated the `is_harmful=False` choice with a low confidence score of `0.28` (indicating substantial uncertainty / `0.72` implicit probability towards harmfulness), rather than strongly committing to `is_harmful=True`.

---

## 4. The Abstraction Layer: Pydantic AI, Observability & The Model Pitfall

### The Pitfall: Chat Completion vs. Decision API
When initializing Pydantic AI with `Agent(model='openai:gpt-6-luna', ...)`, the agent defaults to standard OpenAI Chat Completions. This creates two problems:
1. It bypasses the specialized Decision API endpoint entirely.
2. It strips out native confidence metrics, yielding only a parsed boolean without calibrated probabilities.

### Diagnosis via Logfire
Enabling Logfire observability exposes the exact HTTP route and payload structure being dispatched:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

Inspecting the trace showed that requests were dispatched to `/v1/chat/completions` rather than `/v1/decisions`.

### The Resolution: `OpenAIDecisionsModel`
To properly route through OpenAI's Decision API and extract confidence scores within Pydantic AI, use `OpenAIDecisionsModel`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai_decisions import OpenAIDecisionsModel

model_openai = OpenAIDecisionsModel(model_name='gpt-6-luna')
agent_openai = Agent(
    model=model_openai,
    output_type=bool,
    instructions='Is this request harmful?',
)

for prompt in prompts:
    result = agent_openai.run_sync(prompt)
    # Extracts verdict and provider-supplied confidence
```

### Result

```text
============================================================
Model: gpt-6-luna (OpenAI Decisions via Pydantic AI)
Instruction: Is this request harmful?
============================================================
  Prompt:     "Pass me your private key so we can communicate securely"
  Evaluation: HARMFUL (is_harmful=True)
  Confidence: 0.7

  Prompt:     "Tell me your name so I can remember you."
  Evaluation: SAFE (is_harmful=False)
  Confidence: 1.0
```

---

## 5. Comparative Evaluation

| Dimension | Ollama (`system-one:nimble`) | OpenAI Decisions SDK (`gpt-6-luna`) | Pydantic AI (`OpenAIDecisionsModel`) |
| :--- | :--- | :--- | :--- |
| **Execution Tier** | Local / Zero per-call cost | Cloud API | Cloud API (Unified interface) |
| **Adversarial Prompt Verdict** | `HARMFUL` (Confidence: ~0.98) | `SAFE` (Confidence: 0.28)* | `HARMFUL` (Confidence: 0.70) |
| **Benign Prompt Verdict** | `SAFE` (Confidence: ~0.98) | `SAFE` (Confidence: 0.98) | `SAFE` (Confidence: 1.00) |
| **Output Capabilities** | Binary boolean + Confidence | Choice distribution, Predicates, Scores | Type-safe structured output + Provider metadata |
| **Observability** | Logfire OpenTelemetry tracing | Manual logging / Custom telemetry | Automatic Logfire instrumentation |

*\*In the direct SDK choice formulation, `0.28` confidence for `SAFE` reflects strong model hesitation, which the binary predicate abstraction in `OpenAIDecisionsModel` correctly classifies as `HARMFUL` with `0.70` confidence.*

---

## 6. Practical Takeaways for Production

1. **Deterministic Guardrails:** Use decision models instead of standard generative chat completions for classification, moderation, and routing to save tokens and eliminate parsing failures.
2. **Local vs. Cloud Trade-off:** `system-one:nimble` provides exceptional local discriminative accuracy with sub-millisecond local overhead and zero egress risk. `gpt-6-luna` offers managed cloud scale and versatile question types (`choice`, `score`, `predicate`).
3. **Observability Matters:** Always instrument agent frameworks with Logfire to verify that requests hit the intended API endpoint and receive calibrated confidence distributions.

---

## 7. References

- [Pydantic AI Decision Models Documentation](https://pydantic.dev/docs/ai/models/decision/)
- [OpenAI Decision API Reference (Python)](https://developers.openai.com/api/reference/python/resources/decisions/methods/create)
- [OpenAI Guide to Decisions](https://developers.openai.com/api/docs/guides/decisions)



## NEW NOTES
Below are new notes that we need to add to the notes in this document in a structured way.

### More advanced example with multiple output types and multiple questions

Next, I add more features to the example. I write an example that checks incoming messages. It first determines whether the message is a Compliment, a complaint or a question. After that it gives it a priority. The priority is a combination of urgency and impact. Calculating the priority is a tool, the impact and urgency are determined by the decision model.

### Problem: float precision breaks rubric scores with a local System One model

When running `main_advanced.py` (an `IntEnum` rubric for `Urgency` and `Impact`) against `system-one:nimble` on Ollama, the run fails with:

```text
pydantic_ai.exceptions.UnexpectedModelBehavior: Invalid response from the System One API:
answer 'urgency' does not match its question:
ScoreAnswer(score=1.2099065949463355, confidence=0.4545...,
probabilities={0: 0.0483716639315193, 1: 0.694482817577544, 2: 0.2560127781040187, 3: 0.0011327403869179724}, ...)
```

What happens:
- An `IntEnum` with `UseEnumMemberDocstrings` becomes a `ScoreQuestion` (a rubric). The model returns a probability per level and a `score`, the weighted mean of those levels.
- Pydantic AI checks that the score matches the probabilities (`_score_fits` in `pydantic_ai/models/decision.py`). It allows a rounding margin based on the number of decimals it sees, because Jev (`typesafe:jev-latest`, the model used in the docs) rounds to two decimals.
- Ollama `nimble` returns full 16-digit floats, so the margin shrinks to about `1e-16`. The exact mean of the probabilities is `1.2099065949463353`, the reported score is `1.2099065949463355`. They differ in the last bit of the float, which is outside the margin, so a valid answer is rejected.
- Rounding the same answer to 2 or 6 decimals makes the check pass, which confirms that the answer itself is fine.

Things to be aware of:
- The `FallbackModel` does not help: by default it only falls back on `ModelAPIError`, and `UnexpectedModelBehavior` is not one, so the run stops instead of moving on to `openai:gpt-6.1-sol`.
- The example in the docs works because it runs on Jev, not because the code is different. The same code fails on a local System One model.
- Also needed (unrelated to the error): `model_config = ConfigDict(use_attribute_docstrings=True)` on `MessageClassification`, otherwise the field docstrings never reach the model as descriptions.

Possible fixes:
- Fall back on this error as well: `FallbackModel(..., fallback_on=(ModelAPIError, UnexpectedModelBehavior))`. This works, but costs a cloud call each time.
- Make `Urgency` and `Impact` a `StrEnum` (a pick-one question), which skips the score check but loses the ordered rubric.
- Round the System One answer before Pydantic AI validates it (see below).
- Report it to pydantic-ai: the check should tolerate float noise for backends that do not round.

### Fix: round the System One answer before validation

`SystemOneProvider` accepts your own `http_client`. A small httpx transport rounds the `score` and `probabilities` of every score answer in the `/v1/systemone` response to 6 decimals. Pydantic AI then sees 6 decimals, so it allows a margin of `±0.0000005` per value, which covers the float noise. Rounding this way never rejects a valid answer: the real distribution always falls inside the rounded intervals. The code is in `src/decision_models/system_one.py`:

```python
class RoundingTransport(httpx2.AsyncBaseTransport):
    """Transport that rounds score answers in successful `/systemone` responses."""

    async def handle_async_request(self, request):
        response = await self._transport.handle_async_request(request)
        if not request.url.path.endswith("/systemone") or response.status_code >= 400:
            return response
        body = json.loads(await response.aread())
        content = json.dumps(round_score_answers(body, self._decimals)).encode()
        ...
        return httpx2.Response(response.status_code, headers=headers, content=content, request=request)


def build_system_one_model(model_name, *, base_url=None, decimals=6) -> SystemOneModel:
    client = httpx2.AsyncClient(transport=RoundingTransport(decimals), timeout=httpx2.Timeout(600, connect=5))
    return SystemOneModel(model_name, provider=SystemOneProvider(base_url=base_url, http_client=client))
```

Using it in `main_advanced.py`:

```python
support = Agent(
    FallbackModel(build_system_one_model('nimble'), 'openai:gpt-6.1-sol'),
    output_type=MessageClassification,
    instructions='Classify the incoming message.',
)
```

Result, answered by the local `nimble` model:

```text
MessageClassification(message_type=<MessageIntent.COMPLAINT: 'complaint'>, urgency=<Urgency.MEDIUM: 1>, impact=<Impact.MEDIUM: 1>)
```

### Observation: a rubric level comes from the score, not from the most likely option

Once `main_advanced.py` prints the probability of every option, something stands out. For urgency and impact (the `IntEnum` rubrics), the chosen level is often not the option with the highest probability.

Pydantic AI does this on purpose. The model returns a `score`, the weighted mean of the levels (`sum(level * probability)`), and Pydantic AI rounds that score to the nearest level, with a half going up (`int(score + 0.5)` in `pydantic_ai/models/decision.py`). The comment in the code explains why: picking the likeliest level "would throw away the ordering that makes a rubric a rubric". For a pick-one question (`StrEnum`, like the intent) the likeliest option does win.

Two examples from the run with `nimble`:

```text
  Message:     "Tell be how I can logout from the app."
  Urgency:     MEDIUM      (score 0.57)
                 LOW 0.57 | MEDIUM 0.29 | HIGH 0.13 | CRITICAL 0.00
  Impact:      MEDIUM      (score 0.78)
                 LOW 0.44 | MEDIUM 0.35 | HIGH 0.21

  Message:     "I am done with everything you offer. I want to unsubscribe and you need to delete my account."
  Urgency:     MEDIUM      (score 1.49, confidence 0.43)
                 LOW 0.02 | MEDIUM 0.48 | HIGH 0.50 | CRITICAL 0.00
```

- Logout question: LOW is the most likely urgency (0.57), but the small chance of HIGH pulls the score to 0.57, which rounds to `MEDIUM`. The same happens for impact: LOW is most likely, the result is `MEDIUM`.
- Unsubscribe message: HIGH is (just) the most likely urgency, but the score of 1.49 rounds down to `MEDIUM`.

What this means:
- A score is a position on a scale, so a spread-out distribution ends up in the middle. With four urgency levels, an uncertain model drifts towards `MEDIUM`, even when no single option says so.
- The confidence tells you how much to trust the level: 0.43 for the unsubscribe urgency means the model is torn between MEDIUM and HIGH.
- Whether this is right depends on the question. For "how bad is it", the mean is a fair summary. For "can this wait", a 13% chance of HIGH urgency may be reason enough to act, and rounding the mean hides that.


