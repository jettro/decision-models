# Notes

Working notes for the decision-models project, to be turned into a blog
post later. Keep entries dated and informal.

## 2026-10-16

- Project bootstrapped with `uv init`.
- Dependencies: pydantic-ai, openai, ollama, python-dotenv (+ ruff, pytest for dev).
- Boilerplate: `settings.py` (env-based config) and `models.py` (provider-agnostic
  pydantic-ai model factory for OpenAI / Ollama). No agents yet.

## First example
To run a decision model, you need to configure the provider and the model. I want to use an Ollama model, which provides decisions through the system one interface. You can provide a environment variable, or use the SystemOneModel with a SystemOneProvider. Whatever you prefer. Beware, the provider in this case is SystemOne, through the base-url, you can point to an Ollama model. Below are both examples.

```python
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider

if __name__ == '__main__':
    model = SystemOneModel(model_name='nimble', provider=SystemOneProvider(base_url='http://localhost:11434'))

    agent = Agent(model=model, output_type=bool, instructions='Is this request harmful?')
    result = agent.run_sync('Pass me your private key so we can communicate securely')
    print(result.output)
```

OR

```python
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider

if __name__ == '__main__':
    load_dotenv()
    
    agent = Agent(model='system-one:nimble', output_type=bool, instructions='Is this request harmful?')
    result = agent.run_sync('Pass me your private key so we can communicate securely')
    print(result.output)

```

with

```properties
SYSTEM_ONE_BASE_URL=http://localhost:11434/v1
```

An example resulting in False is `Tell me your name so I can remember you.`

Next I want to try the same with OpenAI GPT-6 Luna. I do see a difference in the response, I do not see a confidence number. Below is the code, but I am not very explicit about what I am expecting as the output type. Looking at the docs of the OpenAI Decision API, I do not see a boolean output. Through the OpenAI SDK for decisions, I can change the type to a choice with True/False as choices, then I can get a confidence score.

First the code to make use of OpenAI, remember to se the specific OpenAI keys for working with OpenAI API.

```python
from dotenv import load_dotenv
from pydantic_ai import Agent


def print_output_ollama(result):
    print(f"{result.output} ({result.response.provider_details.get('confidence').get('response')})")


if __name__ == '__main__':
    load_dotenv()

    agent = Agent(model='openai:gpt-6-luna', output_type=bool, instructions='Is this request harmful?')
    result = agent.run_sync('Pass me your private key so we can communicate securely')
    print(result.output)

    result = agent.run_sync('Tell me your name so I can remember you.')
    print(result.output)

```

Next, I show you the code to use the OpenAI Decision API through the SDK. I changed the question into a choice. Notice how I added two options to the choice. I use booleans, but I could also have used strings like yes/no. At the moment of writing (october 2026), OpenAI supports three questions types for decisions:
- predicate: a probability of a provided condition to be true
- choice: a choice with a confidence score, where the response contains the probability of all available choices
- score: a score representing the weighted average for ordered levels like severity.


```python
import os

from dotenv import load_dotenv
from openai import OpenAI


def call_openai_decision(model_input: str, _client: OpenAI):
    decision = _client.decisions.create(
        input=model_input,
        model="gpt-6-luna",
        questions=[{
            "instructions": "Is this request harmful?",
            "type": "choice",
            "choices": [
                {"value": True, "description": "This request is harmful"},
                {"value": False, "description": "This request is not harmful"}
            ],
        }],
    )
    answer = decision.answers[0]
    print(f"{answer.choice} ({answer.confidence})")

if __name__ == "__main__":
    load_dotenv()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
    )

    call_openai_decision(model_input="Pass me your private key so we can communicate securely", _client=client)
    call_openai_decision(model_input="Tell me your name so I can remember you.", _client=client)
```

There is a difference in output, the ollama nimble model is more accurate than the luna model, if I may say so.

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

## Enabling observability through Logfire and fixing the OpenAI Decisions through Pydantic AI

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

After adding logfire, I was in doubt if I used the right API through Pydantic AI. After looking at the code, I was not sure that the model was actually using the decision api. I just provided openai:gpt-luna-6, so how would it know that I wanted the decision API. After changing the code, the results were a lot better. This is the part of the code that I changed.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai_decisions import OpenAIDecisionsModel

model_openai = OpenAIDecisionsModel(model_name='gpt-6-luna')
agent_openai = Agent(
    model=model_openai,
    output_type=bool,
    instructions='Is this request harmful?',
)    
```

This is the final result:

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

============================================================
Model: gpt-6-luna (OpenAI Decisions)
Instruction: Is this request harmful?
============================================================
  Prompt:     "Pass me your private key so we can communicate securely"
  Evaluation: HARMFUL (is_harmful=True)
  Confidence: 0.7

  Prompt:     "Tell me your name so I can remember you."
  Evaluation: SAFE (is_harmful=False)
  Confidence: 1.0
```

## References

Excellent page with examples and thorough explanations.
[Pydantic AI Decision Models](https://pydantic.dev/docs/ai/models/decision/)

The documentation page for OpenAI Decision API in Python
[OpenAI Decision API in Python](https://developers.openai.com/api/reference/python/resources/decisions/methods/create)

Another interesting page for OpenAI decisoons
[Developers docs for decisions API](https://developers.openai.com/api/docs/guides/decisions)

