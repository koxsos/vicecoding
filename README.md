# Python script for an LLM API

This example calls an LLM through OpenRouter's OpenAI-compatible API. The
model can call a local `calculator` tool; its result is sent back to the model
to produce the final answer.

## Setup

Requires Python 3.9+ and an OpenRouter API key.

```bash
python -m pip install -r requirements.txt
export OPENROUTER_API_KEY="your-openrouter-api-key"
python pythonLLMscript.py
```

In PowerShell, set the key for the current terminal session with:

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-api-key"
python pythonLLMscript.py
```

Never paste your key into source code, commit it, or share it in chat.

The default model is OpenRouter's free
`google/gemma-4-31b-it:free`. You can choose another model available on
OpenRouter with `--model`:

```bash
python pythonLLMscript.py --model "google/gemma-4-31b-it:free" "What is 25 * (10 + 2)?"
```

Alternatively, set `OPENROUTER_MODEL` to use a different model by default.

The calculator supports numeric expressions using `+`, `-`, `*`, `/`, `//`,
`%`, and `**`. It uses a restricted AST parser rather than `eval`.

## Offline tests (no API key or credits)

```bash
python -m unittest discover -s tests -v
```

These tests check calculator behavior, the mocked tool-call round trip, and
OpenRouter client configuration. They do not make network requests.

## Live API test

After setting `OPENROUTER_API_KEY`, run:

```bash
python pythonLLMscript.py "What is 25 * (10 + 2)?"
```

This sends a request to OpenRouter. Free models can have rate limits or
temporary availability constraints.
The selected model must be available to your OpenRouter account and support
tool/function calling.

If OpenRouter returns `429` (`RateLimitError`), the selected free provider may
be busy or your account may have reached a rate limit. Wait and retry, or try
another free model that supports tool calling:

```bash
python pythonLLMscript.py --model "another/provider-model:free" "hello"
```

## List or test all free tool-capable models

List models currently advertised by OpenRouter as free and tool-capable (this
fetches the public model catalog but does not call any models):

```bash
python pythonLLMscript.py --list-free-models
```

To run a calculator-tool round trip against every model in that list:

```bash
python pythonLLMscript.py --test-free-models
```

This makes up to two API calls per model, sequentially, with a two-second
pause between models. It may take a while and can hit free-tier rate limits;
the sweep stops on the first `429`. Adjust the pause with `--delay 5`, or
set it to `0` to omit the pause.
