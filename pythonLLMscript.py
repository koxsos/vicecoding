import argparse
import ast
import json
import math
import operator
import os
import sys
import time
import urllib.request

from openai import APIError, OpenAI, RateLimitError


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_node(node):
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent musí být v rozsahu -100 až 100.")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    raise ValueError("Výraz obsahuje nepovolenou operaci.")


def calculator(expression: str) -> str:
    """Vyhodnotí bezpečně jednoduchý matematický výraz."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Matematický výraz nesmí být prázdný.")
    if len(expression) > 200:
        raise ValueError("Matematický výraz je příliš dlouhý.")
    result = _evaluate_node(ast.parse(expression, mode="eval"))
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("Výsledek není konečné číslo.")
    return str(result)


def _tool_result(tool_call) -> dict:
    if tool_call.function.name != "calculator":
        return {"error": "Neznámý nástroj."}
    try:
        arguments = json.loads(tool_call.function.arguments)
        return {"result": calculator(arguments["expression"])}
    except (ArithmeticError, KeyError, SyntaxError, TypeError, ValueError) as error:
        return {"error": str(error)}


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Vyhodnotí matematický výraz.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Matematický výraz, např. 25 * (10 + 2)",
                    }
                },
                "required": ["expression"],
            }
        }
    }
]


def run(prompt: str, client: OpenAI, model: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    for _ in range(5):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            if assistant_message.content is None:
                raise RuntimeError("LLM vrátilo prázdnou odpověď.")
            return assistant_message.content

        messages.append(assistant_message.model_dump(exclude_none=True))
        for tool_call in assistant_message.tool_calls:
            result = _tool_result(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    raise RuntimeError("LLM překročilo maximální počet kol nástrojových volání.")


def get_free_tool_models() -> list[str]:
    request = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/models",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        catalog = json.load(response)

    models = catalog.get("data")
    if not isinstance(models, list):
        raise ValueError("OpenRouter returned an invalid model catalog.")

    free_tool_models = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        pricing = model.get("pricing")
        supported_parameters = model.get("supported_parameters")
        if (
            isinstance(model_id, str)
            and model_id.endswith(":free")
            and isinstance(pricing, dict)
            and pricing.get("prompt") == "0"
            and pricing.get("completion") == "0"
            and isinstance(supported_parameters, list)
            and "tools" in supported_parameters
        ):
            free_tool_models.append(model_id)
    return sorted(free_tool_models)


def test_model_tool_round_trip(client: OpenAI, model: str) -> str:
    messages = [
        {
            "role": "user",
            "content": "Use the calculator tool to compute 19 * 23. Call the tool.",
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    assistant_message = response.choices[0].message
    if not assistant_message.tool_calls:
        raise RuntimeError("Model did not call the calculator tool.")

    messages.append(assistant_message.model_dump(exclude_none=True))
    for tool_call in assistant_message.tool_calls:
        result = _tool_result(tool_call)
        if "error" in result or result.get("result") != "437":
            raise RuntimeError(f"Calculator tool call failed: {result}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    answer = final_response.choices[0].message.content
    if not answer:
        raise RuntimeError("Model returned an empty final answer.")
    return answer


def test_free_models(client: OpenAI, models: list[str], delay: float) -> int:
    if not models:
        print("No free models advertising tool support were found.")
        return 1

    print(
        f"Testing {len(models)} free tool-capable models sequentially "
        "(up to 2 API calls each)."
    )
    failures = 0
    completed = 0
    for index, model in enumerate(models):
        if index:
            time.sleep(delay)
        print(f"[{index + 1}/{len(models)}] {model} ... ", end="", flush=True)
        try:
            answer = test_model_tool_round_trip(client, model)
        except RateLimitError as error:
            print(f"RATE LIMITED: {error}")
            print("Stopping the model sweep to avoid further rate-limit requests.")
            return 1
        except (APIError, RuntimeError, ValueError, KeyError, TypeError) as error:
            failures += 1
            print(f"FAILED: {error}")
        else:
            completed += 1
            print(f"PASS: {answer}")

    print(f"\nFinished: {completed} passed, {failures} failed.")
    return 0 if failures == 0 else 1


def create_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Chybí proměnná prostředí OPENROUTER_API_KEY.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pošle dotaz do OpenRouter API a podle potřeby použije kalkulačku."
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--list-free-models",
        action="store_true",
        help="List free OpenRouter models that advertise tool support.",
    )
    action.add_argument(
        "--test-free-models",
        action="store_true",
        help="Test the calculator tool round trip against every listed free model.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between model tests (default: 2).",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Kolik je 25 * (10 + 2)?",
        help="Dotaz pro LLM (výchozí dotaz použije kalkulačku).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        help=f"OpenRouter model slug (výchozí: {DEFAULT_MODEL}; lze nastavit OPENROUTER_MODEL).",
    )
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay must be zero or greater.")

    if args.list_free_models:
        try:
            models = get_free_tool_models()
        except (APIError, OSError, json.JSONDecodeError, ValueError) as error:
            print(f"Could not fetch OpenRouter model list: {error}", file=sys.stderr)
            return 1
        if not models:
            print("No free models advertising tool support were found.")
            return 1
        for model in models:
            print(model)
        print(f"\n{len(models)} free tool-capable models found.")
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Chybí proměnná prostředí OPENROUTER_API_KEY.", file=sys.stderr)
        return 1

    if args.test_free_models:
        try:
            models = get_free_tool_models()
        except (APIError, OSError, json.JSONDecodeError, ValueError) as error:
            print(f"Could not fetch OpenRouter model list: {error}", file=sys.stderr)
            return 1
        return test_free_models(create_client(), models, args.delay)

    try:
        answer = run(args.prompt, create_client(), args.model)
    except RateLimitError as error:
        print(
            f"OpenRouter rate limit for model '{args.model}': {error}\n"
            "Wait and retry, or select another free model that supports tool calling "
            "with --model.",
            file=sys.stderr,
        )
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())