import argparse
import ast
import json
import math
import operator
import os
import sys

from openai import OpenAI


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
            if tool_call.function.name != "calculator":
                result = {"error": "Neznámý nástroj."}
            else:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    result = {"result": calculator(arguments["expression"])}
                except (ArithmeticError, KeyError, SyntaxError, TypeError, ValueError) as error:
                    result = {"error": str(error)}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    raise RuntimeError("LLM překročilo maximální počet kol nástrojových volání.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pošle dotaz do OpenAI API a podle potřeby použije kalkulačku."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Kolik je 25 * (10 + 2)?",
        help="Dotaz pro LLM (výchozí dotaz použije kalkulačku).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Název modelu (výchozí: gpt-4o-mini).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Chybí proměnná prostředí OPENAI_API_KEY.", file=sys.stderr)
        return 1

    print(run(args.prompt, OpenAI(), args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())