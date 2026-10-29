import json
import unittest
from contextlib import redirect_stderr
from io import BytesIO, StringIO
from types import SimpleNamespace
from unittest.mock import patch

from httpx import Request, Response
from openai import RateLimitError

from pythonLLMscript import (
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    calculator,
    create_client,
    get_free_tool_models,
    main,
    run,
    test_model_tool_round_trip,
)


class CalculatorTests(unittest.TestCase):
    def test_evaluates_arithmetic(self):
        self.assertEqual(calculator("25 * (10 + 2)"), "300")
        self.assertEqual(calculator("-5 + 2 ** 3"), "3")
        self.assertEqual(calculator("17 // 5"), "3")

    def test_rejects_unsupported_or_invalid_expressions(self):
        for expression in (
            "__import__('os').system('echo unsafe')",
            "1 / 0",
            "2 ** 101",
            "",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises((ArithmeticError, SyntaxError, ValueError)):
                    calculator(expression)


class ToolCallingTests(unittest.TestCase):
    def test_returns_final_llm_answer_after_calculator_call(self):
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="calculator",
                arguments='{"expression":"25 * (10 + 2)"}',
            ),
        )

        class FakeMessage:
            def __init__(self, content=None, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

            def model_dump(self, exclude_none=True):
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": self.tool_calls,
                }

        class FakeCompletions:
            def __init__(self):
                self.responses = [
                    FakeMessage(tool_calls=[tool_call]),
                    FakeMessage(content="The result is 300."),
                ]

            def create(self, **kwargs):
                if len(self.responses) == 1:
                    self.assert_tool_result(kwargs["messages"])
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=self.responses.pop(0))]
                )

            @staticmethod
            def assert_tool_result(messages):
                self_tool_message = messages[-1]
                if self_tool_message["content"] != '{"result": "300"}':
                    raise AssertionError("Calculator result was not sent back to LLM")

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )

        answer = run("What is 25 * (10 + 2)?", client, "test-model")

        self.assertEqual(answer, "The result is 300.")


class OpenRouterConfigurationTests(unittest.TestCase):
    def test_defaults_to_requested_free_model(self):
        self.assertEqual(DEFAULT_MODEL, "google/gemma-4-31b-it:free")

    def test_creates_openai_compatible_client_for_openrouter(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch("pythonLLMscript.OpenAI") as openai_client:
                create_client()

        openai_client.assert_called_once_with(
            api_key="test-key",
            base_url=OPENROUTER_BASE_URL,
        )

    def test_requires_openrouter_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                create_client()

    def test_rate_limit_has_actionable_message_without_traceback(self):
        response = Response(
            429,
            request=Request("POST", OPENROUTER_BASE_URL + "/chat/completions"),
        )
        rate_limit_error = RateLimitError(
            "Provider returned error",
            response=response,
            body={"error": "Provider returned error"},
        )
        stderr = StringIO()

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch("sys.argv", ["pythonLLMscript.py"]),
            patch("pythonLLMscript.create_client"),
            patch("pythonLLMscript.run", side_effect=rate_limit_error),
            redirect_stderr(stderr),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertIn("OpenRouter rate limit", stderr.getvalue())
        self.assertIn("Wait and retry", stderr.getvalue())

    def test_free_model_filter_requires_zero_pricing_and_tool_support(self):
        catalog = {
            "data": [
                {
                    "id": "vendor/tool-free:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "supported_parameters": ["tools"],
                },
                {
                    "id": "vendor/no-tools:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "supported_parameters": [],
                },
                {
                    "id": "vendor/paid",
                    "pricing": {"prompt": "0", "completion": "0.01"},
                    "supported_parameters": ["tools"],
                },
            ]
        }

        with patch(
            "pythonLLMscript.urllib.request.urlopen",
            return_value=BytesIO(json.dumps(catalog).encode("utf-8")),
        ):
            models = get_free_tool_models()

        self.assertEqual(models, ["vendor/tool-free:free"])

    def test_tool_round_trip_requires_tool_call_and_returns_final_answer(self):
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="calculator",
                arguments='{"expression":"19 * 23"}',
            ),
        )
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[tool_call],
                            model_dump=lambda exclude_none: {
                                "role": "assistant",
                                "tool_calls": [tool_call],
                            },
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="The result is 437.")
                    )
                ]
            ),
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: responses.pop(0)
                )
            )
        )

        answer = test_model_tool_round_trip(client, "vendor/tool-free:free")

        self.assertEqual(answer, "The result is 437.")


if __name__ == "__main__":
    unittest.main()
