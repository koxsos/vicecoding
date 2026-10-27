import unittest
from types import SimpleNamespace

from pythonLLMscript import calculator, run


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


if __name__ == "__main__":
    unittest.main()
