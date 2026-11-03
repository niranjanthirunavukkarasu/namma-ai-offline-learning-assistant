import json
import unittest

from model_runner import ModelRunner, ModelRunnerError, MockModelBackend
from passage_input import build_model_instructions, validate_passage


class ModelRunnerTests(unittest.TestCase):
    def test_mock_backend_returns_explicit_mock_response(self):
        prompt = build_model_instructions("A sample educational passage.")
        response = ModelRunner(MockModelBackend()).run(prompt)

        result = json.loads(response)
        self.assertTrue(result["mock"])
        self.assertIn("no AI inference", result["message"])
        self.assertIn("summary", result)
        self.assertEqual(len(result["questions"]), 3)

    def test_runner_passes_prompt_to_backend(self):
        class RecordingBackend:
            def __init__(self):
                self.prompt = None

            def generate(self, prompt):
                self.prompt = prompt
                return "model response"

        prompt = build_model_instructions("A sample educational passage.")
        backend = RecordingBackend()

        response = ModelRunner(backend).run(prompt)

        self.assertEqual(backend.prompt, prompt)
        self.assertEqual(response, "model response")

    def test_runner_works_as_passage_validator_callback(self):
        passage = "learning " * 150
        runner = ModelRunner(MockModelBackend())

        response = validate_passage(passage, runner.run)

        self.assertTrue(json.loads(response)["mock"])

    def test_runner_wraps_backend_failure(self):
        class FailingBackend:
            def generate(self, prompt):
                raise OSError("backend unavailable")

        with self.assertRaises(ModelRunnerError) as raised:
            ModelRunner(FailingBackend()).run("prompt")

        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_runner_rejects_unusable_backend_responses(self):
        class FixedBackend:
            def __init__(self, response):
                self.response = response

            def generate(self, prompt):
                return self.response

        for response in (None, " \n"):
            with self.subTest(response=response):
                with self.assertRaises(ModelRunnerError):
                    ModelRunner(FixedBackend(response)).run("prompt")

    def test_runner_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            ModelRunner(MockModelBackend()).run("  \n")


if __name__ == "__main__":
    unittest.main()