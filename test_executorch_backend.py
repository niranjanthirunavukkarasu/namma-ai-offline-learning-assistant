import io
import json
import tempfile
import unittest
from pathlib import Path

from executorch_backend import (
    ExecuTorchBackend,
    ExecuTorchConfig,
    ExecuTorchIntegrationPendingError,
)
from main import run_application
from model_runner import ModelRunner
from performance_monitor import PerformanceMonitor


def fake_response():
    return json.dumps(
        {
            "summary": "Fake runner response for integration testing.",
            "questions": [
                {
                    "question": f"Fake question {index}?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                }
                for index in range(1, 4)
            ],
        }
    )


class FakeRunner:
    def __init__(self):
        self.received_prompts = []

    def generate(self, prompt):
        self.received_prompts.append(prompt)
        return fake_response()


class ExecuTorchBackendTests(unittest.TestCase):
    def test_config_preserves_caller_supplied_command_and_model_path(self):
        config = ExecuTorchConfig(
            runner_command=("/custom/runner", "--custom-flag"),
            model_path=Path("/custom/models/learning-model.pte"),
        )

        self.assertEqual(config.runner_command, ("/custom/runner", "--custom-flag"))
        self.assertEqual(config.model_path, Path("/custom/models/learning-model.pte"))

    def test_config_rejects_empty_command_or_non_path_model_location(self):
        with self.assertRaises(ValueError):
            ExecuTorchConfig(runner_command=(), model_path=Path("model.pte"))
        with self.assertRaises(TypeError):
            ExecuTorchConfig(runner_command=("runner",), model_path="model.pte")

    def test_exec_command_remains_disabled_until_interface_is_supplied(self):
        backend = ExecuTorchBackend(
            ExecuTorchConfig(
                runner_command=("provided-later",),
                model_path=Path("provided-later.pte"),
            )
        )

        with self.assertRaisesRegex(
            ExecuTorchIntegrationPendingError, "no command was executed"
        ):
            backend.generate("test prompt")

    def test_fake_runner_integrates_with_existing_application(self):
        fake_runner = FakeRunner()
        runner = ModelRunner(fake_runner)

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            monitor = PerformanceMonitor(Path(directory) / "performance.log")

            status = run_application(
                passage="test learning " * 75,
                model_runner=runner,
                performance_monitor=monitor,
                execution_kind="fake_runner_test",
                output_stream=output,
            )

        self.assertEqual(status, 0)
        self.assertEqual(len(fake_runner.received_prompts), 1)
        self.assertEqual(json.loads(fake_runner.received_prompts[0])["passage"], "test learning " * 75)
        self.assertIn("Fake runner response", output.getvalue())
        self.assertIn("Correct answer: A. A", output.getvalue())
        self.assertNotIn("no AI inference was performed", output.getvalue())


if __name__ == "__main__":
    unittest.main()