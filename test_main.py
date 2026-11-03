import io
import json
import tempfile
import unittest
from pathlib import Path

from main import run_application
from model_runner import ModelRunner
from performance_monitor import PerformanceMonitor


def valid_model_response():
    return json.dumps(
        {
            "summary": "The passage explains how plants use light to make food.",
            "questions": [
                {
                    "question": f"Question {index}?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option A",
                }
                for index in range(1, 4)
            ],
        }
    )


class FixedResponseBackend:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response


class FailingBackend:
    def generate(self, prompt):
        raise OSError("test backend failure")


class MainApplicationTests(unittest.TestCase):
    def make_dependencies(self, directory, backend=None):
        stream = io.StringIO()
        log_path = Path(directory) / "logs" / "performance.log"
        runner = ModelRunner(backend or FixedResponseBackend(valid_model_response()))
        monitor = PerformanceMonitor(log_path)
        return stream, log_path, runner, monitor

    def test_complete_manual_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, log_path, runner, monitor = self.make_dependencies(directory)
            lines = iter(["word " * 50, "word " * 50, "word " * 50, ""])

            status = run_application(
                model_runner=runner,
                performance_monitor=monitor,
                execution_kind="test_model",
                input_fn=lambda: next(lines),
                output_stream=stream,
            )

            log_record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(status, 0)
        self.assertIn("SUMMARY", stream.getvalue())
        self.assertIn("QUIZ", stream.getvalue())
        self.assertIn("PERFORMANCE REPORT", stream.getvalue())
        self.assertIn("Application RAM (this process)", stream.getvalue())
        self.assertEqual(log_record["execution_kind"], "test_model")

    def test_reads_passage_from_text_file(self):
        with tempfile.TemporaryDirectory() as directory:
            passage_path = Path(directory) / "passage.txt"
            passage_path.write_text("education " * 150, encoding="utf-8")
            stream, _, runner, monitor = self.make_dependencies(directory)

            status = run_application(
                file_path=passage_path,
                model_runner=runner,
                performance_monitor=monitor,
                execution_kind="test_model",
                output_stream=stream,
            )

        self.assertEqual(status, 0)
        self.assertIn("Reading passage from", stream.getvalue())
        self.assertIn("SUMMARY", stream.getvalue())

    def test_rejects_invalid_passage_without_running_model(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, log_path, runner, monitor = self.make_dependencies(directory)

            status = run_application(
                passage="too short",
                model_runner=runner,
                performance_monitor=monitor,
                output_stream=stream,
            )

        self.assertEqual(status, 1)
        self.assertIn("Invalid passage", stream.getvalue())
        self.assertFalse(log_path.exists())

    def test_handles_model_execution_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, log_path, runner, monitor = self.make_dependencies(
                directory, FailingBackend()
            )

            status = run_application(
                passage="learning " * 150,
                model_runner=runner,
                performance_monitor=monitor,
                execution_kind="test_failure",
                output_stream=stream,
            )

            log_record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(status, 1)
        self.assertIn("Model execution failed", stream.getvalue())
        self.assertEqual(log_record["execution_kind"], "test_failure")
        self.assertIn("ModelRunnerError", log_record["error"])

    def test_handles_malformed_model_response(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, log_path, runner, monitor = self.make_dependencies(
                directory, FixedResponseBackend("{invalid json")
            )

            status = run_application(
                passage="learning " * 150,
                model_runner=runner,
                performance_monitor=monitor,
                output_stream=stream,
            )
            log_exists = log_path.exists()

        self.assertEqual(status, 1)
        self.assertIn("Could not process model response", stream.getvalue())
        self.assertIn("PERFORMANCE REPORT", stream.getvalue())
        self.assertTrue(log_exists)

    def test_marks_default_mock_output_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = io.StringIO()
            monitor = PerformanceMonitor(Path(directory) / "performance.log")

            status = run_application(
                passage="learning " * 150,
                performance_monitor=monitor,
                output_stream=stream,
            )

        self.assertEqual(status, 0)
        self.assertIn("MOCK RESPONSE: no AI inference was performed.", stream.getvalue())
        self.assertIn("Operation duration (mock; not AI inference)", stream.getvalue())


if __name__ == "__main__":
    unittest.main()