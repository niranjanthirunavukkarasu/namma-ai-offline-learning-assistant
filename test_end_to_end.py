import io
import json
import tempfile
import unittest
from pathlib import Path

from main import run_application
from model_runner import ModelRunner
from performance_monitor import PerformanceMonitor


def make_response():
    return json.dumps(
        {
            "summary": "The passage describes how plants convert light into energy.",
            "questions": [
                {
                    "question": f"End-to-end question {index}?",
                    "options": ["Choice A", "Choice B", "Choice C", "Choice D"],
                    "correct_answer": "Choice C",
                }
                for index in range(1, 4)
            ],
        }
    )


class ResponseBackend:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else make_response()
        self.error = error
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response


class EndToEndTests(unittest.TestCase):
    def create_context(self, directory, backend=None):
        output = io.StringIO()
        log_path = Path(directory) / "logs" / "performance.log"
        runner = ModelRunner(backend or ResponseBackend())
        monitor = PerformanceMonitor(log_path)
        return output, log_path, runner, monitor

    def test_e2e_01_manual_workflow_summary_quiz_and_performance(self):
        input_lines = ["plant learning " * 25 for _ in range(3)]
        passage = "\n".join(input_lines)
        with tempfile.TemporaryDirectory() as directory:
            output, log_path, runner, monitor = self.create_context(directory)
            lines = iter([*input_lines, ""])

            result = run_application(
                model_runner=runner,
                performance_monitor=monitor,
                execution_kind="test_backend",
                input_fn=lambda: next(lines),
                output_stream=output,
            )
            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        rendered = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("The passage describes", rendered)
        self.assertEqual(rendered.count("End-to-end question"), 3)
        self.assertEqual(rendered.count("Correct answer: C. Choice C"), 3)
        self.assertIn("PERFORMANCE REPORT", rendered)
        prompt = json.loads(runner._backend.prompts[0])
        self.assertEqual(prompt["passage"], passage)
        self.assertEqual(record["execution_kind"], "test_backend")

    def test_e2e_02_reads_and_processes_text_file(self):
        with tempfile.TemporaryDirectory() as directory:
            passage_file = Path(directory) / "lesson.txt"
            passage_file.write_text("water cycle " * 75, encoding="utf-8")
            output, _, runner, monitor = self.create_context(directory)

            result = run_application(
                file_path=passage_file,
                model_runner=runner,
                performance_monitor=monitor,
                output_stream=output,
            )

        self.assertEqual(result, 0)
        self.assertIn("Reading passage from", output.getvalue())
        self.assertIn("SUMMARY", output.getvalue())

    def test_e2e_03_rejects_both_invalid_word_count_edges(self):
        for word_count in (149, 301):
            with self.subTest(word_count=word_count):
                with tempfile.TemporaryDirectory() as directory:
                    output, log_path, runner, monitor = self.create_context(directory)

                    result = run_application(
                        passage="word " * word_count,
                        model_runner=runner,
                        performance_monitor=monitor,
                        output_stream=output,
                    )

                    self.assertEqual(result, 1)
                    self.assertIn("Invalid passage", output.getvalue())
                    self.assertEqual(runner._backend.prompts, [])
                    self.assertFalse(log_path.exists())

    def test_e2e_04_rejects_empty_manual_input(self):
        with tempfile.TemporaryDirectory() as directory:
            output, log_path, runner, monitor = self.create_context(directory)

            result = run_application(
                model_runner=runner,
                performance_monitor=monitor,
                input_fn=lambda: "",
                output_stream=output,
            )

        self.assertEqual(result, 1)
        self.assertIn("passage must not be empty", output.getvalue())
        self.assertEqual(runner._backend.prompts, [])
        self.assertFalse(log_path.exists())

    def test_e2e_05_mock_execution_is_not_reported_as_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            monitor = PerformanceMonitor(Path(directory) / "performance.log")

            result = run_application(
                passage="learning " * 150,
                performance_monitor=monitor,
                output_stream=output,
            )

            record = json.loads(
                monitor.log_path.read_text(encoding="utf-8").splitlines()[0]
            )

        rendered = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("MOCK RESPONSE: no AI inference was performed.", rendered)
        self.assertIn("Operation duration (mock; not AI inference)", rendered)
        self.assertEqual(record["execution_kind"], "mock")

    def test_e2e_06_malformed_model_json_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            output, log_path, runner, monitor = self.create_context(
                directory, ResponseBackend(response="not-json")
            )

            result = run_application(
                passage="learning " * 150,
                model_runner=runner,
                performance_monitor=monitor,
                output_stream=output,
            )
            log_exists = log_path.exists()

        self.assertEqual(result, 1)
        self.assertIn("Could not process model response", output.getvalue())
        self.assertIn("PERFORMANCE REPORT", output.getvalue())
        self.assertTrue(log_exists)

    def test_e2e_07_model_execution_failure_is_reported_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            output, log_path, runner, monitor = self.create_context(
                directory, ResponseBackend(error=OSError("backend offline"))
            )

            result = run_application(
                passage="learning " * 150,
                model_runner=runner,
                performance_monitor=monitor,
                execution_kind="failing_test_backend",
                output_stream=output,
            )
            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(result, 1)
        self.assertIn("Model execution failed", output.getvalue())
        self.assertEqual(record["execution_kind"], "failing_test_backend")
        self.assertIn("ModelRunnerError", record["error"])

    def test_e2e_08_performance_report_has_process_memory_and_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            output, log_path, runner, monitor = self.create_context(directory)

            result = run_application(
                passage="learning " * 150,
                model_runner=runner,
                performance_monitor=monitor,
                output_stream=output,
            )

            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        rendered = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("Application RAM (this process)", rendered)
        self.assertIn("Operation duration:", rendered)
        self.assertGreaterEqual(record["duration_seconds"], 0)
        self.assertGreater(record["application_ram_before_bytes"], 0)
        self.assertGreater(record["application_ram_after_bytes"], 0)

    def test_e2e_09_missing_passage_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            output, log_path, runner, monitor = self.create_context(directory)

            result = run_application(
                file_path=Path(directory) / "missing.txt",
                model_runner=runner,
                performance_monitor=monitor,
                output_stream=output,
            )

        self.assertEqual(result, 1)
        self.assertIn("Could not read passage file", output.getvalue())
        self.assertEqual(runner._backend.prompts, [])
        self.assertFalse(log_path.exists())


if __name__ == "__main__":
    unittest.main()