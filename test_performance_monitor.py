import json
import tempfile
import time
import unittest
from pathlib import Path

from performance_monitor import PerformanceMonitor, PerformanceMonitorError


class PerformanceMonitorTests(unittest.TestCase):
    def test_measures_duration_and_application_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = PerformanceMonitor(Path(directory) / "logs" / "performance.log")

            result = monitor.execute(
                lambda: "mock response", execution_kind="mock"
            )

        self.assertEqual(result.output, "mock response")
        self.assertEqual(result.metrics.execution_kind, "mock")
        self.assertGreaterEqual(result.metrics.duration_seconds, 0)
        self.assertGreater(result.metrics.application_ram_before_bytes, 0)
        self.assertGreater(result.metrics.application_ram_after_bytes, 0)
        self.assertGreaterEqual(
            result.metrics.application_ram_peak_bytes,
            max(
                result.metrics.application_ram_before_bytes,
                result.metrics.application_ram_after_bytes,
            ),
        )
        self.assertIn("subprocess memory is not included", result.metrics.memory_scope)

    def test_samples_peak_memory_during_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = PerformanceMonitor(
                Path(directory) / "performance.log",
                sample_interval_seconds=0.001,
            )

            result = monitor.execute(lambda: time.sleep(0.02))

        self.assertGreater(result.metrics.duration_seconds, 0)
        self.assertGreaterEqual(
            result.metrics.application_ram_peak_bytes,
            result.metrics.application_ram_before_bytes,
        )

    def test_calculates_model_file_size_when_path_is_given(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.bin"
            model_path.write_bytes(b"model-data")
            monitor = PerformanceMonitor(Path(directory) / "performance.log")

            result = monitor.execute(lambda: "done", model_path=model_path)

        self.assertEqual(result.metrics.model_file_size_bytes, len(b"model-data"))

    def test_model_file_size_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = PerformanceMonitor(Path(directory) / "performance.log")

            result = monitor.execute(lambda: "done")

        self.assertIsNone(result.metrics.model_file_size_bytes)

    def test_logs_performance_metrics_as_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "logs" / "performance.log"
            monitor = PerformanceMonitor(log_path)

            monitor.execute(lambda: "done", execution_kind="mock")

            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(record["execution_kind"], "mock")
        self.assertIn("duration_seconds", record)
        self.assertIn("application_ram_before_bytes", record)
        self.assertIn("application_ram_after_bytes", record)
        self.assertIn("application_ram_peak_bytes", record)
        self.assertIn("subprocess memory is not included", record["memory_scope"])

    def test_rejects_invalid_model_path_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = PerformanceMonitor(Path(directory) / "performance.log")
            executed = []

            with self.assertRaises(PerformanceMonitorError):
                monitor.execute(
                    lambda: executed.append(True),
                    model_path=Path(directory) / "missing-model.bin",
                )

        self.assertEqual(executed, [])

    def test_logs_and_reraises_operation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "performance.log"
            monitor = PerformanceMonitor(log_path)

            with self.assertRaisesRegex(RuntimeError, "operation failed"):
                monitor.execute(
                    lambda: (_ for _ in ()).throw(RuntimeError("operation failed")),
                    execution_kind="mock",
                )

            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(record["error"], "RuntimeError: operation failed")


if __name__ == "__main__":
    unittest.main()