import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import psutil


class PerformanceMonitorError(RuntimeError):
    """Raised when performance data cannot be collected or saved."""


@dataclass(frozen=True)
class PerformanceMetrics:
    execution_kind: str
    duration_seconds: float
    application_ram_before_bytes: int
    application_ram_after_bytes: int
    application_ram_peak_bytes: int
    model_file_size_bytes: Optional[int]
    memory_scope: str = (
        "current Python process RSS; subprocess memory is not included"
    )
    error: Optional[str] = None


@dataclass(frozen=True)
class MonitoredExecution:
    output: Any
    metrics: PerformanceMetrics


class PerformanceMonitor:
    """Measure callable execution and current-process RAM, then append JSON logs."""

    def __init__(
        self,
        log_path: Path = Path("logs/performance.log"),
        sample_interval_seconds: float = 0.01,
    ):
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be greater than zero")
        self.log_path = Path(log_path)
        self.sample_interval_seconds = sample_interval_seconds
        self._log_lock = threading.Lock()
        self._process = psutil.Process(os.getpid())

    def execute(
        self,
        operation: Callable[..., Any],
        *args: Any,
        model_path: Optional[Path] = None,
        execution_kind: str = "unspecified",
        **kwargs: Any,
    ) -> MonitoredExecution:
        """Run an operation and return its output alongside application metrics.

        Set execution_kind explicitly (for example, ``mock`` or
        ``executorch_subprocess``). RAM values measure this Python process only;
        memory used by a subprocess is excluded.
        """
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not isinstance(execution_kind, str) or not execution_kind.strip():
            raise ValueError("execution_kind must be a non-empty string")

        model_file_size = self._get_model_file_size(model_path)
        memory_before = self._read_rss()
        peak_memory = [memory_before]
        sampling_errors = []
        stop_sampling = threading.Event()

        def sample_peak_memory() -> None:
            while not stop_sampling.wait(self.sample_interval_seconds):
                try:
                    peak_memory[0] = max(peak_memory[0], self._read_rss())
                except PerformanceMonitorError as error:
                    sampling_errors.append(error)
                    stop_sampling.set()
                    return

        sampler = threading.Thread(target=sample_peak_memory, daemon=True)
        sampler.start()
        started = time.perf_counter()
        output = None
        execution_error = None
        try:
            output = operation(*args, **kwargs)
        except Exception as error:
            execution_error = error
        duration = time.perf_counter() - started

        stop_sampling.set()
        sampler.join()
        memory_after = self._read_rss()
        if sampling_errors:
            raise PerformanceMonitorError(
                "Could not sample application process memory."
            ) from sampling_errors[0]
        peak_memory[0] = max(peak_memory[0], memory_after)

        metrics = PerformanceMetrics(
            execution_kind=execution_kind,
            duration_seconds=duration,
            application_ram_before_bytes=memory_before,
            application_ram_after_bytes=memory_after,
            application_ram_peak_bytes=peak_memory[0],
            model_file_size_bytes=model_file_size,
            error=(
                f"{type(execution_error).__name__}: {execution_error}"
                if execution_error is not None
                else None
            ),
        )

        try:
            self._write_metrics(metrics)
        except PerformanceMonitorError as log_error:
            if execution_error is not None:
                raise log_error from execution_error
            raise

        if execution_error is not None:
            raise execution_error
        return MonitoredExecution(output=output, metrics=metrics)

    def _read_rss(self) -> int:
        try:
            return self._process.memory_info().rss
        except psutil.Error as error:
            raise PerformanceMonitorError(
                "Could not read application process memory."
            ) from error

    @staticmethod
    def _get_model_file_size(model_path: Optional[Path]) -> Optional[int]:
        if model_path is None:
            return None
        try:
            path = Path(model_path)
            if not path.is_file():
                raise PerformanceMonitorError(
                    f"Model path is not a file: {path}"
                )
            return path.stat().st_size
        except (OSError, TypeError) as error:
            raise PerformanceMonitorError(
                f"Could not read model file at {model_path!s}."
            ) from error

    def _write_metrics(self, metrics: PerformanceMetrics) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **asdict(metrics),
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
        except OSError as error:
            raise PerformanceMonitorError(
                f"Could not write performance log at {self.log_path}."
            ) from error