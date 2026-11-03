import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, TextIO

from model_runner import ModelRunner, ModelRunnerError, MockModelBackend
from output_formatter import OutputFormatError, display_response
from passage_input import validate_passage, validate_passage_file
from performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
    PerformanceMonitorError,
)


def _read_manual_passage(input_fn: Callable[[], str], stream: TextIO) -> str:
    print("Paste the educational passage, then enter a blank line:", file=stream)
    lines = []
    while True:
        try:
            line = input_fn()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


def _display_performance_report(
    metrics: PerformanceMetrics, stream: TextIO
) -> None:
    print("\nPERFORMANCE REPORT", file=stream)
    print(f"Execution kind: {metrics.execution_kind}", file=stream)
    if metrics.execution_kind == "mock":
        duration_label = "Operation duration (mock; not AI inference)"
    else:
        duration_label = "Operation duration"
    print(f"{duration_label}: {metrics.duration_seconds:.6f} seconds", file=stream)
    print(
        "Application RAM (this process): "
        f"before {metrics.application_ram_before_bytes} bytes, "
        f"after {metrics.application_ram_after_bytes} bytes, "
        f"peak {metrics.application_ram_peak_bytes} bytes",
        file=stream,
    )
    if metrics.model_file_size_bytes is None:
        print("Model file size: not provided", file=stream)
    else:
        print(f"Model file size: {metrics.model_file_size_bytes} bytes", file=stream)
    print(f"Memory scope: {metrics.memory_scope}", file=stream)


def run_application(
    passage: Optional[str] = None,
    file_path: Optional[Path] = None,
    *,
    model_runner: Optional[ModelRunner] = None,
    performance_monitor: Optional[PerformanceMonitor] = None,
    execution_kind: Optional[str] = None,
    input_fn: Optional[Callable[[], str]] = None,
    output_stream: Optional[TextIO] = None,
) -> int:
    """Run the terminal workflow and return zero on success, one on failure."""
    stream = output_stream if output_stream is not None else sys.stdout
    is_default_mock = model_runner is None
    runner = model_runner or ModelRunner(MockModelBackend())
    monitor = performance_monitor or PerformanceMonitor()
    run_kind = execution_kind or ("mock" if is_default_mock else "unspecified")
    read_input = input_fn if input_fn is not None else input
    measured_metrics = None

    def run_model(prompt: str) -> str:
        nonlocal measured_metrics
        measured = monitor.execute(
            runner.run,
            prompt,
            execution_kind=run_kind,
        )
        measured_metrics = measured.metrics
        return measured.output

    print("Offline Smart Learning Assistant", file=stream)
    if is_default_mock:
        print("Using mock backend; no AI inference will be performed.", file=stream)

    try:
        if passage is not None and file_path is not None:
            raise ValueError("provide passage text or a file path, not both")

        if file_path is not None:
            print(f"Reading passage from {file_path}...", file=stream)
            response = validate_passage_file(file_path, run_model)
        else:
            if passage is None:
                passage = _read_manual_passage(read_input, stream)
            print("Validating passage...", file=stream)
            response = validate_passage(passage, run_model)

        display_response(response, stream=stream)
        if measured_metrics is not None:
            _display_performance_report(measured_metrics, stream)
        return 0
    except OutputFormatError as error:
        print(f"Could not process model response: {error}", file=stream)
        if measured_metrics is not None:
            _display_performance_report(measured_metrics, stream)
        return 1
    except ModelRunnerError as error:
        print(f"Model execution failed: {error}", file=stream)
        print(f"Performance record: {monitor.log_path}", file=stream)
        return 1
    except PerformanceMonitorError as error:
        print(f"Performance monitoring failed: {error}", file=stream)
        return 1
    except OSError as error:
        print(f"Could not read passage file: {error}", file=stream)
        return 1
    except UnicodeError as error:
        print(f"Could not decode passage file as UTF-8: {error}", file=stream)
        return 1
    except (TypeError, ValueError) as error:
        print(f"Invalid passage: {error}", file=stream)
        return 1


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize an educational passage and create a quiz."
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="read passage text from a UTF-8 text file instead of manual input",
    )
    arguments = parser.parse_args(argv)
    return run_application(file_path=arguments.file)


if __name__ == "__main__":
    raise SystemExit(main())