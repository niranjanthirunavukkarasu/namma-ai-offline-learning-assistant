# Offline Smart Learning Assistant

A terminal-based Python application that accepts an educational passage, checks
its length, creates a structured prompt, and formats a summary and quiz from a
backend response. The repository currently ships with a mock backend for
development and testing. **Actual AI inference and ExecuTorch execution are not
implemented yet.**

## Features

- Accept a passage entered in the terminal or read from a UTF-8 text file.
- Reject empty passages and passages outside the inclusive 150–300 word range.
- Generate a JSON prompt containing a task and the original passage.
- Validate and display a response summary and at least three multiple-choice
  questions, each with four options and a correct answer.
- Record operation duration and current Python-process memory in a JSON Lines
  performance log.
- Keep the model backend replaceable through the `generate(prompt)` interface.

## Requirements

- Ubuntu or another Linux distribution.
- Python 3.8 or newer.
- `psutil>=5.9` (installed from `requirements.txt`).

The remaining imports use the Python standard library. ExecuTorch and a model
artifact are not currently project dependencies because the real runner
contract has not yet been supplied.

## Install On Ubuntu/Linux

Install Python and virtual-environment support if they are not already
available:

```bash
sudo apt update
sudo apt install python3 python3-venv
```

From the project directory, create and activate a virtual environment, then
install the declared dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run The Application

Run with manual passage input:

```bash
python main.py
```

Paste the passage and enter a blank line to finish. The passage must contain
150–300 whitespace-separated words.

Read a passage from a UTF-8 text file:

```bash
python main.py --file path/to/passage.txt
```

The current command-line application uses `MockModelBackend` by default. It
prints a mock notice and returns sample content that is not based on or evaluated
against the submitted passage. Its measured duration is labeled as mock
operation time, not AI inference time.

## Application Flow

```text
Manual or file input
  -> passage validation (150–300 words)
  -> JSON prompt generation
  -> ModelRunner backend
  -> response schema validation and terminal formatting
  -> performance report and logs/performance.log
```

`main.run_application()` also accepts an injected `ModelRunner`, monitor, input
function, and output stream for integration and testing. The normal `main.py`
CLI currently selects the mock backend only.

## Model Backend Contract

The replaceable interface is:

```python
class ModelBackend:
    def generate(self, prompt: str) -> str:
        ...
```

The `prompt` argument is a JSON string with these top-level fields:

```json
{
  "task": "Review the educational passage and return the requested result.",
  "passage": "The original passage text"
}
```

The backend must return a JSON string accepted by `output_formatter.py`. The
required structure is a non-empty `summary` and a `questions` list with at least
three entries. Each entry must contain a non-empty `question`, exactly four
non-empty string `options`, and a `correct_answer` matching one of the options.
The optional `mock: true` field is reserved for clearly identifying mock
responses; actual backends must not label real inference as mock.

Example response shape (illustrative data only, not a model inference result):

```json
{
  "summary": "Illustrative summary text.",
  "questions": [
    {
      "question": "Illustrative question?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A"
    },
    {
      "question": "Illustrative question?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option B"
    },
    {
      "question": "Illustrative question?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option C"
    }
  ]
}
```

### ExecuTorch Handover

`executorch_backend.py` currently provides configuration fields for a caller-
supplied command and model path, but its `generate()` method deliberately raises
`ExecuTorchIntegrationPendingError`. It does not launch a subprocess and does
not perform inference. Do not treat the configuration placeholder or fake
runner tests as an operational integration.

Before implementing the adapter, the model/runtime owner must supply and verify:

- Model identity and version, artifact format, and actual artifact location.
- Exact executable and argument contract, including how the model path is
  supplied. This project does not assume or prescribe a command.
- Whether the prompt is passed on stdin, as an argument, or through another
  mechanism.
- The stdout response contract, stderr behavior, exit-code meanings, and timeout
  requirements.
- Runtime/package requirements and a representative successful response that
  matches the schema above.

The application backend should capture stdout/stderr, enforce the agreed
timeout, and translate process failures into clear exceptions while preserving
the existing `generate(prompt) -> str` interface.

## Performance Measurements

The monitor measures the duration of `runner.run()` using
`time.perf_counter()`. RSS measurements cover the current Python process only;
child-process RAM is excluded. Peak RAM is sampled periodically and can miss
short-lived spikes. Model-file size is reported only when a path is supplied to
the monitor.

Records append as JSON Lines to `logs/performance.log` by default. The `logs/`
directory is ignored by Git. Mock operation duration must not be interpreted as
AI inference latency.

## Automated Tests

Activate the virtual environment and run the complete test suite:

```bash
python -m unittest -v
```

The audited suite contains **44 passing tests**, including module tests,
application workflows, validation and error cases, performance logging, and
fake-runner integration. Tests do not execute ExecuTorch or actual AI inference.
See [TEST_REPORT.md](TEST_REPORT.md) for end-to-end cases and results, and
[FINAL_AUDIT.md](FINAL_AUDIT.md) for the full technical audit.

## Project Structure

```text
.
├── main.py
├── passage_input.py
├── model_runner.py
├── executorch_backend.py
├── output_formatter.py
├── performance_monitor.py
├── requirements.txt
├── test_passage_input.py
├── test_model_runner.py
├── test_executorch_backend.py
├── test_output_formatter.py
├── test_performance_monitor.py
├── test_main.py
├── test_end_to_end.py
├── TEST_REPORT.md
├── FINAL_AUDIT.md
└── README.md
```

## GitHub Handover Notes

`.gitignore` excludes virtual environments, Python cache/bytecode, logs,
temporary/editor files, common local secret files, and common large model weight
formats. Keep private credentials and model artifacts out of source control.
The model artifact should be distributed through the team's approved storage
process, not committed to this repository.