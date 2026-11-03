import json
from typing import Protocol


class ModelBackend(Protocol):
    """Interface for a replaceable local model backend."""

    def generate(self, prompt: str) -> str:
        """Generate a response from a text prompt."""


class ModelRunnerError(RuntimeError):
    """Raised when a model backend fails or returns an unusable response."""


class MockModelBackend:
    """Return a clearly labeled sample response without running AI inference."""

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be text")

        return json.dumps(
            {
                "mock": True,
                "message": (
                    "Mock response only; no AI inference or passage analysis "
                    "was performed."
                ),
                "summary": "Mock summary only; passage content was not analyzed.",
                "questions": [
                    {
                        "question": f"Which option is the mock answer for question {index}?",
                        "options": [
                            "Mock option A",
                            "Mock option B",
                            "Mock option C",
                            "Mock option D",
                        ],
                        "correct_answer": "Mock option A",
                    }
                    for index in range(1, 4)
                ],
            }
        )


class ModelRunner:
    """Pass generated prompts to an injected backend and return its response."""

    def __init__(self, backend: ModelBackend):
        self._backend = backend

    def run(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be text")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        try:
            response = self._backend.generate(prompt)
        except Exception as error:
            raise ModelRunnerError("Model backend failed to generate a response.") from error

        if not isinstance(response, str):
            raise ModelRunnerError("Model backend returned a non-text response.")
        if not response.strip():
            raise ModelRunnerError("Model backend returned an empty response.")
        return response