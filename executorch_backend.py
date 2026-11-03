from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


class ExecuTorchIntegrationPendingError(RuntimeError):
    """Raised until the ExecuTorch runner's actual invocation contract is known."""


@dataclass(frozen=True)
class ExecuTorchConfig:
    """Caller-supplied runner settings; values are not interpreted or executed."""

    runner_command: Tuple[str, ...]
    model_path: Path

    def __post_init__(self) -> None:
        if not self.runner_command or any(
            not isinstance(part, str) or not part.strip()
            for part in self.runner_command
        ):
            raise ValueError("runner_command must contain non-empty command parts")
        if not isinstance(self.model_path, Path):
            raise TypeError("model_path must be a pathlib.Path")


class ExecuTorchBackend:
    """Reserved adapter for a future runner, implementing the app's backend shape.

    The command and model path are configurable, but generate() intentionally
    remains disabled until the actual runner interface is provided.
    """

    def __init__(self, config: ExecuTorchConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        raise ExecuTorchIntegrationPendingError(
            "ExecuTorch runner interface is not supplied; no command was executed."
        )