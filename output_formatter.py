import json
import sys
from typing import TextIO


class OutputFormatError(ValueError):
    """Raised when a model response is malformed or incomplete."""


def format_response(response: str) -> str:
    """Validate a model response and format its summary and quiz for a terminal."""
    if not isinstance(response, str):
        raise TypeError("model response must be text")

    try:
        data = json.loads(response)
    except json.JSONDecodeError as error:
        raise OutputFormatError("Model response is not valid JSON.") from error

    if not isinstance(data, dict):
        raise OutputFormatError("Model response must be a JSON object.")

    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise OutputFormatError("Model response is missing a non-empty summary.")

    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) < 3:
        raise OutputFormatError("Model response must contain at least three questions.")

    validated_questions = []
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise OutputFormatError(f"Question {index} must be a JSON object.")

        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise OutputFormatError(f"Question {index} is missing question text.")

        options = item.get("options")
        if (
            not isinstance(options, list)
            or len(options) != 4
            or any(not isinstance(option, str) or not option.strip() for option in options)
        ):
            raise OutputFormatError(
                f"Question {index} must have exactly four non-empty answer options."
            )

        correct_answer = item.get("correct_answer")
        if not isinstance(correct_answer, str) or correct_answer not in options:
            raise OutputFormatError(
                f"Question {index} must have a correct_answer matching one option."
            )

        validated_questions.append((question, options, correct_answer))

    lines = []
    if data.get("mock") is True:
        lines.extend(
            [
                "MOCK RESPONSE: no AI inference was performed.",
                "",
            ]
        )
    lines.extend(["SUMMARY", "-------", summary.strip(), "", "QUIZ", "----"])

    option_labels = "ABCD"
    for index, (question, options, correct_answer) in enumerate(
        validated_questions, start=1
    ):
        lines.append(f"{index}. {question.strip()}")
        for label, option in zip(option_labels, options):
            lines.append(f"   {label}. {option.strip()}")
        answer_label = option_labels[options.index(correct_answer)]
        lines.append(f"   Correct answer: {answer_label}. {correct_answer.strip()}")
        lines.append("")

    return "\n".join(lines).rstrip()


def display_response(response: str, stream: TextIO = sys.stdout) -> None:
    """Print a validated, readable response to the selected terminal stream."""
    print(format_response(response), file=stream)