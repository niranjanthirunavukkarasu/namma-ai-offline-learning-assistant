import io
import json
import unittest

from model_runner import ModelRunner, MockModelBackend
from output_formatter import OutputFormatError, display_response, format_response


def valid_response():
    return json.dumps(
        {
            "summary": "Plants use sunlight to make sugars from water and carbon dioxide.",
            "questions": [
                {
                    "question": f"Which option is correct for question {index}?",
                    "options": ["Answer A", "Answer B", "Answer C", "Answer D"],
                    "correct_answer": "Answer B",
                }
                for index in range(1, 4)
            ],
        }
    )


class OutputFormatterTests(unittest.TestCase):
    def test_formats_summary_questions_options_and_answers(self):
        formatted = format_response(valid_response())

        self.assertIn("SUMMARY", formatted)
        self.assertIn("Plants use sunlight", formatted)
        self.assertEqual(formatted.count("Correct answer: B. Answer B"), 3)
        self.assertIn("   D. Answer D", formatted)

    def test_displays_formatted_response_in_terminal_stream(self):
        stream = io.StringIO()

        display_response(valid_response(), stream=stream)

        self.assertIn("SUMMARY", stream.getvalue())
        self.assertIn("QUIZ", stream.getvalue())

    def test_formats_mock_response_with_clear_notice(self):
        response = ModelRunner(MockModelBackend()).run("test prompt")

        formatted = format_response(response)

        self.assertIn("MOCK RESPONSE: no AI inference was performed.", formatted)
        self.assertIn("Mock summary only", formatted)

    def test_rejects_missing_summary(self):
        data = json.loads(valid_response())
        del data["summary"]

        with self.assertRaisesRegex(OutputFormatError, "summary"):
            format_response(json.dumps(data))

    def test_rejects_fewer_than_three_questions(self):
        data = json.loads(valid_response())
        data["questions"] = data["questions"][:2]

        with self.assertRaisesRegex(OutputFormatError, "at least three"):
            format_response(json.dumps(data))

    def test_rejects_invalid_json(self):
        with self.assertRaisesRegex(OutputFormatError, "valid JSON"):
            format_response("{not json")

    def test_rejects_missing_answer_options(self):
        data = json.loads(valid_response())
        data["questions"][0]["options"] = ["Only one option"]

        with self.assertRaisesRegex(OutputFormatError, "four"):
            format_response(json.dumps(data))

    def test_rejects_missing_correct_answer(self):
        data = json.loads(valid_response())
        del data["questions"][0]["correct_answer"]

        with self.assertRaisesRegex(OutputFormatError, "correct_answer"):
            format_response(json.dumps(data))


if __name__ == "__main__":
    unittest.main()