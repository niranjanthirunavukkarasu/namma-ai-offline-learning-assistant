import tempfile
import unittest
import json
from pathlib import Path

from passage_input import validate_passage, validate_passage_file


class PassageInputTests(unittest.TestCase):
    def test_validate_passage_sends_model_instructions(self):
        passage = "  " + "Photosynthesis converts light into chemical energy. " * 30 + "\n"
        received = []

        result = validate_passage(passage, lambda text: received.append(text) or "ok")

        instructions = json.loads(received[0])
        self.assertEqual(instructions["passage"], passage)
        self.assertIn("Review the educational passage", instructions["task"])
        self.assertEqual(result, "ok")

    def test_validate_passage_file_passes_file_contents(self):
        passage = "Water moves through evaporation, condensation, and precipitation. " * 30
        received = []

        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "passage.txt"
            file_path.write_text(passage, encoding="utf-8")

            result = validate_passage_file(
                file_path, lambda text: received.append(text) or len(text)
            )

        instructions = json.loads(received[0])
        self.assertEqual(instructions["passage"], passage)
        self.assertEqual(result, len(received[0]))

    def test_validate_passage_rejects_empty_or_whitespace_input(self):
        for passage in ("", " \t\n "):
            with self.subTest(passage=repr(passage)):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    validate_passage(passage, lambda text: text)

    def test_validate_passage_enforces_inclusive_word_limit(self):
        for word_count in (149, 301):
            with self.subTest(word_count=word_count):
                passage = "word " * word_count
                with self.assertRaisesRegex(ValueError, f"got {word_count}"):
                    validate_passage(passage, lambda text: text)

        for word_count in (150, 300):
            with self.subTest(word_count=word_count):
                passage = "word " * word_count
                self.assertEqual(
                    json.loads(validate_passage(passage, lambda text: text))["passage"],
                    passage,
                )


if __name__ == "__main__":
    unittest.main()