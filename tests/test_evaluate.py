import unittest
from pathlib import Path

import fitz

from eval.evaluate import check_hit, load_questions, load_refusal_questions, threshold_failures


class EvaluationDatasetTests(unittest.TestCase):
    def test_answerable_questions_are_grounded_in_their_source_pdfs(self):
        questions = load_questions()
        documents_dir = Path(__file__).resolve().parents[1] / "documents"

        self.assertEqual(len(questions), 20)
        for question in questions:
            self.assertTrue(question["answerable"])
            source_path = documents_dir / question["source_file"]
            with fitz.open(source_path) as document:
                text = " ".join(" ".join(page.get_text().lower().split()) for page in document)
            for keyword in question["expected_keywords"]:
                self.assertIn(" ".join(keyword.lower().split()), text, question["question"])

    def test_refusal_questions_are_separate_and_unanswerable(self):
        questions = load_refusal_questions()

        self.assertEqual(len(questions), 10)
        self.assertTrue(all(not question["answerable"] for question in questions))
        self.assertTrue(all(not question["expected_keywords"] for question in questions))

    def test_hit_requires_all_keywords_in_one_chunk(self):
        chunks = [{"text": "The required attendance is 75 percent."}]

        self.assertTrue(check_hit(chunks, ["attendance", "75 percent"]))
        self.assertFalse(check_hit(chunks, ["attendance", "80 percent"]))

    def test_threshold_failure_is_reported(self):
        failures = threshold_failures(60, 90, 70, 100)

        self.assertEqual(len(failures), 2)


if __name__ == "__main__":
    unittest.main()