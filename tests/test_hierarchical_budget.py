import unittest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from backend.hierarchical_summarizer import (
    calculate_word_budget,
    clean_and_filter_narrative,
    words_to_token_budget,
    enforce_word_budget,
)


class TestHierarchicalBudget(unittest.TestCase):
    def test_clean_and_filter_narrative(self):
        dirty = (
            "Speaker 1: [00:15] Good morning everyone.\n"
            "Interviewer: (12:34:56) Let us begin the discussion today.\n"
            "- First point about AI technology.\n"
            "• Second point about machine learning."
        )
        cleaned = clean_and_filter_narrative(dirty)
        self.assertNotIn("Speaker 1:", cleaned)
        self.assertNotIn("[00:15]", cleaned)
        self.assertNotIn("(12:34:56)", cleaned)
        self.assertIn("Good morning everyone.", cleaned)
        self.assertIn("First point about AI technology.", cleaned)

    def test_calculate_word_budget_edge_cases(self):
        # 1. Extreme small input (<= 20 words)
        # short: max=45%, min=25%
        min_w, max_w = calculate_word_budget(20, "short")
        self.assertEqual(min_w, 5)
        self.assertEqual(max_w, 9)

        # 2. Short input (<= 60 words): max >= 10, min >= 5
        min_w, max_w = calculate_word_budget(50, "short")
        self.assertGreaterEqual(max_w, 10)
        self.assertGreaterEqual(min_w, 5)
        self.assertLessEqual(min_w, max_w)

        # 3. Standard input (> 60 words): short max=45%, min=25%, cap=250
        min_w, max_w = calculate_word_budget(200, "short")
        self.assertEqual(min_w, 50)  # 200 * 0.25 = 50
        self.assertEqual(max_w, 90)  # 200 * 0.45 = 90

        # medium: max=60%, min=45%, cap=450
        min_w, max_w = calculate_word_budget(200, "medium")
        self.assertEqual(min_w, 90)   # 200 * 0.45 = 90
        self.assertEqual(max_w, 120)  # 200 * 0.60 = 120

        # long: max=80%, min=75%, cap=800
        min_w, max_w = calculate_word_budget(200, "long")
        self.assertEqual(min_w, 150)  # 200 * 0.75 = 150
        self.assertEqual(max_w, 160)  # 200 * 0.80 = 160

        # Large input exceeding caps
        min_w, max_w = calculate_word_budget(2000, "short")
        self.assertEqual(max_w, 250)  # paragraph cap 250

        min_w, max_w = calculate_word_budget(2000, "medium")
        self.assertEqual(max_w, 450)  # paragraph cap 450

        min_w, max_w = calculate_word_budget(2000, "long")
        self.assertEqual(max_w, 800)  # paragraph cap 800

    def test_words_to_token_budget(self):
        min_tok, max_tok = words_to_token_budget(50, 90, max_token_cap=360)
        self.assertGreaterEqual(min_tok, 14)
        self.assertLessEqual(max_tok, 360)
        self.assertLess(min_tok, max_tok)

    def test_enforce_word_budget(self):
        text = (
            "Sentence one is concise and clear. "
            "Sentence two provides additional detailed context for the reader. "
            "Sentence three adds even more information to complete the paragraph."
        )
        # Sentence 1 has 6 words. Sentence 1+2 has 15 words. Sentence 1+2+3 has 25 words.
        # Enforcing max_words = 16 should keep sentences 1 and 2.
        enforced = enforce_word_budget(text, 16)
        self.assertEqual(
            enforced,
            "Sentence one is concise and clear. Sentence two provides additional detailed context for the reader."
        )


if __name__ == "__main__":
    unittest.main()
