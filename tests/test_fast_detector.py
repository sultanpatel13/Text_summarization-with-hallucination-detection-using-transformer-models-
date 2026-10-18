import unittest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from backend.hallucination_detector import HallucinationDetector


class TestFastDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = HallucinationDetector()
        cls.source_text = (
            "Renewable energy technologies such as solar and wind power are expanding globally. "
            "Solar photovoltaic panels convert sunlight directly into electricity with high efficiency."
        )
        cls.summary = (
            "Solar panels convert sunlight directly into electricity. "
            "Renewable energy technologies are expanding around the world."
        )

    def test_01_predict_nli_batch(self):
        pairs = [
            ("Solar panels convert sunlight into electricity.", "Solar panels generate electricity from sun."),
            ("The sky is blue.", "The sky is green and made of cheese."),
        ]
        results = self.detector._predict_nli_batch(pairs)
        self.assertEqual(len(results), 2)
        # Pair 1 should be strongly entailed
        self.assertGreater(results[0]["ENTAILMENT"], results[0]["CONTRADICTION"])
        # Pair 2 should be strongly contradicted
        self.assertGreater(results[1]["CONTRADICTION"], results[1]["ENTAILMENT"])

    def test_02_analyze_summary_optimized(self):
        analysis = self.detector.analyze_summary(self.source_text, self.summary)
        self.assertIn("overall_label", analysis)
        self.assertIn("sentence_analysis", analysis)
        self.assertGreater(len(analysis["sentence_analysis"]), 0)
        self.assertEqual(analysis["overall_label"], "SUPPORTED")
        self.assertEqual(analysis["supported_claims"], 2)
        self.assertEqual(analysis["contradicted_claims"], 0)


if __name__ == "__main__":
    unittest.main()
