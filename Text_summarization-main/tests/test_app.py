import unittest

from backend.app import app


class SummarizeApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_rejects_unknown_model(self):
        response = self.client.post(
            "/summarize",
            json={"text": "A valid sentence.", "model_type": "unknown"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Model type", response.get_json()["error"])

    def test_rejects_text_removed_by_normalization(self):
        response = self.client.post("/summarize", json={"text": "um uh hmm"})
        self.assertEqual(response.status_code, 400)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
