import os
import sys
import unittest
from pathlib import Path

# Add project root and backend to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

import torch
from backend.app import app
from backend.device_utils import (
    clear_device_cache,
    configure_torch_threads,
    get_device,
    get_device_info,
    is_cpu_forced,
)


class TestDeviceSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    def test_01_get_device_returns_valid_string(self):
        device = get_device()
        self.assertIn(device, ["cuda", "cpu"])
        if not torch.cuda.is_available():
            self.assertEqual(device, "cpu")

    def test_02_get_device_info_metadata(self):
        info = get_device_info()
        self.assertIn("active_device", info)
        self.assertIn("is_gpu", info)
        self.assertIn("cpu_count", info)
        self.assertIn("cuda_available", info)
        self.assertIsInstance(info["active_device"], str)
        self.assertIsInstance(info["is_gpu"], bool)
        self.assertGreaterEqual(info["cpu_count"], 1)

    def test_03_force_cpu_flag(self):
        original = os.environ.get("FORCE_CPU")
        try:
            os.environ["FORCE_CPU"] = "1"
            self.assertTrue(is_cpu_forced())
            self.assertEqual(get_device(), "cpu")

            os.environ["FORCE_CPU"] = "true"
            self.assertTrue(is_cpu_forced())
            self.assertEqual(get_device(), "cpu")

            os.environ["FORCE_CPU"] = "0"
            self.assertFalse(is_cpu_forced())
        finally:
            if original is not None:
                os.environ["FORCE_CPU"] = original
            else:
                os.environ.pop("FORCE_CPU", None)

    def test_04_configure_torch_threads(self):
        # Should execute safely without raising exceptions
        configure_torch_threads()
        threads = torch.get_num_threads()
        self.assertGreaterEqual(threads, 1)

    def test_05_clear_device_cache(self):
        # Should execute safely on both CPU and GPU
        clear_device_cache("cpu")
        clear_device_cache("cuda")

    def test_06_device_info_endpoint(self):
        response = self.client.get("/device_info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("active_device", data)
        self.assertIn("cpu_count", data)

        # Also test /api/device_info alias
        res_alias = self.client.get("/api/device_info")
        self.assertEqual(res_alias.status_code, 200)
        self.assertEqual(res_alias.get_json()["active_device"], data["active_device"])

    def test_07_offline_mode_environment(self):
        """Verify that offline environment variables are configured when requested."""
        from backend.device_utils import configure_offline_mode
        configure_offline_mode(force=True)
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
        # Clean up so test runner is not impacted
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    def test_08_load_pretrained_safe_offline(self):
        """Verify load_pretrained_safe loads cached components strictly offline."""
        from backend.device_utils import load_pretrained_safe
        from transformers import AutoTokenizer

        tokenizer = load_pretrained_safe(
            AutoTokenizer.from_pretrained,
            "cross-encoder/nli-deberta-v3-base"
        )
        self.assertIsNotNone(tokenizer)
        tokens = tokenizer.encode("Testing offline inference.")
        self.assertGreater(len(tokens), 0)


if __name__ == "__main__":
    unittest.main()
