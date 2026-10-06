"""Prepared-video lookup must never guess from a similar title."""
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_backend import DemoBackend  # noqa: E402


class DemoBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "video.mp4").write_bytes(b"sample")
        self.manifest = self.root / "cache.json"
        self.write_manifest([{"demo_id": "specific", "title": "Salt analysis", "experiment_id": "chem_12_301",
                              "path": "video.mp4", "review_status": "needs_visual_review"}])

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, videos):
        self.manifest.write_text(json.dumps({"videos": videos}), encoding="utf-8")

    def test_exact_lookup_and_missing_file(self):
        backend = DemoBackend(self.root, self.manifest)
        self.assertTrue(backend.video_for_experiment("chem_12_301")["available"])
        self.assertIsNone(backend.video_for_experiment("chem_12_302"))
        (self.root / "video.mp4").unlink()
        self.assertFalse(backend.video_for_experiment("chem_12_301")["available"])
        self.assertIsNone(backend.video_path("specific"))

    def test_rejects_duplicate_experiment_and_path_escape(self):
        entry = {"demo_id": "a", "experiment_id": "chem_9_001", "path": "video.mp4"}
        self.write_manifest([entry, {**entry, "demo_id": "b"}])
        with self.assertRaisesRegex(ValueError, "Duplicate experiment_id"):
            DemoBackend(self.root, self.manifest)
        self.write_manifest([{**entry, "path": "../outside.mp4"}])
        with self.assertRaisesRegex(ValueError, "Unsafe video path"):
            DemoBackend(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
