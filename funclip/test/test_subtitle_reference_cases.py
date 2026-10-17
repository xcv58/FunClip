import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.eval_subtitle_pipeline import ROOT, evaluate, main


class SubtitleReferenceTests(unittest.TestCase):
    def test_unresolved_audio_and_future_semantics_are_not_scored_as_correct(self):
        manifest = json.loads((ROOT / "eval/subtitle_pipeline_cases/manifest.json").read_text())
        report = evaluate(manifest)
        self.assertEqual(report["summary"], {"passed": 35, "catalogued_not_scored": 3})
        self.assertFalse(report["content_accuracy_certified"])
        changed = copy.deepcopy(manifest)
        changed["cases"][0]["kind"] = "misspelled-quality"
        with self.assertRaisesRegex(ValueError, "Unknown reference case"):
            evaluate(changed)

    def test_changed_reference_hash_fails_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "final.srt"
            source.write_bytes(b"final reference")
            manifest = {"sources": {"final": {"path": source.name,
                        "sha256": hashlib.sha256(b"different reference").hexdigest()}}, "cases": []}
            with self.assertRaisesRegex(ValueError, "Reference hash mismatch"):
                evaluate(manifest, root)
            self.assertEqual(source.read_bytes(), b"final reference")

    def test_cli_rejects_report_output_inside_read_only_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "final.srt"
            source.write_bytes(b"final reference")
            argv = ["eval_subtitle_pipeline.py", "--references-root", str(root), "--output", str(source)]
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(source.read_bytes(), b"final reference")

    def test_historical_meaning_excerpt_must_match_pinned_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b"1\n00:00:01,000 --> 00:00:02,000\nprice 2910\n"
            source = root / "final.srt"
            source.write_bytes(original)
            manifest = {
                "sources": {"final": {"path": source.name, "sha256": hashlib.sha256(original).hexdigest()}},
                "cases": [{"id": "historical-price", "kind": "meaning", "reference_srt": original.decode().replace("2910", "291"),
                           "reference_bindings": [{"source_id": "final", "cue_ids": [1], "srt_field": "reference_srt"}]}],
            }
            with self.assertRaisesRegex(ValueError, "Meaning excerpt does not match reference"):
                evaluate(manifest, root)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
