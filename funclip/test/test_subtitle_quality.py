import hashlib
import json
import unittest

from funclip.subtitle_quality import build_srt_quality_report, quality_status_suffix


def srt(text, start="00:00:00,000", end="00:00:04,000", cue_id=1):
    return f"{cue_id}\n{start} --> {end}\n{text}\n"


class SubtitleQualityTests(unittest.TestCase):
    def test_exact_millisecond_boundaries_and_original_bytes_survive(self):
        source = "\ufeff" + srt("你好，红𫊻！", end="00:00:00,700").replace("\n", "\r\n")
        original = source.encode()
        report = build_srt_quality_report(source)
        self.assertEqual(report["issues"], [])
        self.assertFalse(report["content_verified"])
        self.assertEqual(report["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(source.encode(), original)
        short = build_srt_quality_report(srt("对", end="00:00:00,699"))
        self.assertEqual(short["issue_counts"], {"short_cue": 1})

    def test_long_sparse_caption_reports_real_interval_without_splitting(self):
        source = srt("七龙珠之长湖夏日花火大一年级", "00:06:44,320", "00:10:05,130", 125)
        report = build_srt_quality_report(source)
        self.assertEqual(report["issue_counts"], {"long_cue": 1, "sparse_long_cue": 1})
        self.assertEqual(report["cue_count"], 1)
        self.assertEqual(report["issues"][0]["start_ms"], 404320)
        self.assertEqual(report["issues"][0]["end_ms"], 605130)

    def test_invalid_cues_overlap_and_media_bounds_are_diagnostics(self):
        source = (srt("第一句") + "\n" + srt("第二句", "00:00:03,000", "00:00:06,000", 1)
                  + "\n" + srt("倒退", "00:00:02,000", "00:00:02,000", 3)
                  + "\n4\ninvalid timing\n原文\n")
        report = build_srt_quality_report(source, media_duration_seconds=5)
        for code in ("duplicate_cue_id", "overlap", "outside_media", "out_of_order", "nonpositive_duration", "malformed_cue"):
            self.assertIn(code, report["issue_counts"])
        self.assertEqual(report["status"], "needs_review")
        json.dumps(report, allow_nan=False)

    def test_repetition_warns_but_ordinary_long_gap_does_not(self):
        self.assertEqual(build_srt_quality_report(srt("在这个地方" * 5))["issue_counts"],
                         {"repeated_text": 1, "chinese_line_length": 1})
        captions = "\n".join(srt("谢谢", f"00:00:0{i},000", f"00:00:0{i+1},000", i+1) for i in range(3))
        self.assertEqual(build_srt_quality_report(captions)["issue_counts"], {"repeated_adjacent_cues": 1})
        gap = srt("谢谢") + "\n" + srt("谢谢", "00:09:00,000", "00:09:04,000", 2)
        gap += "\n" + srt("谢谢", "00:19:00,000", "00:19:04,000", 3)
        self.assertEqual(build_srt_quality_report(gap)["issues"], [])

    def test_empty_recognition_and_invalid_media_duration(self):
        self.assertEqual(build_srt_quality_report("")["issues"], [])
        for value in (True, -1, float("nan"), float("inf"), "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_srt_quality_report(srt("字幕"), media_duration_seconds=value)

    def test_ui_hint_is_bounded_and_absent_when_clear(self):
        clear = build_srt_quality_report(srt("正常字幕"))
        self.assertEqual(quality_status_suffix([clear]), "")
        repeated = build_srt_quality_report(srt("重复" * 10, end="00:00:00,300"))
        hint = quality_status_suffix([repeated] * 100)
        self.assertIn("Review suggested", hint)
        self.assertLess(len(hint), 600)
        repeated["issues"][0]["cue_ids"] = list(range(1, 3001))
        self.assertLess(len(quality_status_suffix([repeated])), 600)


if __name__ == "__main__":
    unittest.main()
