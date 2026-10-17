#!/usr/bin/env python3
"""Replay local pipeline reference cases without providers, ASR, or source writes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from funclip.llm.chinese_converter import convert_srt_to_traditional
from funclip.subtitle_quality import build_srt_quality_report
from funclip.subtitle_meaning import check_meaning_preservation, MeaningPreservationError


def _select_cues(source, ids):
    blocks = re.split(r"(?:\r?\n)[ \t]*(?:\r?\n)+", source.lstrip("\ufeff"))
    selected = [b.strip("\r\n") for b in blocks
                if b.strip() and int(b.splitlines()[0]) in ids]
    if len(selected) != len(ids):
        raise ValueError("Reference cue selection is incomplete.")
    return "\n\n".join(selected) + "\n"


def evaluate(manifest, references_root=None):
    source_bytes = {}
    verified = {}
    if references_root is not None:
        references_root = references_root.resolve()
        for name, source in manifest["sources"].items():
            path = (references_root / source["path"]).resolve()
            if not path.is_relative_to(references_root):
                raise ValueError("Reference path escapes its read-only root.")
            original = path.read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            if digest != source["sha256"]:
                raise ValueError(f"Reference hash mismatch: {name}")
            source_bytes[name] = original.decode("utf-8")
            verified[name] = digest

    results = []
    for case in manifest["cases"]:
        source_id = case.get("source_id")
        if source_id in source_bytes:
            if _select_cues(source_bytes[source_id], case["source_cue_ids"]) != case["input_srt"]:
                raise ValueError(f"Embedded excerpt does not match reference: {case['id']}")
        expected_source = case.get("expected_source_id")
        if expected_source in source_bytes:
            if _select_cues(source_bytes[expected_source], case["source_cue_ids"]) != case["expected_srt"]:
                raise ValueError(f"Expected excerpt does not match reference: {case['id']}")
        for binding in case.get("reference_bindings", []):
            if binding["source_id"] in source_bytes:
                excerpt = _select_cues(source_bytes[binding["source_id"]], binding["cue_ids"])
                if excerpt != case[binding["srt_field"]]:
                    raise ValueError(f"Meaning excerpt does not match reference: {case['id']}")
        result = {"id": case["id"], "kind": case["kind"]}
        if case["kind"] == "quality":
            report = build_srt_quality_report(case["input_srt"])
            result.update(status="passed" if report["issue_counts"] == case["expected_issue_counts"] else "failed",
                          expected_issue_counts=case["expected_issue_counts"], quality_report=report)
        elif case["kind"] == "traditional":
            converted = convert_srt_to_traditional(case["input_srt"])
            result.update(status="passed" if converted == case["expected_srt"] else "failed",
                          expected_srt=case["expected_srt"], actual_srt=converted)
        elif case["kind"] == "meaning":
            issues = []
            try:
                check_meaning_preservation(case["source_cues"], case["candidate_cues"], mode=case["mode"])
            except MeaningPreservationError as exc:
                issues = exc.issues
            expected = set(case["expected_issue_codes"])
            actual = {issue["code"] for issue in issues}
            result.update(status="passed" if actual == expected else "failed",
                          expected_issue_codes=sorted(expected), actual_issues=issues)
        elif case["kind"] in {"deferred-semantic", "unresolved"}:
            result["status"] = "catalogued_not_scored"
        else:
            raise ValueError(f"Unknown reference case kind: {case['kind']}")
        results.append(result)
    return {"version": 1, "network_used": False, "source_writes": False,
            "verified_sources": verified, "content_accuracy_certified": False,
            "summary": dict(Counter(r["status"] for r in results)), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "eval/subtitle_pipeline_cases/manifest.json")
    parser.add_argument("--references-root", type=Path, help="Optional read-only archive root; verify pinned hashes and excerpts.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == args.manifest.resolve() or (
            args.references_root and output.is_relative_to(args.references_root.resolve())):
        parser.error("Report output must be outside the manifest and read-only reference archive.")
    data = args.manifest.read_bytes()
    try:
        report = evaluate(json.loads(data), args.references_root)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Reference evaluation failed: {exc}\n")
    report["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return int(report["summary"].get("failed", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
