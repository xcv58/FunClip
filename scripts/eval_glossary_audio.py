#!/usr/bin/env python3
"""Paired local ASR benchmark with fixed cached weights and untouched references."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "funclip")]
from funclip.glossary import resolve_glossary

MODEL_IDS = (
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--references-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()
    audio_root, references, out = args.audio_root.resolve(), args.references_root.resolve(), args.output.resolve()
    if out.is_relative_to(audio_root) or out.is_relative_to(references):
        parser.error("Benchmark output must be outside the read-only source directories.")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    finals = sorted(references.glob("*/01_成品字幕/*.srt"))
    final_before = {str(p.relative_to(references)): sha(p) for p in finals}
    if len(finals) != 24:
        raise ValueError("Expected the unchanged 24-file reference archive.")
    paths = [Path.home() / ".cache/modelscope/hub/models" / name for name in MODEL_IDS]
    if not all(p.is_dir() for p in paths):
        raise ValueError("Current ASR VAD and punctuation models must already be cached.")
    model_files = [f for p in paths for f in p.iterdir() if f.suffix in {".pt", ".json", ".yaml", ".mvn"} or f.name == "seg_dict"]
    model_hashes = {str(f): sha(f) for f in model_files}
    for case in manifest["cases"]:
        path = (audio_root / case["audio"]).resolve()
        if not path.is_relative_to(audio_root) or sha(path) != case["sha256"]:
            raise ValueError("Benchmark audio changed or escapes its read-only root.")
        resolve_glossary(case["profile"])
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from funasr import AutoModel
    import soundfile as sf
    from funclip.videoclipper import VideoClipper
    from funclip.service_contract import recognition_evidence
    from funclip.subtitle_style import SubtitleFormattingError
    from opencc import OpenCC
    normalizer = OpenCC("t2s")
    normalize = lambda t: "".join(normalizer.convert(t).casefold().split())
    model = AutoModel(model=str(paths[0]), vad_model=str(paths[1]), punc_model=str(paths[2]),
                      device=args.device, disable_update=True, trust_remote_code=False, log_level="ERROR", disable_pbar=True)
    clipper = VideoClipper(model)
    clipper.lang = "zh"
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "model_ids": MODEL_IDS,
                "model_sha256": model_hashes, "device": args.device, "local_audio_only": True,
                "network_requests_intended": 0, "case_count": len(manifest["cases"]),
                "manifest_sha256": sha(args.manifest), "code_sha256": {f: sha(ROOT / f) for f in ("funclip/glossary.py", "funclip/videoclipper.py")},
                "score_definition": "Exact accepted term presence per selected clip, not full-transcript accuracy. Unsupported spellings and unresolved dialogue are reported separately.",
                "final_srt_hashes_before": final_before}
    save(out / "metadata.json", metadata)
    results = []
    for index, case in enumerate(manifest["cases"]):
        wav, sr = sf.read(audio_root / case["audio"], dtype="float32")
        if sr != 16000 or wav.ndim != 1:
            raise ValueError("Benchmark requires the existing 16 kHz mono derivatives.")
        result = {"id": case["id"], "profile": case["profile"], "seconds": len(wav) / sr,
                  "expected_terms": case["expected_terms"], "unsupported_reference_terms": case.get("unsupported_reference_terms", []),
                  "negative_control": case.get("negative_control", False), "modes": {}}
        # Alternate order; both modes use the same model object and unmodified audio.
        for mode in (["baseline", "glossary"] if index % 2 == 0 else ["glossary", "baseline"]):
            started = time.perf_counter()
            try:
                text, srt, state = clipper.recog((sr, wav), glossary=case["profile"] if mode == "glossary" else None)
                formatting = "passed"
            except SubtitleFormattingError as exc:
                state = exc.recognition_state
                text, srt, formatting = state.get("recognition_text", ""), "", str(exc)
            evidence = recognition_evidence(text, srt, state)
            hits = {name: any(normalize(form) in normalize(text) for form in forms)
                    for name, forms in case["expected_terms"].items()}
            profile = resolve_glossary(case["profile"])
            mentioned = [e.id for e in profile.entries if any(normalize(f) in normalize(text) for f in e.forms)]
            result["modes"][mode] = {"text": text, "evidence": evidence, "term_hits": hits,
                "mentioned_profile_terms": mentioned, "elapsed_seconds": time.perf_counter() - started, "formatting": formatting}
            save(out / case["id"] / (mode + ".json"), result["modes"][mode])
            print(case["id"], mode, sum(hits.values()), "/", len(hits), flush=True)
        results.append(result)
        save(out / "partial_results.json", results)
    final_after = {str(p.relative_to(references)): sha(p) for p in finals}
    assert final_before == final_after
    assert model_hashes == {str(f): sha(f) for f in model_files}
    counts = {mode: {"term_clip_hits": sum(sum(r["modes"][mode]["term_hits"].values()) for r in results),
                     "term_clip_targets": sum(len(r["expected_terms"]) for r in results)} for mode in ("baseline", "glossary")}
    insertions = [{"id": r["id"], "new_terms": sorted(set(r["modes"]["glossary"]["mentioned_profile_terms"]) - set(r["modes"]["baseline"]["mentioned_profile_terms"]))}
                  for r in results if r["negative_control"]]
    summary = {"finished_at": datetime.now(timezone.utc).isoformat(), "case_count": len(results),
               "audio_seconds": sum(r["seconds"] for r in results), "scores": counts,
               "negative_control_new_term_insertions": insertions,
               "final_srt_files_unchanged": len(finals), "model_files_unchanged": True,
               "limits": manifest["limits"], "results": results}
    save(out / "results.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
