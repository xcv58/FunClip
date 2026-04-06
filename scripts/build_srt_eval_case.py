#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
FUNCLIP_DIR = REPO_ROOT / "funclip"

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

if str(FUNCLIP_DIR) not in sys.path:
    sys.path.append(str(FUNCLIP_DIR))


AUDIO_SUFFIXES = {".wav", ".mp3", ".aac", ".m4a", ".flac"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".flv", ".mov", ".webm", ".ts", ".mpeg"}


def sanitize_case_id(raw_value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw_value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "case"


def detect_media_mode(media_path: Path) -> str:
    suffix = media_path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise ValueError(f"Unsupported media file extension: {media_path.suffix}")


def build_model(lang: str, device: str | None):
    from funasr import AutoModel

    if lang == "zh":
        kwargs = {
            "model": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "vad_model": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "punc_model": "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            "spk_model": "damo/speech_campplus_sv_zh-cn_16k-common",
        }
    elif lang == "en":
        kwargs = {
            "model": "iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
            "vad_model": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "punc_model": "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            "spk_model": "damo/speech_campplus_sv_zh-cn_16k-common",
        }
    else:
        raise ValueError(f"Unsupported language: {lang}")

    if device:
        kwargs["device"] = device

    return AutoModel(**kwargs)


def generate_raw_srt(media_path: Path, lang: str, device: str | None) -> tuple[str, str]:
    from funclip.videoclipper import VideoClipper

    media_mode = detect_media_mode(media_path)
    model = build_model(lang=lang, device=device)
    clipper = VideoClipper(model)
    clipper.lang = lang

    if media_mode == "audio":
        import librosa

        wav, sample_rate = librosa.load(str(media_path), sr=16000)
        raw_text, raw_srt, _ = clipper.recog((sample_rate, wav), sd_switch="no")
        return raw_text, raw_srt

    with TemporaryDirectory() as temp_dir:
        raw_text, raw_srt, _ = clipper.video_recog(
            str(media_path),
            sd_switch="no",
            output_dir=temp_dir,
        )
    return raw_text, raw_srt


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one SRT auto-correction eval case from a media file and a final "
            "human-approved SRT file."
        )
    )
    parser.add_argument("--media", required=True, help="Path to an audio or video file.")
    parser.add_argument("--gold-srt", required=True, help="Path to the final corrected SRT file.")
    parser.add_argument(
        "--case-id",
        help="Optional case id. Defaults to the media filename stem.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "eval" / "srt_correction_cases"),
        help="Directory where eval cases are stored.",
    )
    parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        default="zh",
        help="Recognition language.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional FunASR device override, e.g. mps or cuda:0.",
    )
    parser.add_argument(
        "--copy-media",
        action="store_true",
        help="Copy the source media file into the case directory for later re-runs.",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Regenerate input.srt even if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = load_args()
    media_path = Path(args.media).expanduser().resolve()
    gold_srt_path = Path(args.gold_srt).expanduser().resolve()

    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")
    if not gold_srt_path.exists():
        raise FileNotFoundError(f"Gold SRT file not found: {gold_srt_path}")

    case_id = sanitize_case_id(args.case_id or media_path.stem)
    case_dir = Path(args.output_root).expanduser().resolve() / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    input_srt_path = case_dir / "original_transcribed.srt"
    gold_out_path = case_dir / "final_traditional.srt"
    raw_text_path = case_dir / "raw_text.txt"
    metadata_path = case_dir / "meta.json"

    generated_input = False
    if args.force_transcribe or not input_srt_path.exists():
        raw_text, raw_srt = generate_raw_srt(
            media_path=media_path,
            lang=args.lang,
            device=args.device,
        )
        write_text_file(input_srt_path, raw_srt)
        write_text_file(raw_text_path, raw_text or "")
        generated_input = True

    shutil.copy2(gold_srt_path, gold_out_path)

    media_copy_path = None
    if args.copy_media:
        media_copy_path = case_dir / f"source{media_path.suffix.lower()}"
        shutil.copy2(media_path, media_copy_path)

    metadata = {
        "case_id": case_id,
        "lang": args.lang,
        "media_mode": detect_media_mode(media_path),
        "source_media_path": str(media_path),
        "source_gold_srt_path": str(gold_srt_path),
        "copied_media_path": str(media_copy_path) if media_copy_path else None,
        "generated_input_srt": generated_input,
        "input_srt_path": str(input_srt_path),
        "gold_srt_path": str(gold_out_path),
        "raw_text_path": str(raw_text_path) if raw_text_path.exists() else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asr_device": args.device,
    }
    write_text_file(metadata_path, json.dumps(metadata, ensure_ascii=True, indent=2) + "\n")

    print(f"case_dir={case_dir}")
    print(f"input_srt={input_srt_path}")
    print(f"gold_srt={gold_out_path}")
    print(f"meta={metadata_path}")
    if raw_text_path.exists():
        print(f"raw_text={raw_text_path}")
    if media_copy_path is not None:
        print(f"media_copy={media_copy_path}")
    if not generated_input:
        print("reused_cached_input=true")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
