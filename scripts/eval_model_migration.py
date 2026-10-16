#!/usr/bin/env python3
"""Resume a small, live comparison through FunClip's real LLM functions."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import litellm
from funclip.llm import srt_corrector, srt_translator, youtube_chapters
from funclip.llm.chinese_converter import normalize_traditional_for_validation
from scripts.eval_srt_correction import (
    canonicalize_reading_text,
    compute_metrics,
    compute_structural_metrics,
    flatten_srt_text,
    split_srt_blocks,
)

litellm.suppress_debug_info = True
CONTEXT = threading.local()
# Standard text rates per million tokens, checked against official model pages.
RATES = {"gpt-4o-mini": (0.15, 0.075, 0.60), "gpt-6-luna": (0.10, 0.01, 0.50)}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def safe_error(exc):
    message = str(exc)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return {"type": type(exc).__name__, "http_status": getattr(exc, "status_code", None), "message": message[:1500]}


def measured_completion(**kwargs):
    """Record timing/usage while passing every request to the real provider."""
    started = time.perf_counter()
    call = {
        "model": kwargs["model"],
        "reasoning_effort": kwargs.get("reasoning_effort"),
        "max_tokens": kwargs.get("max_tokens"),
        "max_completion_tokens": kwargs.get("max_completion_tokens"),
        "messages_sha256": hashlib.sha256(json.dumps(kwargs["messages"], ensure_ascii=False).encode()).hexdigest(),
    }
    try:
        response = litellm.completion(**kwargs)
        call.update(
            resolved_model=response.model,
            response_id=response.id,
            usage=srt_corrector.make_json_safe(response.usage),
            finish_reason=response.choices[0].finish_reason,
            response_content=response.choices[0].message.content,
        )
        return response
    except Exception as exc:
        call["error"] = safe_error(exc)
        raise
    finally:
        call["latency_seconds"] = time.perf_counter() - started
        CONTEXT.calls.append(call)


def make_srt(lines, cue_seconds=4):
    def timestamp(seconds):
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d},000"
    return "\n\n".join(
        f"{i}\n{timestamp((i-1)*cue_seconds)} --> {timestamp(i*cue_seconds)}\n{text}"
        for i, text in enumerate(lines, 1)
    )


def jobs_for_models(models, out, synthetic_only=False, correction_repeats=1):
    cases = ROOT / "eval/srt_correction_cases"
    clean = make_srt([
        "今天去澎湖看花火節。", "我用 iPhone 拍攝，再用 Final Cut Pro 剪輯。",
        "這家店叫阿昇，朋友阿珩也會來。", "我們在觀音亭休憩區集合。",
        "API、FIRE 和 ETF 都是這集會提到的詞。", "他說得很快，我沒有聽清楚。",
    ])
    errors = make_srt([
        "今天的天汽很好。", "我们明天坐飞鸡去澎湖。",
        "我用 iPhnoe 拍摄了澎湖花火节。", "这家店叫阿昇，朋友阿珩也会来。",
    ])
    error_gold = errors.replace("天汽", "天气").replace("飞鸡", "飞机").replace("iPhnoe", "iPhone")
    travel = make_srt([
        "今天我們來到澎湖，準備參加澎湖國際海上花火節。",
        "煙火在觀音亭休憩區施放，我們提早到現場找位置。",
        "我用 iPhone 拍攝，再用 Final Cut Pro 剪輯。",
        "這家店叫阿昇，請保留店名，不要換成別的名字。",
        "明天要搭船去七美，今天先確認船班和天氣。",
        "如果風浪太大，我們就留在馬公市區走走。",
    ])
    short_travel = make_srt([
        "今天來到澎湖。", "晚上去看花火節。", "我們在觀音亭集合。",
        "我用 iPhone 拍攝。", "回家用 Final Cut Pro 剪輯。", "明天搭船去七美。",
        "如果風浪太大，就改留在馬公。", "先確認船班，再決定行程。",
    ])
    chapter_source = make_srt([
        "今天我們在澎湖馬公市區散步，先介紹步行路線和集合地點。",
        "從碼頭出發，沿著海邊步道走到市區，途中有幾個休息點。",
        "早上天氣晴朗，出門前還是要準備飲水和遮陽帽。",
        "接下來進入中央老街，看看當地建築和傳統店家。",
        "下午準備前往觀音亭休憩區，參加澎湖國際海上花火節。",
        "我們提早抵達海邊找位置，並確認現場的步行動線。",
        "拍攝煙火時，我用 iPhone 固定曝光，避免畫面忽明忽暗。",
        "收好器材後，我們一起等待煙火表演開始。",
        "明天打算搭船去七美，現在先確認船班和天氣。",
        "如果風浪太大，航班可能調整，出發前要再次確認。",
        "船班取消時，我們就改在馬公市區參觀博物館。",
        "今天先把交通和備案安排好，明天就能安心出門。",
    ], cue_seconds=15)
    fixtures = {"clean": clean, "errors": errors, "errors_gold": error_gold, "travel": travel, "short_travel": short_travel, "chapter_source": chapter_source}
    for name, value in fixtures.items():
        (out / f"{name}.srt").write_text(value)
    jobs = []
    for model in models:
        for case in ([] if synthetic_only else sorted(cases.iterdir())):
            if case.is_dir():
                for repeat in range(1, correction_repeats + 1):
                    jobs.append(dict(model=model, task="correction", case=case.name, repeat=repeat,
                                     source=(case / "original_transcribed.srt").read_text(),
                                     gold=(case / "final_simplified.srt").read_text(), historical=True))
        for name, source, gold in [("clean", clean, clean), ("errors", errors, error_gold)]:
            for repeat in (1, 2):
                jobs.append(dict(model=model, task="controls", case=name, repeat=repeat, source=source, gold=gold))
        translations = [("travel", travel), ("short_travel", short_travel)]
        if not synthetic_only:
            real_translation = "\n\n".join(split_srt_blocks((cases / "fire_01/final_traditional.srt").read_text())[:40])
            translations.append(("fire_01_first40", real_translation))
        for name, source in translations:
            jobs.append(dict(model=model, task="translation", case=name, repeat=1, source=source))
        for name in (() if synthetic_only else ("fire_01", "mazu_05")):
            jobs.append(dict(model=model, task="chapters", case=name, repeat=1,
                             source=normalize_traditional_for_validation((cases / name / "final_traditional.srt").read_text())))
        if synthetic_only:
            jobs.append(dict(model=model, task="chapters", case="synthetic_travel", repeat=1, source=normalize_traditional_for_validation(chapter_source)))
    return jobs


def estimate_cost(model, calls):
    input_rate, cache_rate, output_rate = RATES[model]
    total = 0
    for call in calls:
        usage = call.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or 0
        total += ((usage.get("prompt_tokens", 0) - cached) * input_rate
                  + cached * cache_rate + usage.get("completion_tokens", 0) * output_rate) / 1_000_000
        if model == "gpt-6-luna":
            total += (details.get("cache_write_tokens") or 0) * input_rate * 0.25 / 1_000_000
    return total


def run_job(job, out):
    folder = out / job["model"] / job["task"] / job["case"] / f"repeat_{job['repeat']:02d}"
    result_path = folder / "result.json"
    fingerprint_data = {
        "job": job,
        "source_hashes": {
            filename: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            for filename in ("funclip/llm/srt_corrector.py", "funclip/llm/srt_translator.py", "funclip/llm/youtube_chapters.py", "funclip/llm/model_config.py", "funclip/llm/chinese_converter.py", "scripts/eval_model_migration.py")
        },
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("fingerprint") == fingerprint:
            print(f"cached {job['model']} {job['task']} {job['case']} {job['repeat']}", flush=True)
            return cached
        raise RuntimeError(f"Source changed since cached run; use a new --output directory: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.srt").write_text(job["source"])
    CONTEXT.calls = []
    result = {key: value for key, value in job.items() if key not in {"source", "gold"}}
    result.update(fingerprint=fingerprint, started_at=datetime.now(timezone.utc).isoformat())
    started = time.perf_counter()
    try:
        if job["task"] in {"correction", "controls"}:
            response = srt_corrector.request_srt_correction(
                job["source"], model=job["model"], timeout=240, num_retries=0,
                max_completion_tokens=16_384,
            )
            output = response["corrected_content"]
            (folder / "output.srt").write_text(output)
            result["metrics"] = compute_metrics(job["source"], output, job["gold"], response["usage"], job["model"])
            result["validation"] = response["validation"]
            result["correction_attempts"] = response["attempts"]
            if job["task"] == "controls":
                result["metrics"]["reading_text_matches_expected"] = (
                    canonicalize_reading_text(flatten_srt_text(output))
                    == canonicalize_reading_text(flatten_srt_text(job["gold"]))
                )
                result["metrics"]["uncertain_names_preserved"] = all(term in output for term in ("阿昇", "阿珩"))
        elif job["task"] == "translation":
            output = srt_translator.translate_srt_to_english(
                job["source"], model=job["model"], request_timeout_seconds=120, max_retries=0, validation_retries=1,
            )
            (folder / "output.srt").write_text(output)
            result["metrics"] = compute_structural_metrics(job["source"], output)
            result["metrics"]["contains_cjk"] = bool(re.search(r"[\u3400-\u9fff]", flatten_srt_text(output)))
            if job["case"] == "travel":
                result["metrics"]["terms_preserved"] = {term: term.lower() in output.lower() for term in ("Penghu", "Guanyinting", "iPhone", "Final Cut Pro", "Magong")}
        else:
            response = youtube_chapters.generate_youtube_chapters(
                job["source"], model=job["model"], title_transform=normalize_traditional_for_validation,
                request_timeout_seconds=120, max_retries=0, validation_retries=1,
            )
            save_json(folder / "output.json", response)
            (folder / "chapters.txt").write_text(response["chapters_text"])
            result["metrics"] = {"hard_gate_pass": True, "chapter_count": len(response["chapters"])}
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="error", error=safe_error(exc))
    result.update(latency_seconds=time.perf_counter() - started, calls=CONTEXT.calls)
    result["estimated_cost_usd"] = estimate_cost(job["model"], CONTEXT.calls)
    save_json(result_path, result)
    print(f"{result['status']} {job['model']} {job['task']} {job['case']} {job['repeat']} {result['latency_seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=list(RATES), default=list(RATES))
    parser.add_argument("--synthetic-only", action="store_true", help="Send only newly generated test text; never read or send the existing subtitle corpus.")
    parser.add_argument("--correction-repeats", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--tasks", nargs="+", choices=["correction", "controls", "translation", "chapters"], help="Optionally run only selected tasks.")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for module in (srt_corrector, srt_translator, youtube_chapters):
        module.completion = measured_completion
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "packages": {name: importlib.metadata.version(name) for name in ("litellm", "openai")},
        "rates_per_million": RATES,
        "scope": ("Generated text only." if args.synthetic_only else "Five approved historical correction inputs plus task-specific fixtures; gold stays local."),
        "correction_repeats": args.correction_repeats,
        "tasks": args.tasks,
        "limitations": "Historical gold includes editorial changes; similarity is not correction accuracy. Small sample. No full-service browser/deployment test.",
    }
    save_json(out / "metadata.json", metadata)
    results = []
    jobs = jobs_for_models(args.models, out, synthetic_only=args.synthetic_only, correction_repeats=args.correction_repeats)
    if args.tasks:
        jobs = [job for job in jobs if job["task"] in args.tasks]
    # Alternate models so queue order does not confound an entire model with time.
    jobs.sort(key=lambda j: (j["task"], j["case"], j["repeat"], j["model"]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_job, job, out) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: (r["model"], r["task"], r["case"], r["repeat"]))
    summaries = {}
    for model in args.models:
        rows = [r for r in results if r["model"] == model]
        summaries[model] = {
            "completed": sum(r["status"] == "completed" for r in rows),
            "errors": sum(r["status"] != "completed" for r in rows),
            "estimated_cost_usd": sum(r["estimated_cost_usd"] for r in rows),
            "tasks": {},
        }
        for task in ("correction", "controls", "translation", "chapters"):
            task_rows = [r for r in rows if r["task"] == task]
            complete = [r for r in task_rows if r["status"] == "completed"]
            summaries[model]["tasks"][task] = {
                "attempted": len(task_rows), "completed": len(complete),
                "hard_gate_passes": sum(r.get("metrics", {}).get("hard_gate_pass", False) for r in task_rows),
                "median_latency_seconds": statistics.median(r["latency_seconds"] for r in complete) if complete else None,
            }
    save_json(out / "summary.json", {"summaries": summaries, "results": results})
    print(json.dumps(summaries, indent=2), flush=True)
    return 0 if all(r["status"] == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
