#!/usr/bin/env python3
import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FUNCLIP_DIR = REPO_ROOT / "funclip"

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

if str(FUNCLIP_DIR) not in sys.path:
    sys.path.append(str(FUNCLIP_DIR))


from dotenv import load_dotenv
from litellm import get_model_info, token_counter

from funclip.llm.srt_corrector import build_correction_prompt, request_srt_correction


load_dotenv()


CASE_INPUT_FILENAME = "original_transcribed.srt"
CASE_GOLD_FILENAME = "final_traditional.srt"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sanitize_token(raw_value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "item"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def discover_cases(cases_root: Path, selected_case_ids: set[str] | None) -> list[dict]:
    cases = []
    for case_dir in sorted(cases_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        if selected_case_ids and case_id not in selected_case_ids:
            continue

        input_path = case_dir / CASE_INPUT_FILENAME
        gold_path = case_dir / CASE_GOLD_FILENAME
        if not input_path.exists() or not gold_path.exists():
            continue

        cases.append(
            {
                "case_id": case_id,
                "case_dir": case_dir,
                "input_path": input_path,
                "gold_path": gold_path,
            }
        )

    return cases


def split_srt_blocks(srt_content: str) -> list[str]:
    normalized = normalize_newlines(srt_content).strip()
    if not normalized:
        return []
    return [block for block in re.split(r"\n\s*\n", normalized) if block.strip()]


def parse_srt_structure(srt_content: str) -> dict:
    blocks = split_srt_blocks(srt_content)
    if not blocks:
        return {"parse_ok": False, "error": "empty_or_unparseable", "segments": []}

    segments = []
    for index, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 2:
            return {
                "parse_ok": False,
                "error": f"segment_{index}_missing_timestamp",
                "segments": segments,
            }
        segment_index = lines[0].strip()
        timestamp = lines[1].strip()
        if not segment_index.isdigit():
            return {
                "parse_ok": False,
                "error": f"segment_{index}_bad_index",
                "segments": segments,
            }
        if "-->" not in timestamp:
            return {
                "parse_ok": False,
                "error": f"segment_{index}_bad_timestamp",
                "segments": segments,
            }

        segments.append(
            {
                "index": segment_index,
                "timestamp": timestamp,
                "text_lines": lines[2:],
            }
        )

    return {"parse_ok": True, "error": None, "segments": segments}


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for col_index, right_char in enumerate(right, start=1):
            insert_cost = current[col_index - 1] + 1
            delete_cost = previous[col_index] + 1
            replace_cost = previous[col_index - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return previous[-1]


def approximate_text_distance(left: str, right: str) -> int:
    distance = 0
    matcher = difflib.SequenceMatcher(a=left, b=right)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            distance += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            distance += i2 - i1
        elif tag == "insert":
            distance += j2 - j1
    return distance


def segment_text_distance(left_srt: str, right_srt: str) -> int:
    left_parse = parse_srt_structure(left_srt)
    right_parse = parse_srt_structure(right_srt)

    if left_parse["parse_ok"] and right_parse["parse_ok"]:
        left_segments = left_parse["segments"]
        right_segments = right_parse["segments"]
        if len(left_segments) == len(right_segments):
            return sum(
                levenshtein_distance(
                    "\n".join(left_segment["text_lines"]),
                    "\n".join(right_segment["text_lines"]),
                )
                for left_segment, right_segment in zip(left_segments, right_segments)
            )

    return approximate_text_distance(left_srt, right_srt)


def compute_structural_metrics(input_srt: str, output_srt: str) -> dict:
    input_parse = parse_srt_structure(input_srt)
    output_parse = parse_srt_structure(output_srt)

    output_parse_ok = output_parse["parse_ok"]
    same_segment_count = False
    same_indices = False
    same_timestamps = False

    if input_parse["parse_ok"] and output_parse_ok:
        input_segments = input_parse["segments"]
        output_segments = output_parse["segments"]
        same_segment_count = len(input_segments) == len(output_segments)
        same_indices = [segment["index"] for segment in input_segments] == [
            segment["index"] for segment in output_segments
        ]
        same_timestamps = [segment["timestamp"] for segment in input_segments] == [
            segment["timestamp"] for segment in output_segments
        ]

    hard_gate_pass = all(
        [
            output_parse_ok,
            same_segment_count,
            same_indices,
            same_timestamps,
        ]
    )

    return {
        "output_parse_ok": output_parse_ok,
        "output_parse_error": output_parse["error"],
        "same_segment_count": same_segment_count,
        "same_indices": same_indices,
        "same_timestamps": same_timestamps,
        "hard_gate_pass": hard_gate_pass,
    }


def estimate_cost_usd(model_name: str, usage: dict | None) -> float | None:
    if not usage:
        return None

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None

    try:
        model_info = get_model_info(model_name)
    except Exception:
        return None

    input_cost_per_token = model_info.get("input_cost_per_token")
    output_cost_per_token = model_info.get("output_cost_per_token")
    if input_cost_per_token is None or output_cost_per_token is None:
        return None

    return float(prompt_tokens) * float(input_cost_per_token) + float(completion_tokens) * float(
        output_cost_per_token
    )


def estimate_cost_usd_from_candidates(model_names: list[str | None], usage: dict | None) -> float | None:
    for model_name in model_names:
        if not model_name:
            continue
        estimated = estimate_cost_usd(model_name=model_name, usage=usage)
        if estimated is not None:
            return estimated
    return None


def estimate_usage_from_text(model_name: str, input_srt: str, output_srt: str) -> dict | None:
    try:
        messages = [
            {"role": "system", "content": build_correction_prompt()},
            {"role": "user", "content": input_srt},
        ]
        prompt_tokens = token_counter(model=model_name, messages=messages)
        completion_tokens = token_counter(
            model=model_name,
            text=output_srt,
            count_response_tokens=True,
        )
    except Exception:
        return None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def compute_metrics(input_srt: str, output_srt: str, gold_srt: str, usage: dict | None, model_name: str) -> dict:
    normalized_input = normalize_newlines(input_srt)
    normalized_output = normalize_newlines(output_srt)
    normalized_gold = normalize_newlines(gold_srt)

    structural = compute_structural_metrics(normalized_input, normalized_output)

    distance_input_to_output = segment_text_distance(normalized_input, normalized_output)
    distance_input_to_gold = segment_text_distance(normalized_input, normalized_gold)
    distance_output_to_gold = segment_text_distance(normalized_output, normalized_gold)

    return {
        **structural,
        "exact_match_gold": normalized_output == normalized_gold,
        "exact_match_input": normalized_output == normalized_input,
        "char_edit_distance_input_to_output": distance_input_to_output,
        "char_edit_distance_input_to_gold": distance_input_to_gold,
        "char_edit_distance_output_to_gold": distance_output_to_gold,
        "estimated_cost_usd": estimate_cost_usd(model_name=model_name, usage=usage),
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "total_tokens": usage.get("total_tokens") if usage else None,
    }


def hydrate_result_payload(payload: dict) -> dict:
    output_path = Path(payload["output_path"])
    if not output_path.exists():
        return payload

    input_srt = read_text(Path(payload["input_path"]))
    output_srt = read_text(output_path)
    metrics = dict(payload["metrics"])
    usage = payload.get("usage")
    requested_model = payload.get("requested_model")
    resolved_model = payload.get("resolved_model")

    if (
        metrics.get("prompt_tokens") is None
        or metrics.get("completion_tokens") is None
        or metrics.get("total_tokens") is None
    ) and requested_model:
        estimated_usage = estimate_usage_from_text(
            model_name=requested_model,
            input_srt=input_srt,
            output_srt=output_srt,
        )
        if estimated_usage is not None:
            metrics["prompt_tokens"] = estimated_usage["prompt_tokens"]
            metrics["completion_tokens"] = estimated_usage["completion_tokens"]
            metrics["total_tokens"] = estimated_usage["total_tokens"]
            if usage is None:
                usage = estimated_usage

    if metrics.get("estimated_cost_usd") is None:
        metrics["estimated_cost_usd"] = estimate_cost_usd_from_candidates(
            model_names=[requested_model, resolved_model],
            usage=usage,
        )

    hydrated = dict(payload)
    hydrated["metrics"] = metrics
    return hydrated


def load_completed_result(result_path: Path, output_path: Path) -> dict | None:
    if not result_path.exists() or not output_path.exists():
        return None
    try:
        payload = json.loads(read_text(result_path))
    except json.JSONDecodeError:
        return None
    if payload.get("status") != "completed":
        return None
    return payload


def recover_completed_result(
    result_path: Path,
    output_path: Path,
    case: dict,
    model_name: str,
    repeat_index: int,
    prompt_sha256: str,
    reasoning_effort: str | None,
) -> dict | None:
    if not output_path.exists():
        return None

    input_srt = read_text(case["input_path"])
    gold_srt = read_text(case["gold_path"])
    corrected_srt = read_text(output_path)
    metrics = compute_metrics(
        input_srt=input_srt,
        output_srt=corrected_srt,
        gold_srt=gold_srt,
        usage=None,
        model_name=model_name,
    )

    payload = {
        "status": "completed",
        "suite_name": result_path.parents[5].name,
        "case_id": case["case_id"],
        "repeat_index": repeat_index,
        "requested_model": model_name,
        "resolved_model": model_name,
        "response_id": None,
        "latency_seconds": None,
        "started_at": None,
        "completed_at": now_utc_iso(),
        "usage": None,
        "metrics": metrics,
        "input_path": str(case["input_path"]),
        "gold_path": str(case["gold_path"]),
        "output_path": str(output_path),
        "result_path": str(result_path),
        "prompt_sha256": prompt_sha256,
        "reasoning_effort": reasoning_effort,
        "recovered_from_existing_output": True,
    }
    write_json(result_path, payload)
    return payload


def write_suite_metadata(suite_root: Path, prompt_text: str, cases_root: Path, repeats: int) -> dict:
    suite_meta_path = suite_root / "suite.json"
    prompt_path = suite_root / "prompt.txt"
    prompt_sha256 = sha256_text(prompt_text)

    suite_root.mkdir(parents=True, exist_ok=True)
    write_text(prompt_path, prompt_text + "\n")

    if suite_meta_path.exists():
        suite_meta = json.loads(read_text(suite_meta_path))
        existing_prompt_sha = suite_meta.get("prompt_sha256")
        if existing_prompt_sha != prompt_sha256:
            raise RuntimeError(
                "Existing suite uses a different correction prompt. "
                "Create a new --suite-name for this benchmark."
            )
    else:
        suite_meta = {
            "created_at": now_utc_iso(),
        }

    suite_meta.update(
        {
            "updated_at": now_utc_iso(),
            "cases_root": str(cases_root),
            "prompt_sha256": prompt_sha256,
            "prompt_path": str(prompt_path),
            "default_repeats": repeats,
        }
    )
    write_json(suite_meta_path, suite_meta)
    return suite_meta


def aggregate_results(suite_root: Path, requested_models: list[str] | None = None) -> tuple[list[dict], dict]:
    requested_set = set(requested_models or [])
    results = []
    models_root = suite_root / "models"
    if not models_root.exists():
        return results, {}

    for result_path in sorted(models_root.glob("*/cases/*/repeat_*/result.json")):
        payload = hydrate_result_payload(json.loads(read_text(result_path)))
        if payload.get("status") != "completed":
            continue
        if requested_set and payload.get("requested_model") not in requested_set:
            continue
        results.append(payload)

    model_summaries = {}
    for model_name in sorted({result["requested_model"] for result in results}):
        model_results = [result for result in results if result["requested_model"] == model_name]
        latencies = [result["latency_seconds"] for result in model_results if result.get("latency_seconds") is not None]
        costs = [result["metrics"]["estimated_cost_usd"] for result in model_results if result["metrics"]["estimated_cost_usd"] is not None]
        exact_matches = [result["metrics"]["exact_match_gold"] for result in model_results]
        hard_gate_passes = [result["metrics"]["hard_gate_pass"] for result in model_results]
        distance_to_gold = [result["metrics"]["char_edit_distance_output_to_gold"] for result in model_results]

        model_summaries[model_name] = {
            "completed_results": len(model_results),
            "cases_covered": sorted({result["case_id"] for result in model_results}),
            "exact_match_rate": (sum(exact_matches) / len(exact_matches)) if exact_matches else None,
            "hard_gate_pass_rate": (sum(hard_gate_passes) / len(hard_gate_passes)) if hard_gate_passes else None,
            "latency_sample_count": len(latencies),
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "p95_latency_seconds": percentile(latencies, 95) if latencies else None,
            "mean_char_edit_distance_output_to_gold": (
                statistics.mean(distance_to_gold) if distance_to_gold else None
            ),
            "total_prompt_tokens": sum(
                result["metrics"]["prompt_tokens"] or 0 for result in model_results
            ),
            "total_completion_tokens": sum(
                result["metrics"]["completion_tokens"] or 0 for result in model_results
            ),
            "total_estimated_cost_usd": sum(costs) if costs else None,
        }

    return results, model_summaries


def percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    rank = (pct / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def write_summary_outputs(suite_root: Path, results: list[dict], model_summaries: dict) -> None:
    summary_json_path = suite_root / "summary.json"
    summary_csv_path = suite_root / "results.csv"
    summary_md_path = suite_root / "report.md"

    write_json(
        summary_json_path,
        {
            "updated_at": now_utc_iso(),
            "models": model_summaries,
            "result_count": len(results),
        },
    )

    fieldnames = [
        "requested_model",
        "resolved_model",
        "case_id",
        "repeat_index",
        "latency_seconds",
        "exact_match_gold",
        "hard_gate_pass",
        "char_edit_distance_output_to_gold",
        "char_edit_distance_input_to_output",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "result_path",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]
            writer.writerow(
                {
                    "requested_model": result["requested_model"],
                    "resolved_model": result.get("resolved_model"),
                    "case_id": result["case_id"],
                    "repeat_index": result["repeat_index"],
                    "latency_seconds": result["latency_seconds"],
                    "exact_match_gold": metrics["exact_match_gold"],
                    "hard_gate_pass": metrics["hard_gate_pass"],
                    "char_edit_distance_output_to_gold": metrics["char_edit_distance_output_to_gold"],
                    "char_edit_distance_input_to_output": metrics["char_edit_distance_input_to_output"],
                    "prompt_tokens": metrics["prompt_tokens"],
                    "completion_tokens": metrics["completion_tokens"],
                    "total_tokens": metrics["total_tokens"],
                    "estimated_cost_usd": metrics["estimated_cost_usd"],
                    "result_path": result["result_path"],
                }
            )

    report_lines = ["# SRT Auto-Correction Report", ""]
    for model_name, summary in model_summaries.items():
        report_lines.extend(
            [
                f"## {model_name}",
                "",
                f"- completed results: {summary['completed_results']}",
                f"- exact match rate: {format_float(summary['exact_match_rate'])}",
                f"- hard gate pass rate: {format_float(summary['hard_gate_pass_rate'])}",
                f"- latency samples: {summary['latency_sample_count']}",
                f"- median latency (s): {format_float(summary['median_latency_seconds'])}",
                f"- p95 latency (s): {format_float(summary['p95_latency_seconds'])}",
                f"- mean distance to gold: {format_float(summary['mean_char_edit_distance_output_to_gold'])}",
                f"- total estimated cost (USD): {format_float(summary['total_estimated_cost_usd'], digits=6)}",
                "",
            ]
        )

    write_text(summary_md_path, "\n".join(report_lines).rstrip() + "\n")


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached, resumable SRT auto-correction model comparisons."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to compare, e.g. gpt-4o-mini gpt-5-mini gpt-5.4-mini",
    )
    parser.add_argument(
        "--cases-root",
        default=str(REPO_ROOT / "eval" / "srt_correction_cases"),
        help="Root directory containing eval cases.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Optional subset of case ids to run.",
    )
    parser.add_argument(
        "--suite-name",
        default="baseline_v1",
        help="Stable suite name used for caching and incremental reruns.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "eval" / "srt_correction_runs"),
        help="Root directory where model comparison runs are stored.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated calls per case and model.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional explicit API key. Defaults to env/.env handling.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL override.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed case/model/repeat results instead of using the cache.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="Optional reasoning effort forwarded to reasoning models.",
    )
    return parser.parse_args()


def main() -> int:
    args = load_args()

    cases_root = Path(args.cases_root).expanduser().resolve()
    suite_root = Path(args.output_root).expanduser().resolve() / sanitize_token(args.suite_name)
    selected_case_ids = set(args.case_ids) if args.case_ids else None

    cases = discover_cases(cases_root=cases_root, selected_case_ids=selected_case_ids)
    if not cases:
        raise RuntimeError(f"No eval cases found in {cases_root}")

    prompt_text = build_correction_prompt()
    suite_meta = write_suite_metadata(
        suite_root=suite_root,
        prompt_text=prompt_text,
        cases_root=cases_root,
        repeats=args.repeats,
    )

    completed = 0
    skipped = 0
    failed = 0

    for model_name in args.models:
        model_slug = sanitize_token(model_name)
        for case in cases:
            input_srt = read_text(case["input_path"])
            gold_srt = read_text(case["gold_path"])
            for repeat_index in range(1, args.repeats + 1):
                repeat_dir = (
                    suite_root
                    / "models"
                    / model_slug
                    / "cases"
                    / case["case_id"]
                    / f"repeat_{repeat_index:02d}"
                )
                output_path = repeat_dir / "corrected.srt"
                result_path = repeat_dir / "result.json"

                existing = None if args.force else load_completed_result(result_path=result_path, output_path=output_path)
                if existing is not None:
                    skipped += 1
                    print(
                        f"skip model={model_name} case={case['case_id']} repeat={repeat_index} "
                        f"path={result_path}"
                    )
                    continue

                recovered = None if args.force else recover_completed_result(
                    result_path=result_path,
                    output_path=output_path,
                    case=case,
                    model_name=model_name,
                    repeat_index=repeat_index,
                    prompt_sha256=suite_meta["prompt_sha256"],
                    reasoning_effort=args.reasoning_effort,
                )
                if recovered is not None:
                    skipped += 1
                    print(
                        f"recover model={model_name} case={case['case_id']} repeat={repeat_index} "
                        f"path={result_path}"
                    )
                    continue

                repeat_dir.mkdir(parents=True, exist_ok=True)

                request_kwargs = {}
                if args.reasoning_effort:
                    request_kwargs["reasoning_effort"] = args.reasoning_effort

                started_at = now_utc_iso()
                started_clock = time.perf_counter()
                try:
                    response = request_srt_correction(
                        srt_content=input_srt,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        model=model_name,
                        **request_kwargs,
                    )
                    latency_seconds = time.perf_counter() - started_clock
                    corrected_srt = response["corrected_content"]
                    write_text(output_path, corrected_srt)

                    metrics = compute_metrics(
                        input_srt=input_srt,
                        output_srt=corrected_srt,
                        gold_srt=gold_srt,
                        usage=response.get("usage"),
                        model_name=response.get("resolved_model") or model_name,
                    )

                    payload = {
                        "status": "completed",
                        "suite_name": suite_root.name,
                        "case_id": case["case_id"],
                        "repeat_index": repeat_index,
                        "requested_model": model_name,
                        "resolved_model": response.get("resolved_model"),
                        "response_id": response.get("response_id"),
                        "latency_seconds": latency_seconds,
                        "started_at": started_at,
                        "completed_at": now_utc_iso(),
                        "usage": response.get("usage"),
                        "metrics": metrics,
                        "input_path": str(case["input_path"]),
                        "gold_path": str(case["gold_path"]),
                        "output_path": str(output_path),
                        "result_path": str(result_path),
                        "prompt_sha256": suite_meta["prompt_sha256"],
                        "reasoning_effort": args.reasoning_effort,
                    }
                    write_json(result_path, payload)
                    completed += 1
                    print(
                        f"done model={model_name} case={case['case_id']} repeat={repeat_index} "
                        f"exact_match={metrics['exact_match_gold']} hard_gate={metrics['hard_gate_pass']} "
                        f"latency_s={latency_seconds:.2f}"
                    )
                except Exception as exc:
                    failed += 1
                    error_payload = {
                        "status": "error",
                        "suite_name": suite_root.name,
                        "case_id": case["case_id"],
                        "repeat_index": repeat_index,
                        "requested_model": model_name,
                        "started_at": started_at,
                        "failed_at": now_utc_iso(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "input_path": str(case["input_path"]),
                        "gold_path": str(case["gold_path"]),
                    }
                    write_json(result_path, error_payload)
                    print(
                        f"error model={model_name} case={case['case_id']} repeat={repeat_index} "
                        f"type={type(exc).__name__}"
                    )

    results, model_summaries = aggregate_results(suite_root=suite_root)
    write_summary_outputs(suite_root=suite_root, results=results, model_summaries=model_summaries)

    print(f"suite_root={suite_root}")
    print(f"completed={completed}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    print(f"summary_json={suite_root / 'summary.json'}")
    print(f"summary_csv={suite_root / 'results.csv'}")
    print(f"summary_md={suite_root / 'report.md'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
