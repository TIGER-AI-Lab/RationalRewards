import glob
import json
import os
import re
from typing import Any, Dict, List


def _extract_float_from_raw(raw_val):
    if not isinstance(raw_val, str):
        if isinstance(raw_val, (int, float)):
            return float(raw_val)
        return None
    cleaned = raw_val.strip().replace("*", "").replace("~", "")
    if cleaned.upper() == "N/A":
        return "N/A"
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_numeric_score(score_value, fallback=2.5):
    extracted = _extract_float_from_raw(score_value)
    if extracted is None or extracted == "N/A":
        return fallback
    return float(extracted)


def get_ranking(score_a: float, score_b: float, tolerance: float = 0.0) -> str:
    if abs(score_a - score_b) <= tolerance:
        return "A=B"
    return "A>B" if score_a > score_b else "B>A"


def compute_pred_total(data: Dict[str, Any], task_type: str, mode: str, candidate: str, fallback: float) -> float:
    score_key = f"score_{candidate}"
    if task_type == "edit":
        text = extract_numeric_score(data.get("text_faithfulness", {}).get(score_key), fallback)
        image = extract_numeric_score(data.get("image_faithfulness", {}).get(score_key), fallback)
        physical = extract_numeric_score(data.get("physical_quality", {}).get(score_key), fallback)
        rendering = extract_numeric_score(data.get("text_rendering", {}).get(score_key), fallback)
        if mode == "text":
            return text
        if mode == "visual":
            return image + physical
        return text + image + physical + rendering

    text = extract_numeric_score(data.get("text_faithfulness", {}).get(score_key), fallback)
    physical = extract_numeric_score(data.get("physical_quality", {}).get(score_key), fallback)
    rendering = extract_numeric_score(data.get("text_rendering", {}).get(score_key), fallback)
    if mode == "text":
        return text
    if mode == "visual":
        return physical
    return text + physical + rendering


def _normalize_chosen(chosen_value) -> str:
    if chosen_value is None:
        return "UNKNOWN"
    v = str(chosen_value).strip().lower()
    if v in {"a", "left", "leftvote", "1", "model_1"}:
        return "A>B"
    if v in {"b", "right", "rightvote", "2", "model_2"}:
        return "B>A"
    if v in {"tie", "draw", "equal", "a=b"}:
        return "A=B"
    return "UNKNOWN"


def _gt_ranking_from_numeric(item: Dict[str, Any], mode: str) -> str:
    i1 = float(item.get("instruction_following_1", 0) or 0)
    i2 = float(item.get("instruction_following_2", 0) or 0)
    v1 = float(item.get("visual_quality_1", 0) or 0)
    v2 = float(item.get("visual_quality_2", 0) or 0)
    if mode == "text":
        return get_ranking(i1, i2)
    if mode == "visual":
        return get_ranking(v1, v2)
    return get_ranking(i1 + v1, i2 + v2)


def evaluate_directory(result_dir: str, task_type: str, mode: str, label_source: str, tolerance: float, fallback: float):
    files = sorted(glob.glob(os.path.join(result_dir, "*_pairwise.json")))
    total = 0
    success = 0
    skipped = 0
    failures: List[Dict[str, Any]] = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            item = json.load(f)
        if not item.get("success", True):
            skipped += 1
            continue
        pred_a = compute_pred_total(item, task_type, mode, "a", fallback)
        pred_b = compute_pred_total(item, task_type, mode, "b", fallback)
        pred_rank = get_ranking(pred_a, pred_b, tolerance)

        if label_source == "chosen":
            gt_rank = _normalize_chosen(item.get("chosen"))
        else:
            gt_rank = _gt_ranking_from_numeric(item, mode)

        if gt_rank == "UNKNOWN":
            skipped += 1
            continue

        total += 1
        if pred_rank == gt_rank:
            success += 1
        else:
            failures.append(
                {
                    "pair_id": item.get("pair_id"),
                    "pred_rank": pred_rank,
                    "gt_rank": gt_rank,
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "file": os.path.abspath(fp),
                }
            )

    acc = (success / total) if total else 0.0
    return {
        "result_dir": os.path.abspath(result_dir),
        "task_type": task_type,
        "mode": mode,
        "label_source": label_source,
        "total_evaluated": total,
        "passed": success,
        "skipped": skipped,
        "accuracy": acc,
        "failed_cases": failures,
    }

