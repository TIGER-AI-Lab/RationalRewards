from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def _to_image_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, bytes):
        return [{"bytes": value}]
    return []


def load_parquet_input(data_file: str) -> pd.DataFrame:
    data_path = Path(data_file)
    if data_path.is_dir():
        files = sorted(glob(str(data_path / "*.parquet")))
        dflist = [pd.read_parquet(fp) for fp in files]
        return pd.concat(dflist, ignore_index=True) if dflist else pd.DataFrame()
    return pd.read_parquet(data_file)


def infer_dataset_type(columns: Iterable[str], task_type: str) -> str:
    cols = set(columns)
    if {"instruction", "source_image", "candidate_1", "candidate_2"}.issubset(cols):
        return "editreward_imgedit"
    if {"prompt", "left_image", "right_image"}.issubset(cols):
        return "genaibench_imggen"
    if {"source_prompt", "target_prompt", "left_output_image", "right_output_image"}.issubset(cols):
        return "genaibench_imgedit"
    if {"pair_id", "prompt_text", "response_a_images", "response_b_images"}.issubset(cols):
        return "mmrb2_imgedit" if task_type == "edit" else "mmrb2_imggen"
    raise ValueError(f"Cannot infer dataset type from columns: {sorted(cols)}")


def normalize_dataframe(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    out = pd.DataFrame()

    if dataset_type in {"mmrb2_imgedit", "mmrb2_imggen"}:
        out["pair_id"] = df["pair_id"].astype(str) if "pair_id" in df.columns else df.index.astype(str)
        out["prompt_text"] = df["prompt_text"]
        out["prompt_images"] = df["prompt_images"].apply(_to_image_list) if "prompt_images" in df.columns else [[] for _ in range(len(df))]
        out["response_a_images"] = df["response_a_images"].apply(_to_image_list)
        out["response_b_images"] = df["response_b_images"].apply(_to_image_list)
        out["chosen"] = df["chosen"] if "chosen" in df.columns else None
        out["response_a_model"] = df["response_a_model"] if "response_a_model" in df.columns else ""
        out["response_b_model"] = df["response_b_model"] if "response_b_model" in df.columns else ""
        return out

    if dataset_type == "genaibench_imggen":
        out["pair_id"] = df["pair_id"].astype(str) if "pair_id" in df.columns else df.index.astype(str)
        out["prompt_text"] = df["prompt"]
        out["prompt_images"] = [[] for _ in range(len(df))]
        out["response_a_images"] = df["left_image"].apply(_to_image_list)
        out["response_b_images"] = df["right_image"].apply(_to_image_list)
        out["response_a_model"] = df["left_model"] if "left_model" in df.columns else ""
        out["response_b_model"] = df["right_model"] if "right_model" in df.columns else ""
        vote = df["vote_type"] if "vote_type" in df.columns else None
        if vote is not None:
            out["chosen"] = vote.map({"leftvote": "left", "rightvote": "right"}).fillna("tie")
        else:
            out["chosen"] = None
        return out

    if dataset_type == "genaibench_imgedit":
        out["pair_id"] = df["pair_id"].astype(str) if "pair_id" in df.columns else [f"genaibench_{i}" for i in range(len(df))]
        out["prompt_text"] = df["instruct_prompt"]
        out["prompt_images"] = df["source_image"].apply(_to_image_list)
        out["response_a_images"] = df["left_output_image"].apply(_to_image_list)
        out["response_b_images"] = df["right_output_image"].apply(_to_image_list)
        out["response_a_model"] = df["left_model"] if "left_model" in df.columns else ""
        out["response_b_model"] = df["right_model"] if "right_model" in df.columns else ""
        vote = df["vote_type"] if "vote_type" in df.columns else None
        if vote is not None:
            out["chosen"] = vote.map({"leftvote": "left", "rightvote": "right"}).fillna("tie")
        else:
            out["chosen"] = None
        return out

    if dataset_type == "editreward_imgedit":
        out["pair_id"] = df.index.astype(str)
        out["prompt_text"] = df["instruction"]
        out["prompt_images"] = df["source_image"].apply(_to_image_list)
        out["response_a_images"] = df["candidate_1"].apply(_to_image_list)
        out["response_b_images"] = df["candidate_2"].apply(_to_image_list)
        out["response_a_model"] = df["model_1"] if "model_1" in df.columns else ""
        out["response_b_model"] = df["model_2"] if "model_2" in df.columns else ""
        out["chosen"] = df["ranking"] if "ranking" in df.columns else None
        for key in (
            "instruction_following_1",
            "instruction_following_2",
            "visual_quality_1",
            "visual_quality_2",
        ):
            out[key] = df[key] if key in df.columns else None
        return out

    raise ValueError(f"Unsupported dataset type: {dataset_type}")

