import argparse
import json
import os
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import ray
import torch
from datasets import Dataset, load_from_disk
from diffusers import FluxKontextPipeline, QwenImageEditPlusPipeline
from peft import PeftModel
from PIL import Image
from tqdm import tqdm


def collect_processed_keys(base_output_dir: str) -> set[str]:
    processed_keys = set()
    if not os.path.exists(base_output_dir):
        return processed_keys
    for root, _, files in os.walk(base_output_dir):
        for file in files:
            if file.endswith(".png"):
                processed_keys.add(os.path.splitext(file)[0])
    print(f"Found {len(processed_keys)} already processed items")
    return processed_keys


def calculate_target_dimensions(width: int, height: int, max_edge: int = 1024) -> Tuple[int, int]:
    if max(width, height) > max_edge:
        scale = max_edge / max(width, height)
        width = int(width * scale)
        height = int(height * scale)
    target_width = int(round(width / 16.0)) * 16
    target_height = int(round(height / 16.0)) * 16
    return target_width, target_height


def load_dataset_by_name(dataname: str, dataset_path: str) -> Dataset:
    if dataname in {"pica", "imgedit"}:
        df = pd.read_parquet(dataset_path)
        dataset = Dataset.from_pandas(df)
    else:
        dataset = load_from_disk(dataset_path)

    if dataname == "pica":
        def add_key_field(example, idx):
            if "key" not in example:
                physics_law = example.get("physics_law", "unknown")
                example["key"] = f"{idx}_{physics_law}"
            return example
        dataset = dataset.map(add_key_field, with_indices=True)
    return dataset


def normalize_row(dataname: str, row: Dict[str, Any]) -> Tuple[str, Image.Image, str]:
    if dataname == "pica":
        key = row["key"]
        source = row["input_image"]
        task_type = row["physics_category"]
    elif dataname == "imgedit":
        key = row["key"]
        source = row["image"]
        task_type = row["edit_type"]
    else:
        key = row["key"]
        source = row["input_image_raw"]
        task_type = row["task_type"]

    if isinstance(source, dict) and "bytes" in source:
        image = Image.open(BytesIO(source["bytes"])).convert("RGB")
    else:
        image = source.convert("RGB") if hasattr(source, "convert") else source
    return key, image, task_type


def load_pipeline(model_family: str, pretrained_name_or_path: str, lora_path: Optional[str]):
    if model_family == "qwen":
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            pretrained_name_or_path,
            torch_dtype=torch.bfloat16,
        )
    elif model_family == "flux":
        pipe = FluxKontextPipeline.from_pretrained(
            pretrained_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        if getattr(pipe, "vae", None) is not None:
            pipe.vae.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")

    if lora_path:
        possible_lora_subdir = os.path.join(lora_path, "lora")
        if os.path.isdir(possible_lora_subdir):
            lora_path = possible_lora_subdir
        if hasattr(pipe, "transformer"):
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer,
                lora_path,
                adapter_name="default",
            )
            pipe.transformer = pipe.transformer.merge_and_unload()
        else:
            pipe.load_lora_weights(
                lora_path,
                weight_name="adapter_model.safetensors",
                adapter_name="lora",
            )
            pipe.set_adapters(["lora"], adapter_weights=[1])

    pipe.to("cuda")
    for component_name in ["vae", "text_encoder", "text_encoder_2", "transformer"]:
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.eval()
            component.requires_grad_(False)
    return pipe


@ray.remote(num_gpus=1)
def process_slice(
    slice_items: Dataset,
    model_family: str,
    pretrained_name_or_path: str,
    lora_path: Optional[str],
    refineprompt_path: str,
    output_dir: str,
    dataname: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    true_cfg_scale: float,
    max_edge: int,
    skip_good_threshold: float,
):
    pipe = load_pipeline(model_family, pretrained_name_or_path, lora_path)

    for row in tqdm(slice_items):
        try:
            key, input_image, task_type = normalize_row(dataname, row)
            refinement_file = os.path.join(refineprompt_path, f"{key}_refinement.json")
            if not os.path.exists(refinement_file):
                continue
            with open(refinement_file, "r", encoding="utf-8") as f:
                refinement_data = json.load(f)

            text_score = refinement_data.get("text_faithfulness", {}).get("score", 0)
            physical_score = refinement_data.get("physical_quality", {}).get("score", 0)
            if text_score >= skip_good_threshold and physical_score >= skip_good_threshold:
                continue

            instruction = refinement_data.get("refined_request") or row.get("prompt") or row.get("instruction")
            if not instruction:
                continue

            out_path = os.path.join(output_dir, task_type, "en")
            os.makedirs(out_path, exist_ok=True)
            save_path = os.path.join(out_path, f"{key}.png")
            if os.path.exists(save_path):
                continue

            target_width, target_height = calculate_target_dimensions(*input_image.size, max_edge=max_edge)
            input_image = input_image.resize((target_width, target_height))
            generator = torch.Generator(device="cuda").manual_seed(seed)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if model_family == "qwen":
                    output_image = pipe(
                        num_inference_steps=num_inference_steps,
                        image=input_image,
                        prompt=instruction,
                        negative_prompt=" ",
                        true_cfg_scale=true_cfg_scale,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                else:
                    output_image = pipe(
                        num_inference_steps=num_inference_steps,
                        image=input_image,
                        prompt=instruction,
                        guidance_scale=guidance_scale,
                        width=target_width,
                        height=target_height,
                        generator=generator,
                    ).images[0]

            output_image.save(save_path)
        except Exception as exc:
            print(f"Failed on row {row.get('key', 'unknown')}: {exc}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Test-time generation with refined prompts (diffusers backend)")
    parser.add_argument("--model-family", choices=["qwen", "flux"], default="qwen")
    parser.add_argument("--pretrained-name-or-path", required=True, type=str)
    parser.add_argument("--lora-path", default=None, type=str)
    parser.add_argument("--dataname", choices=["pica", "imgedit", "gedit"], default="pica")
    parser.add_argument("--dataset-path", required=True, type=str)
    parser.add_argument("--refineprompt-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--num-gpus", type=int, default=max(1, torch.cuda.device_count()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--skip-good-threshold", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ray.init(ignore_reinit_error=True)

    dataset = load_dataset_by_name(args.dataname, args.dataset_path)
    processed_keys = collect_processed_keys(args.output_dir)
    if processed_keys:
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: x["key"] not in processed_keys)
        print(f"Filtered dataset from {original_size} to {len(dataset)} items")

    if len(dataset) == 0:
        print("No samples to process.")
        return

    num_shards = min(args.num_gpus, len(dataset))
    shards = [dataset.shard(num_shards=num_shards, index=i) for i in range(num_shards)]
    ray.get(
        [
            process_slice.remote(
                slice_items=shards[i],
                model_family=args.model_family,
                pretrained_name_or_path=args.pretrained_name_or_path,
                lora_path=args.lora_path,
                refineprompt_path=args.refineprompt_path,
                output_dir=args.output_dir,
                dataname=args.dataname,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                true_cfg_scale=args.true_cfg_scale,
                max_edge=args.max_edge,
                skip_good_threshold=args.skip_good_threshold,
            )
            for i in range(num_shards)
        ]
    )
    print(f"saved results to {args.output_dir}")


if __name__ == "__main__":
    main()