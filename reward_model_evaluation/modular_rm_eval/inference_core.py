import argparse
import asyncio
import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, List

from PIL import Image
from tqdm.asyncio import tqdm

from .dataset_adapters import infer_dataset_type, load_parquet_input, normalize_dataframe
from .http_client import VLMHTTPClient
from .parsing import parse_pairwise_response
from .prompts import EDIT_PROMPT_TEMPLATE, GEN_PROMPT_TEMPLATE


def _b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def build_messages(prompt_text: str, source_images: List[dict], image_a: dict, image_b: dict, task_type: str):
    template = EDIT_PROMPT_TEMPLATE if task_type == "edit" else GEN_PROMPT_TEMPLATE
    text = template.format(request=prompt_text)
    parts = text.split("<image>")
    content: List[Dict[str, Any]] = [{"type": "text", "text": parts[0]}]

    if task_type == "edit":
        for src in source_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(src['bytes'])}"}})
        if len(parts) > 1:
            content.append({"type": "text", "text": parts[1]})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image_a['bytes'])}"}})

    if len(parts) > 2:
        content.append({"type": "text", "text": parts[2] if task_type == "edit" else parts[1]})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image_b['bytes'])}"}})

    tail_start = 3 if task_type == "edit" else 2
    for idx in range(tail_start, len(parts)):
        content.append({"type": "text", "text": parts[idx]})
    return [{"role": "user", "content": content}]


async def evaluate_row(client: VLMHTTPClient, row: Dict[str, Any], task_type: str, semaphore: asyncio.Semaphore, output_dir: str):
    aspects = ["text_faithfulness", "physical_quality", "text_rendering"]
    if task_type == "edit":
        aspects = ["text_faithfulness", "image_faithfulness", "physical_quality", "text_rendering"]

    async with semaphore:
        pair_id = str(row["pair_id"])
        result = {
            "pair_id": pair_id,
            "instruction": row["prompt_text"],
            "chosen": row.get("chosen", None),
            "response_a_model": row.get("response_a_model", ""),
            "response_b_model": row.get("response_b_model", ""),
            "num_source_images": len(row.get("prompt_images", [])),
        }
        for key in ("instruction_following_1", "instruction_following_2", "visual_quality_1", "visual_quality_2"):
            if key in row:
                result[key] = row[key]

        try:
            messages = build_messages(
                prompt_text=row["prompt_text"],
                source_images=row.get("prompt_images", []),
                image_a=row["response_a_images"][0],
                image_b=row["response_b_images"][0],
                task_type=task_type,
            )
            response = await client.generate(messages)
            if not response:
                result["success"] = False
                result["error"] = "No response from model"
            else:
                result.update(parse_pairwise_response(response, aspects))
                result["success"] = True
        except Exception as exc:
            result["success"] = False
            result["error"] = str(exc)

        filepath = os.path.join(output_dir, f"{pair_id}_pairwise.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result


async def run(args):
    df = load_parquet_input(args.data_file)
    if args.max_samples:
        df = df.head(args.max_samples)

    task_type = args.task_type
    dataset_type = args.dataset_type
    if dataset_type == "auto":
        dataset_type = infer_dataset_type(df.columns, task_type)
    normalized = normalize_dataframe(df, dataset_type)

    output_dir = args.output_dir or f"evalresults/{args.evalsetting}"
    os.makedirs(output_dir, exist_ok=True)

    if args.resume:
        finished = set()
        for name in os.listdir(output_dir):
            if name.endswith("_pairwise.json"):
                finished.add(name[: -len("_pairwise.json")])
        if finished:
            normalized = normalized[~normalized["pair_id"].astype(str).isin(finished)]

    client = VLMHTTPClient(
        model_name=args.model_name,
        base_url=args.api_url if args.is_api_server and args.api_url else args.server_host,
        port=args.server_port,
        timeout=args.timeout,
        api_key=args.api_key,
        is_api_server=args.is_api_server,
    )
    if not await client.check_connection():
        raise RuntimeError("Cannot connect to VLM endpoint.")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    for row in normalized.to_dict(orient="records"):
        tasks.append(asyncio.create_task(evaluate_row(client, row, task_type, sem, output_dir)))
    results = await tqdm.gather(*tasks) if tasks else []
    ok = sum(1 for r in results if r.get("success"))
    total = len(results)
    print(f"Done. Success {ok}/{total}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Unified pairwise reward-model inference runner")
    parser.add_argument("--data-file", required=True, type=str, help="Parquet file or directory with parquet files")
    parser.add_argument("--evalsetting", required=True, type=str, help="Name for this evaluation run")
    parser.add_argument("--task-type", choices=["edit", "gen"], required=True)
    parser.add_argument(
        "--dataset-type",
        default="auto",
        choices=[
            "auto",
            "mmrb2_imgedit",
            "mmrb2_imggen",
            "genaibench_imgedit",
            "genaibench_imggen",
            "editreward_imgedit",
        ],
    )
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--concurrency", default=32, type=int)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--server-host", default="http://localhost", type=str)
    parser.add_argument("--server-port", default=6868, type=int)
    parser.add_argument("--timeout", default=300, type=int)
    parser.add_argument("--model-name", default="Qwen3-VL-8B-Instruct", type=str)
    parser.add_argument("--is-api-server", action="store_true")
    parser.add_argument("--api-url", default=None, type=str)
    parser.add_argument("--api-key", default=None, type=str)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

