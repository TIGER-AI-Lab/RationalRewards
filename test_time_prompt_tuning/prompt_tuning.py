import json
import os
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback
import base64
from io import BytesIO
import aiohttp
from PIL import Image
from tqdm.asyncio import tqdm
import pandas as pd
import re
from glob import glob
from datasets import load_dataset, load_from_disk
from concurrent.futures import ThreadPoolExecutor 
do_save_image = False
# Configuration
judgemodel = ""

instruction = """You are an expert image editing evaluator. Your task is to evaluate the quality of an edited image based on a source image and a user instruction. Afterwards, you need to suggest how to refine the original user request to produce better image edits (if any).

User Instruction: {request}
You are provided with two images:
1. Source Image <image>
2. Edited Image <image>

Your task is to evaluate the Edited Image against the Source Image and the User Instruction.
To do this, you must first assess the image on four critical aspects, provide justifications and absolute scores in 1-4 scale. 
About the scores: you should try to give **float scores**. For example, float values are important to reflect fine-grained preferences when you compare two edited images. 

### Critical Aspects & Scoring Rubric
**1. Text Faithfulness** (How accurately does the output follow the instruction?)
- **4 (Full match):** All key elements (objects, colors, actions) are represented exactly as described. No hallucinations or unrequested changes.
- **3 (Minor mismatch):** Most key elements are present, but minor details are missing, incorrect, or slightly inaccurate.
- **2 (Some mismatch):** Some key elements are missing, altered, or interpreted incorrectly.
- **1 (Major deviations):** Key elements are completely missing, altered, or contradicted. Instruction is ignored.

**2. Image Faithfulness** (How well are the non-edited parts and key input elements preserved?)
- **4 (Uses input fully):** All relevant elements from the input (background, style, lighting, identity) are accurately preserved or transformed as instructed.
- **3 (Minor mismatch):** Most relevant elements are preserved, but a few aspects (e.g., background details, lighting consistency) are missing or incorrectly handled.
- **2 (Partial mismatch):** Some elements are carried over, but key aspects of the original image are lost or distorted.
- **1 (Fails to use input):** Key elements of the input image are ignored, misinterpreted, or destroyed.

**3. Physical and Visual Quality** (Technical errors, composition, realism, and physics)
- **4 (No noticeable flaws):** The image is physically plausible (correct lighting, shadows, geometry, anatomy). No visible artifacts (seams, blurring, noise).
- **3 (Minor flaws):** Small inaccuracies that are noticeable but not strongly disruptive (e.g., slight lighting mismatch, minor texture issues).
- **2 (Some flaws):** Clear physical or visual errors that disrupt the image (e.g., incorrect perspective, "floating" objects, wrong shadow direction, obvious seams).
- **1 (Severe flaws):** Major physical/visual errors (e.g., impossible geometry, distorted anatomy, garbled objects, severe artifacts).

**4. Text Rendering** (Only if the instruction involves generating text)
- **4 (Full match):** Text is correct, legible, and integrated well.
- **3 (Mostly match):** Minor misspellings or inconsistent capitalization.
- **2 (Partial match):** Major misspellings or distorted text.
- **1 (Major deviations):** Text is unreadable, severely distorted, or missing. (Use N/A if no text generation is required).

Afterwards, try to construct a refined user request that helps the visual generation model to produce better image edits.
Think of the weaknesses identified in the judgement, then map them to instruction details and apply specific fixes. 
Provide a final new user request that enrich the initial user request.

Output your evaluation in the following format:
[ understanding the user request, and what needs to be considered during image editing ]
# Detailed Judgement
1. Text Faithfulness:
## Score: [ float score ]
## Justification: [Detailed explanation of the score]
2. Image Faithfulness:
## Score: [ float score ]
## Justification: [Detailed explanation of the score]
3. Physical and Visual Quality:
## Score: [ float score ]
## Justification: [Detailed explanation of the score]
4. Text Rendering:
## Score: [ float score or N/A ]
## Justification: [Detailed explanation of the score]
# Summary: [Summary of the evaluation]

# User Request Refinement:
## Refinement Comments: [Specific suggestions for improving the user request]
## Refined Request: [The improved, more specific user request for editing like a standard user instruction]"""


class Qwen3VLInferenceHTTP:
    def __init__(self,
                 model_name="Qwen3-VL-8B-Instruct",
                 base_url="http://localhost",
                 port=8000,
                 timeout=300):
        self.model_name = model_name
        self.base_url = f"{base_url}:{port}"
        self.timeout = timeout
        self.generate_endpoint = f"{self.base_url}/v1/chat/completions"
        self.health_endpoint = f"{self.base_url}/v1/models"
        print(f"Initialized Async HTTP client for model: {model_name}")

    def image_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    async def check_connection(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.health_endpoint, timeout=10) as response:
                    if response.status == 200:
                        print("✓ Server connection successful")
                        return True
                    else:
                        print(f"⚠ Server responded with status code: {response.status}")
                        return False
        except Exception as e:
            print(f"⚠ Warning: Could not connect to server: {e}")
            return False

    async def generate(self, messages, temperature=0.1, max_tokens=20480):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.generate_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"]
                        else:
                            return None
                    else:
                        print(f"Server returned error {response.status}")
                        return None
            except Exception as e:
                print(f"Error during generation: {e}")
                return None

class EditRewardInference:
    def __init__(self, client):
        self.client = client


    def create_single_evaluation_message(self, instruction_text, source_image_bytes, edited_image_bytes):
        """Create the evaluation message for single image editing evaluation"""

        system_prompt = "You are an expert image editing evaluator."

        user_content_text = instruction.format(request=instruction_text)

        # Convert bytes to base64
        source_b64 = base64.b64encode(source_image_bytes).decode()
        edited_b64 = base64.b64encode(edited_image_bytes).decode()

        # Interleave images into the <image> placeholders in user_content_text
        content = []
        parts = user_content_text.split("<image>")

        # First part is before the first <image> placeholder
        content.append({"type": "text", "text": parts[0]})

        # Second part should be the source image
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{source_b64}"}
        })

        # Third part should be the edited image
        content.append({"type": "text", "text": parts[1]})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{edited_b64}"}
        })

        # Add any remaining text after the second image
        content.append({"type": "text", "text": parts[2]})

        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        return messages

    def parse_single_evaluation_response(self, response):
        """
        Parses the single evaluation response by first splitting the text into sections
        for each dimension using header-based partitioning, then extracting values.
        Also parses refinement comments and refined request.
        """
        # --- Helper: Value Extractor ---
        def _extract_values_from_block(block_text: str) -> Dict[str, Any]:
            """Parses a specific text block for one dimension to find scores and justifications."""
            data = {
                "score": None,
                "justification": ""
            }
            lines = block_text.split('\n')

            for line in lines:
                line = line.strip()
                if not line: continue

                # Extract score (supports float values)
                if "## Score:" in line:
                    score_text = line.split("## Score:")[-1].strip()
                    if score_text.upper() == "N/A":
                        data["score"] = "N/A"
                    else:
                        try:
                            # Try to extract float score
                            score_match = re.search(r'(\d+\.?\d*)', score_text)
                            if score_match:
                                data["score"] = float(score_match.group(1))
                        except ValueError:
                            data["score"] = score_text  # Keep as string if parsing fails

                # Extract justification
                elif "## Justification:" in line:
                    data["justification"] = line.split("## Justification:")[-1].strip()

            return data

        # --- Main Logic ---

        # 1. Initialize Result
        result = {
            "raw_response": response,
            "text_faithfulness": {},
            "image_faithfulness": {},
            "physical_quality": {},
            "text_rendering": {},
            "summary": "",
            "refinement_comments": "",
            "refined_request": ""
        }

        # 2. Pre-processing: Separate Summary and Refinement from Body
        content_body = response

        # Extract refinement section first
        if "User Request Refinement:" in response:
            parts = response.split("User Request Refinement:")
            if len(parts) > 1:
                content_body = parts[0]
                refinement_part = parts[1]

                # Parse refinement comments and refined request
                refinement_lines = refinement_part.split('\n')
                for line in refinement_lines:
                    line = line.strip()
                    if "Refinement Comments:" in line:
                        result["refinement_comments"] = line.split("Refinement Comments:")[-1].strip()
                    elif "Refined Request:" in line:
                        result["refined_request"] = line.split("Refined Request:")[-1].strip()

        # Extract summary
        if "Summary:" in content_body:
            parts = content_body.split("Summary:")
            if len(parts) > 1:
                content_body = parts[0]
                result["summary"] = parts[1].strip()

        # 3. Splitting: Cut the text into 4 blocks based on headers
        # We use partition() which splits string at the first occurrence of separator

        # Headers to look for
        h0 = "Detailed Judgement"
        h1 = "1. Text Faithfulness"
        h2 = "2. Image Faithfulness"
        h3 = "3. Physical and Visual Quality"
        h4 = "4. Text Rendering"

        # Logic: Find H1, everything after is rest. From rest, find H2...
        # This assumes the order is fixed (1->2->3->4)
        _, _, rest = content_body.partition(h0)
        _, _, rest = rest.partition(h1)
        block_tf, _, rest = rest.partition(h2)
        block_if, _, rest = rest.partition(h3)
        block_pq, _, rest = rest.partition(h4)
        block_tr = rest # The remainder is Text Rendering

        # Map blocks to keys
        sections = {
            "text_faithfulness": block_tf,
            "image_faithfulness": block_if,
            "physical_quality": block_pq,
            "text_rendering": block_tr
        }

        # 4. Parsing: Extract values from each block
        for key, block_text in sections.items():
            extracted_data = _extract_values_from_block(block_text)
            result[key] = extracted_data

        return result

    async def evaluate_single(self,
                              key,
                              eval_item,
                              semaphore,
                              generated_image_dir,
                              output_dir=None):
        """
        Evaluate a single edited image against its source
        """
        async with semaphore:
            # 1. Extract data from the new dataset format
            instruction_text = eval_item['instruction']
            source_image = eval_item['input_image_raw']
            task_type = eval_item['task_type']

            # 2. Load the edited image from the save path
            lang = 'en'
            edited_image_path = os.path.join(generated_image_dir, task_type, lang, f"{key}.png")
            # edited_image_path2 = os.path.join(generated_image_dir, task_type, f"{key}.png")

            # Check if edited image exists (async)
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=4)
            exists = await loop.run_in_executor(executor, os.path.exists, edited_image_path)
            if not exists:
                result = {
                    "key": key,
                    "instruction": instruction_text,
                    "success": False,
                    "error": f"Edited image not found at {edited_image_path}"
                }
                return result

            # Load edited image (async)
            try:
                edited_image = await loop.run_in_executor(executor, Image.open, edited_image_path)
                buf = BytesIO()
                await loop.run_in_executor(executor, edited_image.save, buf, "PNG")
                edited_image_bytes = buf.getvalue()
            except Exception as e:
                result = {
                    "key": key,
                    "instruction": instruction_text,
                    "success": False,
                    "error": f"Failed to load edited image: {e}"
                }
                return result
            finally:
                executor.shutdown(wait=False)

            # Convert source image to bytes
            try:
                if isinstance(source_image, dict) and "bytes" in source_image:
                    # Handle new "pica" dataset format where source_image is a dict with bytes
                    source_image_bytes = source_image["bytes"]
                else:
                    # Handle existing format where source_image is a PIL Image
                    buf = BytesIO()
                    source_image.save(buf, format="PNG")
                    source_image_bytes = buf.getvalue()
            except Exception as e:
                result = {
                    "key": key,
                    "instruction": instruction_text,
                    "success": False,
                    "error": f"Failed to process source image: {e}"
                }
                return result

            # Create result dict structure
            result = {
                "key": key,
                "instruction": instruction_text,
                "task_type": task_type,
                "edited_image_path": edited_image_path,
            }

            # Create messages for single evaluation
            messages = self.create_single_evaluation_message(
                instruction_text, source_image_bytes, edited_image_bytes
            )

            # Generate evaluation
            try:
                evaluation_response = await self.client.generate(messages)

                if evaluation_response:
                    parsed_result = self.parse_single_evaluation_response(evaluation_response)
                    result.update(parsed_result)
                    result["success"] = True
                else:
                    result["success"] = False
                    result["error"] = "No response from model"

            except Exception as e:
                print(f"Error evaluating item {key}: {e}")
                traceback.print_exc()
                result["success"] = False
                result["error"] = str(e)

            # --- Image Saving Logic ---
            if False:
                try:
                    # Save source image
                    source_filename = f"{key}_source.png"
                    source_filepath = os.path.join(output_dir, source_filename)
                    loop = asyncio.get_event_loop()
                    executor = ThreadPoolExecutor(max_workers=4)
                    try:
                        await loop.run_in_executor(executor, source_image.save, source_filepath)
                    finally:
                        executor.shutdown(wait=False)
                except Exception as e:
                    print(f"Failed to save source image for {key}: {e}")

                try:
                    # Save edited image copy
                    edited_filename = f"{key}_edited.png"
                    edited_filepath = os.path.join(output_dir, edited_filename)
                    loop = asyncio.get_event_loop()
                    executor = ThreadPoolExecutor(max_workers=4)
                    try:
                        await loop.run_in_executor(executor, edited_image.save, edited_filepath)
                    finally:
                        executor.shutdown(wait=False)
                except Exception as e:
                    print(f"Failed to save edited image for {key}: {e}")

            # Save result to the output directory (refined instruction folder)
            if output_dir:
                filename = f"{key}_refinement.json"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"written result to {filepath}")

            return result

async def perform_initial_check(inferencer):
    print("\n" + "="*50)
    print("🧪 PERFORMING INITIAL CHECK")
    print("="*50)

    # Create dummy images
    source_img = Image.new('RGB', (100, 100), color='gray')
    candidate_a_img = Image.new('RGB', (100, 100), color='blue')
    candidate_b_img = Image.new('RGB', (100, 100), color='green')

    buf_source = BytesIO()
    source_img.save(buf_source, format="PNG")
    source_bytes = buf_source.getvalue()

    buf_a = BytesIO()
    candidate_a_img.save(buf_a, format="PNG")
    candidate_a_bytes = buf_a.getvalue()

    buf_b = BytesIO()
    candidate_b_img.save(buf_b, format="PNG")
    candidate_b_bytes = buf_b.getvalue()

    instruction_text = "Change the background to blue or green."
    messages = inferencer.create_pairwise_evaluation_message(instruction_text, [source_bytes], candidate_a_bytes, candidate_b_bytes)

    print("Sending test pairwise request...")
    try:
        response = await inferencer.client.generate(messages)
        if response:
            parsed = inferencer.parse_pairwise_response(response)
            print("✅ Initial check completed successfully.")
        else:
            print("❌ Model returned no response.")
    except Exception as e:
        print(f"❌ Error during initial check: {e}")
    print("="*50 + "\n")

async def main():
    parser = argparse.ArgumentParser(description='Agentic Edit Evaluation using VLM')
    parser.add_argument('--generated-image-dir', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--dataname', type=str, choices=["pica", "imgedit", "gedit"], default="pica")
    parser.add_argument('--round', type=str, default='1')
    parser.add_argument('--server-host', type=str, default='http://localhost')
    parser.add_argument('--server-port', type=int, default=6868)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--concurrency', type=int, default=32)

    args = parser.parse_args()

    modelname = args.generated_image_dir.split('/')[-1]
    if not os.path.exists(args.generated_image_dir):
        raise FileNotFoundError(f"generated image dir does not exist: {args.generated_image_dir}")
    # Set output directory to generated_image_dir + "_refined_instruction"
    if args.output_dir is None:
        args.output_dir = f"{args.generated_image_dir}_{judgemodel}_{args.dataname}_refined_instruction"
        if args.round!='1':
            # round2 
            args.output_dir = f"{args.generated_image_dir}_round2_{args.dataname}_refined_instruction"
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    if args.dataname in ["imgedit"]:
        # Load parquet with pandas first to avoid schema inference issues
        import pandas as pd
        from datasets import Dataset

        df = pd.read_parquet(args.dataset_path)
        dataset = Dataset.from_pandas(df)
    elif args.dataname == "pica":
        dataset = load_dataset('parquet', data_files=args.dataset_path)['train']
        # 1.5. Ensure each row has a 'key' field
        def add_key_field(example, idx):
            if 'key' not in example:
                physics_law = example.get('physics_law', 'unknown')
                example['key'] = f"{idx}_{physics_law}"
            return example
        dataset = dataset.map(add_key_field, with_indices=True)
    else:
        dataset = load_from_disk(args.dataset_path)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    print(f"Dataset size: {len(dataset)}")

    # Verify required columns exist (dataset-specific)
    if args.dataname == "pica":
        required_cols = ['image_path', 'superficial_prompt', 'input_image', 'physics_category']
    elif args.dataname == "imgedit":
        required_cols = ['key', 'prompt', 'image', 'edit_type']
    else:  # gedit
        required_cols = ['key', 'instruction', 'input_image_raw', 'task_type']

    for col in required_cols:
        if col not in dataset.column_names:
            print(f"❌ Error: Required column '{col}' missing from dataset.")
            return

    # Initialize HTTP Client
    client = Qwen3VLInferenceHTTP(base_url=args.server_host, port=args.server_port)

    if not await client.check_connection():
        print("❌ Could not connect to VLM server.")
        return

    inferencer = EditRewardInference(client)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")

    # await perform_initial_check(inferencer)

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []

    print(f"Creating single evaluation tasks for {len(dataset)} items...")
    print(f"from image directory: {args.generated_image_dir}")
    for idx, item in enumerate(dataset):
        if args.dataname == "pica":
            # Create key from image_path or use fallback
            # key = item['image_path'].replace('/', '_').replace('.png', '') if 'image_path' in item else f"pica_{idx}"
            # Normalize column names for evaluate_single method
            key = item['key']
            eval_item = {
                'instruction': item['superficial_prompt'],
                'input_image_raw': item['input_image'],
                'task_type': item['physics_category']
            }
        elif args.dataname == "imgedit":
            key = item['key']
            # Normalize column names for evaluate_single method
            eval_item = {
                'instruction': item['prompt'],
                'input_image_raw': item['image'],
                'task_type': item['edit_type']
            }
        else:  # gedit
            key = item['key']
            eval_item = item

        # Create single evaluation task
        task = asyncio.create_task(
            inferencer.evaluate_single(key, eval_item, semaphore, args.generated_image_dir, args.output_dir)
        )
        tasks.append(task)

    print(f"Starting evaluation for {len(tasks)} items...")
    results = await tqdm.gather(*tasks)

    print(f"Evaluation completed. Results saved to {args.output_dir}")

    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"saved results to {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())