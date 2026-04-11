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
import requests
from PIL import Image
from tqdm.asyncio import tqdm
import pandas as pd
import re
from glob import glob 
do_save_image = False
# Configuration
datafile = "data/genaibench_imgedit.parquet"
modelpath = "/path/to/model"
instruction = """You are an expert image editing evaluator. Your task is to evaluate the quality of an edited image based on a source image and a user instruction.

User Instruction: {request}
You are provided with three images:
1. Source Image <image>
2. Edited Image A <image>
3. Edited Image B <image>

Your task is to compare the two Edited Images according to the User Instruction and source image.
To do this, you must compare the image on four critical aspects, provide absolute scores for each image and determine who wins. 

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

### Scoring Methodology (CRITICAL)
During assessment for each aspect, recall the initial user request, source image and the scoring rubrics of the aspect, provide scores with detailed justifications for each image and reflect fine-grained preferences.  
1. **Anchor:** Have a global inspection. Determine the rough integer score level (1, 2, 3, or 4) according to the definitions provided (you can also refer to the given human preference or rating). 
2. **Justify and Adjust:** Do careful visual analysis and identify specific flaws in generation. Justify the score with concrete evidence and scoring logic. Fine-tune this anchor score into a float value. Add small increments for exceptional execution or deduct points for specific flaws. 
   - *Example:* deduct points from 4.0 for slight flaws if the assessed dimension is close to satisfaction. add increments from 1.0 or 2.0 based on severity of flaws.
3. **Compare:** Ensure the difference between Score A and Score B reflects the correct preference.  

Output your evaluation in the following format:
# User Request Analysis 
[ understanding the user request, and what needs to be considered during image editing ]
# Detailed Judgement
1. Text Faithfulness: 
## Justification: [ Comparative Analysis: Given the request, source image and the scoring rubrics, which image is better in this dimension? Provide concrete evidence and scoring logic. e.g., Image A is roughly [X] score level because [reason]. Deduct/Add points for [specific details] to reach final score. ]
## Score A: [float score for Image A]
## Score B: [float score for Image B]
## Winner: [Image A or Image B or It's a tie]
2. Image Faithfulness: 
## Justification: [ Similar to above. Comparative analysis with concrete evidence and scoring logic for image faithfulness. ]
## Score A: [float score for Image A]
## Score B: [float score for Image B]
## Winner: [Image A or Image B or It's a tie] 
3. Physical and Visual Quality: 
## Justification: [ Similar to above. Comparative analysis with concrete evidence and scoring logic. Since physical/visual quality is often not perfect, give 4.0 sparingly only when it is perfectly realistic. ]
## Score A: [float score for Image A]
## Score B: [float score for Image B]
## Winner: [Image A or Image B or It's a tie]
4. Text Rendering: 
## Justification: [ Similar to above. Comparative analysis with concrete evidence and scoring logic. Since text rendering is often challenging, give 4.0 sparingly only if it is perfect. ]
## Score A: [float score for Image A]
## Score B: [float score for Image B]
## Winner: [N/A or Image A or Image B or It's a tie]
# Summary: [Summary of the evaluation]"""

# instruction = """You are an expert image editing evaluator. Your task is to evaluate the quality of an edited image based on a source image and a user instruction.

# User Instruction: {request}
# You are provided with three images:
# 1. Source Image <image>
# 2. Edited Image A <image>
# 3. Edited Image B <image>

# Your task is to compare the two Edited Images according to the User Instruction and source image.
# To do this, you must compare the image on four critical aspects, provide absolute scores for each image and determine who wins.
# About the scores: you should try to give float scores. For example, float values are important to reflect fine-grained preferences between the two images.

# ### Critical Aspects & Scoring Rubric
# **1. Text Faithfulness** (How accurately does the output follow the instruction?)
# - **4 (Full match):** All key elements (objects, colors, actions) are represented exactly as described. No hallucinations or unrequested changes.
# - **3 (Minor mismatch):** Most key elements are present, but minor details are missing, incorrect, or slightly inaccurate.
# - **2 (Some mismatch):** Some key elements are missing, altered, or interpreted incorrectly.
# - **1 (Major deviations):** Key elements are completely missing, altered, or contradicted. Instruction is ignored.

# **2. Image Faithfulness** (How well are the non-edited parts and key input elements preserved?)
# - **4 (Uses input fully):** All relevant elements from the input (background, style, lighting, identity) are accurately preserved or transformed as instructed.
# - **3 (Minor mismatch):** Most relevant elements are preserved, but a few aspects (e.g., background details, lighting consistency) are missing or incorrectly handled.
# - **2 (Partial mismatch):** Some elements are carried over, but key aspects of the original image are lost or distorted.
# - **1 (Fails to use input):** Key elements of the input image are ignored, misinterpreted, or destroyed.

# **3. Physical and Visual Quality** (Technical errors, composition, realism, and physics)
# - **4 (No noticeable flaws):** The image is physically plausible (correct lighting, shadows, geometry, anatomy). No visible artifacts (seams, blurring, noise).
# - **3 (Minor flaws):** Small inaccuracies that are noticeable but not strongly disruptive (e.g., slight lighting mismatch, minor texture issues).
# - **2 (Some flaws):** Clear physical or visual errors that disrupt the image (e.g., incorrect perspective, "floating" objects, wrong shadow direction, obvious seams).
# - **1 (Severe flaws):** Major physical/visual errors (e.g., impossible geometry, distorted anatomy, garbled objects, severe artifacts).

# **4. Text Rendering** (Only if the instruction involves generating text)
# - **4 (Full match):** Text is correct, legible, and integrated well.
# - **3 (Mostly match):** Minor misspellings or inconsistent capitalization.
# - **2 (Partial match):** Major misspellings or distorted text.
# - **1 (Major deviations):** Text is unreadable, severely distorted, or missing. (Use N/A if no text generation is required).

# Output your evaluation in the following format:
# [ understanding the user request, and what needs to be considered during image editing ]
# # Detailed Judgement
# 1. Text Faithfulness:
# ## Justification: [Detailed explanation of the score]
# ## Score A: [float score for Image A]
# ## Score B: [float score for Image B]
# ## Winner: [Image A or Image B or It's a tie]
# 2. Image Faithfulness:
# ## Justification: [Detailed explanation of the score]
# ## Score A: [float score for Image A]
# ## Score B: [float score for Image B]
# ## Winner: [Image A or Image B or It's a tie]
# 3. Physical and Visual Quality:
# ## Justification: [Detailed explanation of the score]
# ## Score A: [float score for Image A]
# ## Score B: [float score for Image B]
# ## Winner: [Image A or Image B or It's a tie]
# 4. Text Rendering:
# ## Justification: [Detailed explanation of the score]
# ## Score A: [float score for Image A]
# ## Score B: [float score for Image B]
# ## Winner: [N/A or Image A or Image B or It's a tie]
# # Summary: [Summary of the evaluation]"""

class Qwen3VLInferenceHTTP:
    def __init__(self,
                 model_name="Qwen3-VL-8B-Instruct",
                 base_url="http://localhost",
                 port=8000,
                 timeout=300,
                 api_key=None,
                 is_api_server=False):
        self.model_name = model_name
        self.base_url = base_url if is_api_server else f"{base_url}:{port}"
        self.timeout = timeout
        self.api_key = api_key
        self.is_api_server = is_api_server

        if is_api_server:
            self.generate_endpoint = base_url  # API server uses the full URL directly
            self.health_endpoint = None  # API servers might not have a health endpoint
        else:
            self.generate_endpoint = f"{self.base_url}/v1/chat/completions"
            self.health_endpoint = f"{self.base_url}/v1/models"

        print(f"Initialized {'API' if is_api_server else 'Local'} HTTP client for model: {model_name}")

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

    async def generate(self, messages, temperature=0.1, max_tokens=20480, top_p=0.9, top_k=40, logprobs=False):
        if self.is_api_server:
            # API server request (similar to call_model_api function)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "stream": False,
                "model": self.model_name,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "logprobs": logprobs,
                "messages": messages,
            }
            # Special handling for Gemini models as per user snippet
            if "gemini" in self.model_name.lower():
                payload["extra_body"] = {
                    "google": {
                        "thinking_config": {
                            "include_thoughts": True,
                            "thinking_budget": 24576.0,
                        },
                        "thought_tag_marker": "think",
                    }
                }

            try:
                response = requests.post(self.generate_endpoint, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return None
            except Exception as e:
                print(f"Error during API generation: {e}")
                return None
        else:
            # Local server request (original implementation)
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
                    print(f"Error during local generation: {e}")
                    return None

class EditRewardInference:
    def __init__(self, client):
        self.client = client


    def create_pairwise_evaluation_message(self, instruction_text, source_image_bytes_list,
                                           candidate_a_bytes, candidate_b_bytes):
        """Create the evaluation message for pairwise image comparison"""

        system_prompt = "You are an expert image editing evaluator."

        user_content_text = instruction.format(request=instruction_text)

        # Convert bytes to base64
        source_b64_list = [base64.b64encode(img_bytes).decode() for img_bytes in source_image_bytes_list]
        candidate_a_b64 = base64.b64encode(candidate_a_bytes).decode()
        candidate_b_b64 = base64.b64encode(candidate_b_bytes).decode()

        # Interleave images into the <image> placeholders in user_content_text
        content = []
        parts = user_content_text.split("<image>")

        # Handle the first <image> placeholder with multiple source images
        content.append({"type": "text", "text": parts[0]})
        for source_b64 in source_b64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{source_b64}"}
            })

        # Add the remaining parts with single images
        for i in range(1, len(parts)):
            content.append({"type": "text", "text": parts[i]})
            if i == 1:  # Second <image> placeholder (Edited Image A)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{candidate_a_b64}"}
                })
            elif i == 2:  # Third <image> placeholder (Edited Image B)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{candidate_b_b64}"}
                })

        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        return messages

    def parse_evaluation_response(self, response):
        try:
            result = {
                "raw_response": response,
                "text_faithfulness_score": None,
                "text_faithfulness_justification": "",
                "image_faithfulness_score": None,
                "image_faithfulness_justification": "",
                "physical_quality_score": None,
                "physical_quality_justification": "",
                "text_rendering_score": None,
                "text_rendering_justification": "",
                "summary": ""
            }

            if "# Detailed Judgement" in response:
                parts = response.split("# Detailed Judgement")
                if len(parts) > 1:
                    remaining = parts[1]
                    if "# Summary:" in remaining:
                        judgement_part, summary_part = remaining.split("# Summary:")
                        result["summary"] = summary_part.strip()
                        remaining = judgement_part
                    
                    aspects = remaining.split("\n")
                    current_aspect = None

                    for line in aspects:
                        line = line.strip()
                        if not line: continue

                        if "1. Text Faithfulness:" in line: current_aspect = "text_faithfulness"
                        elif "2. Image Faithfulness:" in line: current_aspect = "image_faithfulness"
                        elif "3. Physical and Visual Quality:" in line: current_aspect = "physical_quality"
                        elif "4. Text Rendering:" in line: current_aspect = "text_rendering"
                        elif line.startswith("## Score:"):
                            score_text = line.replace("## Score:", "").strip()
                            if current_aspect:
                                if score_text.upper() == "N/A":
                                    result[f"{current_aspect}_score"] = "N/A"
                                else:
                                    try:
                                        score = int(re.search(r'\d+', score_text).group())
                                        if 1 <= score <= 4:
                                            result[f"{current_aspect}_score"] = score
                                    except: pass
                        elif line.startswith("## Justification:"):
                            justification = line.replace("## Justification:", "").strip()
                            if current_aspect:
                                result[f"{current_aspect}_justification"] = justification
            return result
        except Exception as e:
            return {"raw_response": response, "error": str(e)}

    def parse_pairwise_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the evaluation response by first splitting the text into sections
        for each dimension, then extracting values.
        Triggers a breakpoint if required fields are missing.
        """

        # --- Helper: Value Extractor ---
        def _extract_values_from_block(block_text: str) -> Dict[str, Any]:
            """Parses a specific text block for one dimension to find scores."""
            data = {}
            lines = block_text.split('\n')

            for line in lines:
                line = line.strip()
                # Robust extraction logic avoiding regex
                if "Score A" in line and "##" in line:
                    # Split by the last occurrence of 'Score A' or ':'
                    # Handles: "## Score A: 4" or "## Score A 4"
                    raw_val = line.split("Score A")[-1].replace(":", "").strip()
                    try:
                        data["score_a"] = float(raw_val)
                    except ValueError:
                        data["score_a"] = raw_val # Keep as string if N/A

                elif "Score B" in line and "##" in line:
                    raw_val = line.split("Score B")[-1].replace(":", "").strip()
                    try:
                        data["score_b"] = float(raw_val)
                    except ValueError:
                        data["score_b"] = raw_val

                elif "Winner" in line and "##" in line:
                    data["winner"] = line.split("Winner")[-1].replace(":", "").strip()

                # elif "Justification" in line and "##" in line:
                    # data["justification"] = line.split("Justification")[-1].replace(":", "").strip()
            return data

        # --- Main Logic ---

        # 1. Initialize Result
        result = {
            "raw_response": response,
            "text_faithfulness": {},
            "image_faithfulness": {},
            "physical_quality": {},
            "text_rendering": {},
            "summary": ""
        }

        # 2. Pre-processing: Separate Summary from Body
        # We do this first so the summary text doesn't interfere with the last dimension
        content_body = response
        if "# Summary:" in response:
            parts = response.split("# Summary:")
            if len(parts) > 1:
                content_body = parts[0]
                result["summary"] = parts[1].strip()

        # 3. Splitting: Cut the text into 4 blocks based on headers
        # We use partition() which splits string at the first occurrence of separator

        # Headers to look for
        h0 = "Detailed Judgement"
        h1 = "Text Faithfulness"
        h2 = "Image Faithfulness"
        h3 = "Physical and Visual Quality"
        h4 = "Text Rendering"

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
            # Debugging Tip: If values are missing, check 'block_text' here
            extracted_data = _extract_values_from_block(block_text)
            for k in ['score_a', 'score_b']:
                if k not in extracted_data:
                    raise ValueError(f"Missing required field '{k}' in {key} block - skipping this example")
            result[key] = extracted_data
            # validate({key: block_text}, result)

        # print(json.dumps(result, indent=2))
        return result


    async def evaluate_pairwise(self,
                               pair_id,
                               eval_item,
                               semaphore,
                               output_dir):
        """
        Evaluate both candidates A and B in a pairwise comparison
        """
        async with semaphore:
            # 1. Extract Instruction
            instruction_text = eval_item['prompt_text']

            # 2. Extract Source Images (handle multiple images)
            source_image_bytes = [img['bytes'] for img in eval_item['prompt_images']]

            # 3. Extract both candidate images
            candidate_a_bytes = eval_item['response_a_images'][0]['bytes']
            candidate_b_bytes = eval_item['response_b_images'][0]['bytes']

            # Create result dict structure
            result = {
                "pair_id": pair_id,
                "instruction": instruction_text,
                "chosen": eval_item.get('chosen', None), # Ground truth ranking info
                "response_a_model": eval_item.get('response_a_model', ''),
                "response_b_model": eval_item.get('response_b_model', ''),
                "num_source_images": len(source_image_bytes),
            }

            # Create messages for pairwise comparison
            messages = self.create_pairwise_evaluation_message(
                instruction_text, source_image_bytes, candidate_a_bytes, candidate_b_bytes
            )

            # Generate evaluation
            try:
                evaluation_response = await self.client.generate(messages)

                if evaluation_response:
                    parsed_result = self.parse_pairwise_response(evaluation_response)
                    result.update(parsed_result)
                    result["success"] = True
                else:
                    result["success"] = False
                    result["error"] = "No response from model"

            except Exception as e:
                print(f"Error evaluating pair {pair_id}: {e}")
                traceback.print_exc()
                result["success"] = False
                result["error"] = str(e)

            # --- Image Saving Logic ---
            # Save source images (handle multiple images)
            if do_save_image:
                try:
                    for idx, img_bytes in enumerate(source_image_bytes):
                        source_image = Image.open(BytesIO(img_bytes))
                        source_filename = f"{pair_id}_source_{idx}.png" if len(source_image_bytes) > 1 else f"{pair_id}_source.png"
                        source_filepath = os.path.join(output_dir, source_filename)
                        source_image.save(source_filepath)
                except Exception as e:
                    print(f"Failed to save source image(s) for {pair_id}: {e}")

                # Save candidate A image
                try:
                    candidate_a_image = Image.open(BytesIO(candidate_a_bytes))
                    candidate_a_filename = f"{pair_id}_a.png"
                    candidate_a_filepath = os.path.join(output_dir, candidate_a_filename)
                    candidate_a_image.save(candidate_a_filepath)
                except Exception as e:
                    print(f"Failed to save candidate A image {pair_id}: {e}")

                # Save candidate B image
                try:
                    candidate_b_image = Image.open(BytesIO(candidate_b_bytes))
                    candidate_b_filename = f"{pair_id}_b.png"
                    candidate_b_filepath = os.path.join(output_dir, candidate_b_filename)
                    candidate_b_image.save(candidate_b_filepath)
                except Exception as e:
                    print(f"Failed to save candidate B image {pair_id}: {e}")

            # Save result
            filename = f"{pair_id}_pairwise.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return result

def normalize_dataset_columns(df):
    """
    Normalize different dataset formats to a common structure.
    Supports aaMultimodal-RewardBench-2 and aaGenAI-Bench formats.
    """
    # Check if it's aaGenAI-Bench format
    if 'source_prompt' in df.columns and 'target_prompt' in df.columns:
        print("Detected aaGenAI-Bench format, normalizing columns...")

        # Create normalized dataframe
        normalized_df = pd.DataFrame()

        # Create pair_id from available data
        if 'pair_id' not in df.columns:
            # Generate pair_id from index or other unique identifier
            normalized_df['pair_id'] = [f"genaibench_{i}" for i in range(len(df))]

        # Combine prompts for instruction text
        normalized_df['prompt_text'] = df.apply(
            lambda row: f"{row['instruct_prompt']}",
            axis=1
        )

        # Normalize image columns
        normalized_df['prompt_images'] = df['source_image'].apply(lambda x: [x] if isinstance(x, dict) else x)
        normalized_df['response_a_images'] = df['left_output_image'].apply(lambda x: [x] if isinstance(x, dict) else x)
        normalized_df['response_b_images'] = df['right_output_image'].apply(lambda x: [x] if isinstance(x, dict) else x)

        # Map model names
        normalized_df['response_a_model'] = df['left_model']
        normalized_df['response_b_model'] = df['right_model']

        # Map chosen based on vote_type
        def map_vote_type(vote_type):
            if vote_type == 'leftvote':
                return 'a'
            elif vote_type == 'rightvote':
                return 'b'
            else:
                return None  # for bothbad_vote or other cases

        normalized_df['chosen'] = df['vote_type'].apply(map_vote_type)

        return normalized_df

    # Check if it's already in the expected aaMultimodal-RewardBench-2 format
    elif all(col in df.columns for col in ['pair_id', 'prompt_text', 'prompt_images', 'response_a_images', 'response_b_images']):
        print("Dataset already in expected format (aaMultimodal-RewardBench-2)")
        return df

    else:
        raise ValueError(f"Unrecognized dataset format. Columns found: {df.columns.tolist()}")


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
    parser = argparse.ArgumentParser(description='EditReward Inference using VLM')
    parser.add_argument('--data-file', type=str, default=datafile)
    parser.add_argument('--model-path', type=str, default=modelpath)
    parser.add_argument('--server-host', type=str, default='http://localhost')
    parser.add_argument('--server-port', type=int, default=6868)
    parser.add_argument('--api-url', type=str, default=None, help='API server URL when using --is-api-server')
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--model-name', type=str, default=None, help='Model name for API server')
    parser.add_argument('--is-api-server', action='store_true', help='Use API server instead of local server')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--concurrency', type=int, default=32)

    parser.add_argument('--evalsetting', type=str, default=None)

    args = parser.parse_args()
    
    evalsetting = args.evalsetting

    if args.output_dir is None:
        args.output_dir = f"evalresults/{evalsetting}"

    # Load data
    print(f"Loading dataset from {args.data_file}")

    # Check if data_file is a directory or a single file
    data_path = Path(args.data_file)
    if data_path.is_dir():
        # Load all parquet files from directory
        dflist = []
        for file in glob(str(data_path / "*.parquet")):
            dflist.append(pd.read_parquet(file))
        df = pd.concat(dflist) if dflist else pd.DataFrame()
    else:
        # Load single parquet file
        df = pd.read_parquet(args.data_file)

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Limited to {args.max_samples} samples")

    print(f"Dataset shape: {df.shape}")

    # Detect dataset format and normalize columns
    df = normalize_dataset_columns(df)

    # Verify required columns exist
    required_cols = ['pair_id', 'prompt_text', 'prompt_images', 'response_a_images', 'response_b_images']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Error: Required column '{col}' missing from dataset.")
            return

    # Initialize HTTP Client
    if args.is_api_server:
        client = Qwen3VLInferenceHTTP(
            base_url=getattr(args, 'api_url', args.server_host),
            api_key=args.api_key,
            is_api_server=args.is_api_server,
            model_name=getattr(args, 'model_name', "Qwen3-VL-8B-Instruct")
        )
    else:
        client = Qwen3VLInferenceHTTP(
            base_url=args.server_host,
            port=args.server_port,
            api_key=args.api_key,
            is_api_server=args.is_api_server,
        )

        if not await client.check_connection():
            print("❌ Could not connect to VLM server.")
            return

    inferencer = EditRewardInference(client)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")

    # Check for already processed files and skip them
    existing_files = glob(os.path.join(args.output_dir, "*_pairwise.json"))
    processed_pair_ids = set()

    for filepath in existing_files:
        filename = os.path.basename(filepath)
        if filename.endswith("_pairwise.json"):
            pair_id = filename[:-len("_pairwise.json")]
            processed_pair_ids.add(pair_id)

    if processed_pair_ids:
        original_count = len(df)
        df = df[~df['pair_id'].isin(processed_pair_ids)]
        skipped_count = original_count - len(df)
        print(f"Found {skipped_count} already processed pairs, skipping them.")
        print(f"Remaining pairs to process: {len(df)}")
    else:
        print("No previously processed files found, processing all pairs.")

    if len(df) == 0:
        print("All pairs have already been processed. Exiting.")
        return

    await perform_initial_check(inferencer)

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []

    print(f"Creating pairwise evaluation tasks for {len(df)} pairs...")

    for _, row in df.iterrows():
        # Get pair_id safely (convert to string)
        pair_id = str(row['pair_id'])

        # Create single pairwise task for both candidates
        task = asyncio.create_task(
            inferencer.evaluate_pairwise(pair_id, row, semaphore, args.output_dir)
        )
        tasks.append(task)

    print(f"Starting evaluation for {len(tasks)} pairwise comparisons...")
    results = await tqdm.gather(*tasks)

    print(f"Evaluation completed. Results saved to {args.output_dir}")

    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())