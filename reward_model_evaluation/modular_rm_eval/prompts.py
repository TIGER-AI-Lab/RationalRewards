GEN_PROMPT_TEMPLATE = """You are an expert image evaluator. Your task is to evaluate the quality of two generated images based on a user instruction.

User Instruction: {request}
You are provided with two images:
1. Generated Image A <image>
2. Generated Image B <image>

Your task is to compare the two Generated Images according to the User Instruction.
To do this, you must compare the image on three critical aspects, provide absolute scores for each image and determine who wins.

### Critical Aspects & Scoring Rubric
**1. Text Faithfulness** (How accurately does the output follow the instruction?)
- **4 (Full match):** All key elements are represented exactly as described.
- **3 (Minor mismatch):** Most key elements are present, with only minor issues.
- **2 (Some mismatch):** Some key elements are missing, altered, or incorrect.
- **1 (Major deviations):** Key elements are missing or contradicted.

**2. Physical and Visual Quality**
- **4 (No noticeable flaws):** Realistic and artifact free.
- **3 (Minor flaws):** Slight but acceptable quality issues.
- **2 (Some flaws):** Visible errors that reduce quality.
- **1 (Severe flaws):** Major errors and obvious artifacts.

**3. Text Rendering**
- **4 (Full match):** Text is correct and legible.
- **3 (Mostly match):** Minor text errors.
- **2 (Partial match):** Major text errors.
- **1 (Major deviations):** Text unreadable or missing.

Output your evaluation in the following format:
# User Request Analysis
[brief analysis]
# Detailed Judgement
1. Text Faithfulness:
## Justification: [analysis]
## Score A: [float]
## Score B: [float]
## Winner: [Image A or Image B or It's a tie]
2. Physical and Visual Quality:
## Justification: [analysis]
## Score A: [float]
## Score B: [float]
## Winner: [Image A or Image B or It's a tie]
3. Text Rendering:
## Justification: [analysis]
## Score A: [float or N/A]
## Score B: [float or N/A]
## Winner: [N/A or Image A or Image B or It's a tie]
# Summary: [summary]
"""


EDIT_PROMPT_TEMPLATE = """You are an expert image editing evaluator. Your task is to evaluate the quality of an edited image based on a source image and a user instruction.

User Instruction: {request}
You are provided with three images:
1. Source Image <image>
2. Edited Image A <image>
3. Edited Image B <image>

Your task is to compare the two Edited Images according to the User Instruction and source image.
To do this, you must compare the image on four critical aspects, provide absolute scores for each image and determine who wins.

### Critical Aspects & Scoring Rubric
**1. Text Faithfulness** (How accurately does the output follow the instruction?)
**2. Image Faithfulness** (How well non-edited parts and key elements are preserved?)
**3. Physical and Visual Quality** (Technical quality and realism)
**4. Text Rendering** (Only if text generation is required)

Output your evaluation in the following format:
# User Request Analysis
[brief analysis]
# Detailed Judgement
1. Text Faithfulness:
## Justification: [analysis]
## Score A: [float]
## Score B: [float]
## Winner: [Image A or Image B or It's a tie]
2. Image Faithfulness:
## Justification: [analysis]
## Score A: [float]
## Score B: [float]
## Winner: [Image A or Image B or It's a tie]
3. Physical and Visual Quality:
## Justification: [analysis]
## Score A: [float]
## Score B: [float]
## Winner: [Image A or Image B or It's a tie]
4. Text Rendering:
## Justification: [analysis]
## Score A: [float or N/A]
## Score B: [float or N/A]
## Winner: [N/A or Image A or Image B or It's a tie]
# Summary: [summary]
"""

