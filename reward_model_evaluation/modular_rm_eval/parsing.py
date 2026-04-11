import re
from typing import Any, Dict, List


def _extract_score(line: str):
    raw = line.split(":", 1)[-1].strip()
    match = re.search(r"(\d+(?:\.\d+)?)", raw)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return raw
    if raw.upper() == "N/A":
        return "N/A"
    return raw


def _extract_block_values(block_text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in block_text.splitlines():
        text = line.strip()
        if "Score A" in text:
            data["score_a"] = _extract_score(text)
        elif "Score B" in text:
            data["score_b"] = _extract_score(text)
        elif "Winner" in text and "##" in text:
            data["winner"] = text.split("Winner", 1)[-1].replace(":", "").strip()
    if "score_a" not in data:
        data["score_a"] = "N/A"
    if "score_b" not in data:
        data["score_b"] = "N/A"
    return data


def parse_pairwise_response(response: str, aspects: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "raw_response": response,
        "summary": "",
    }
    for aspect in aspects:
        result[aspect] = {}

    body = response
    if "# Summary:" in response:
        head, tail = response.split("# Summary:", 1)
        body = head
        result["summary"] = tail.strip()

    _, _, rest = body.partition("Detailed Judgement")
    section_texts: Dict[str, str] = {}
    for idx, aspect in enumerate(aspects):
        order = idx + 1
        title = aspect.replace("_", " ").title()
        marker = f"{order}. {title}"
        _, _, after = rest.partition(marker)
        if idx + 1 < len(aspects):
            next_title = aspects[idx + 1].replace("_", " ").title()
            next_marker = f"{order + 1}. {next_title}"
            section, _, _ = after.partition(next_marker)
        else:
            section = after
        section_texts[aspect] = section

    for aspect, block in section_texts.items():
        result[aspect] = _extract_block_values(block)
    return result

