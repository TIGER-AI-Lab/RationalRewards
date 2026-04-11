import json
from typing import Any, Dict, List, Optional

import aiohttp
import requests


class VLMHTTPClient:
    def __init__(
        self,
        model_name: str = "Qwen3-VL-8B-Instruct",
        base_url: str = "http://localhost",
        port: int = 6868,
        timeout: int = 300,
        api_key: Optional[str] = None,
        is_api_server: bool = False,
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key
        self.is_api_server = is_api_server

        if is_api_server:
            self.generate_endpoint = base_url
            self.health_endpoint = None
        else:
            host = f"{base_url}:{port}"
            self.generate_endpoint = f"{host}/v1/chat/completions"
            self.health_endpoint = f"{host}/v1/models"

    async def check_connection(self) -> bool:
        if self.is_api_server:
            return True
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.health_endpoint, timeout=10) as response:
                    return response.status == 200
        except Exception:
            return False

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 20480,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> Optional[str]:
        if self.is_api_server:
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
                "messages": messages,
            }
            try:
                resp = requests.post(
                    self.generate_endpoint,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("choices"):
                    return data["choices"][0]["message"]["content"]
                return None
            except Exception:
                return None

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.generate_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        return None
                    data = await response.json()
                    if data.get("choices"):
                        return data["choices"][0]["message"]["content"]
                    return None
        except Exception:
            return None

