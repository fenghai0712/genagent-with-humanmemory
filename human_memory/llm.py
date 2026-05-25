"""Built-in LLM backends. Auto-detected from environment variables."""

import os
import json
import urllib.request
import urllib.error
from typing import Optional


class DeepSeekLLM:
    """DeepSeek API provider. Auto-detects DEEPSEEK_API_KEY from env.

    Usage:
        # auto-detect from env
        llm = DeepSeekLLM()

        # explicit key
        llm = DeepSeekLLM(api_key="sk-xxx")

        # custom model
        llm = DeepSeekLLM(model="deepseek-chat")
    """

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "deepseek-chat",
                 base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "")
                         or self.BASE_URL).rstrip("/")

    def __call__(self, prompt: str) -> str:
        if not self.api_key:
            return "[错误] 未设置 DEEPSEEK_API_KEY 环境变量，无法调用 DeepSeek。\n" \
                   "设置方式: set DEEPSEEK_API_KEY=sk-xxx  (Windows)\n" \
                   "          export DEEPSEEK_API_KEY=sk-xxx  (macOS/Linux)"

        url = f"{self.base_url}/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个有记忆的 AI 助手。你拥有工作记忆、情景记忆、语义记忆、程序记忆和方案记忆。请根据提供的记忆上下文回答用户问题。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            return f"[API 错误 {e.code}] {err_body[:300]}"
        except Exception as e:
            return f"[网络错误] {e}"


def auto_llm() -> Optional[callable]:
    """Auto-detect available LLM from environment variables.

    Checks in order:
        1. DEEPSEEK_API_KEY → DeepSeekLLM
        2. None → template mode
    """
    if os.environ.get("DEEPSEEK_API_KEY"):
        return DeepSeekLLM()
    return None
