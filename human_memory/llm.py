"""Built-in LLM backends. Auto-detected from environment variables."""

import os
import json
import urllib.request
import urllib.error
from typing import Optional


def validate_deepseek_key(api_key: str, base_url: str = "https://api.deepseek.com/v1") -> tuple[bool, str]:
    """Quick validation: returns (is_valid, message)."""
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "密钥有效"
            return False, f"状态码 {resp.status}"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "密钥无效或已过期 (401 Unauthorized)"
        if e.code == 403:
            return False, "权限不足 (403 Forbidden)"
        return False, f"API 错误: {e.code}"
    except Exception as e:
        return False, f"网络不通: {e}"


def key_status() -> dict:
    """Check current API key status from env."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return {"set": False, "masked": "", "valid": False, "message": "未设置"}
    masked = key[:7] + "***" + key[-4:] if len(key) > 11 else "***"
    return {"set": True, "masked": masked, "valid": None, "message": "待验证（首次对话时检测）"}


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
        self._last_error: Optional[str] = None

    def validate(self) -> tuple[bool, str]:
        """Check if the current API key is valid."""
        if not self.api_key:
            return False, "未设置 API Key"
        return validate_deepseek_key(self.api_key, self.base_url)

    def __call__(self, prompt: str) -> str:
        if not self.api_key:
            return ("[错误] 未设置 DeepSeek API Key。\n"
                    "请在启动时输入 Key，或设置环境变量 DEEPSEEK_API_KEY。\n"
                    "获取 Key: https://platform.deepseek.com/api_keys")

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
            if e.code == 401:
                self._last_error = "密钥无效或已过期"
                return ("[密钥错误] API Key 无效或已过期。\n"
                        "请用 /key 命令更新: /key sk-xxxxxxxxxxxxxxxx")
            self._last_error = f"HTTP {e.code}"
            return f"[API 错误 {e.code}] {err_body[:300]}"
        except Exception as e:
            self._last_error = str(e)
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
