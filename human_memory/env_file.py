"""Read/write key-value config file at ~/.genagent/env. Used for API key persistence."""

import os
from pathlib import Path


def config_dir() -> Path:
    """~/.genagent/ on all platforms."""
    d = Path.home() / ".genagent"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    return config_dir() / "env"


def read_config() -> dict[str, str]:
    """Read key=value pairs from ~/.genagent/env. Comments with #."""
    cfg = {}
    p = config_path()
    if not p.exists():
        return cfg
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def write_config(updates: dict[str, str]) -> None:
    """Merge updates into the config file. Creates if not exists."""
    cfg = read_config()
    cfg.update(updates)
    lines = [
        "# genagent config — created automatically",
        "# Edit directly or use: memory-agent --set-key sk-xxx",
        "",
    ]
    for k, v in sorted(cfg.items()):
        lines.append(f"{k}={v}")
    config_path().write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_api_key() -> str:
    """Get DeepSeek API key: env var > config file > empty."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    return read_config().get("DEEPSEEK_API_KEY", "")


def save_api_key(key: str) -> None:
    """Persist API key to config file."""
    write_config({"DEEPSEEK_API_KEY": key})
    os.environ["DEEPSEEK_API_KEY"] = key
