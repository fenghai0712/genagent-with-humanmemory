"""genagent-with-humanmemory — Windows Online Installer

Downloads and installs everything via Chinese mirrors. No VPN needed.

To compile to standalone exe:
    pip install pyinstaller
    pyinstaller --onefile --console --name genagent-installer installer.py
"""

import subprocess
import sys
import os
import shutil
import tempfile
import urllib.request
import urllib.error
import json
from pathlib import Path

# ── Mirror configuration ──
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
HF_MIRROR = "https://hf-mirror.com"
GITHUB_REPO = "https://github.com/fenghai0712/genagent-with-humanmemory.git"
GITEE_REPO = "https://gitee.com/fenghai0712/genagent-with-humanmemory.git"

PACKAGE_NAME = "genagent-with-humanmemory"
PYTHON_MIN = (3, 11)
REQUIRED_PACKAGES = ["sqlite-vec>=0.1.0", "numpy>=1.24.0", "sentence-transformers>=3.0.0"]


def banner():
    print("=" * 56)
    print("  genagent-with-humanmemory  —  在线安装程序")
    print("  模拟人类记忆的 AI Agent 系统")
    print("=" * 56)
    print()


def check_python() -> str | None:
    """Find a usable Python interpreter."""
    candidates = ["python", "python3", "py"]
    for exe in candidates:
        exe_path = shutil.which(exe)
        if exe_path:
            try:
                result = subprocess.run(
                    [exe_path, "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    version_str = result.stdout.strip()  # "Python 3.11.x"
                    parts = version_str.replace("Python ", "").split(".")
                    major, minor = int(parts[0]), int(parts[1])
                    if (major, minor) >= PYTHON_MIN:
                        return exe_path
                    else:
                        print(f"[警告] {exe} 版本 {major}.{minor} < 3.11，跳过")
            except Exception:
                pass
    return None


def run(cmd: list[str], desc: str = "", timeout: int = 300) -> bool:
    """Run a command with progress display."""
    label = f"[{desc}]" if desc else ""
    print(f"  {label} {' '.join(cmd[:3])}..." + (" ..." if len(cmd) > 3 else ""))
    try:
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [FAIL] 退出码 {e.returncode}")
        return False
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False


def test_mirror(url: str, label: str) -> bool:
    """Quick connectivity test to a mirror."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=5)
        print(f"  [OK] {label}: {url}")
        return True
    except Exception:
        print(f"  [WARN] {label} 不通: {url}")
        return False


def install_package(python_exe: str) -> bool:
    """Install via pip with Tsinghua mirror."""
    print("\n[步骤 2/4] 安装 Python 依赖 (清华镜像)...")

    # Upgrade pip first
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip",
         "-i", PIP_MIRROR, "--quiet"], "pip 升级")

    # Install the package from GitHub with mirror
    cmd = [python_exe, "-m", "pip", "install",
           f"git+{GITHUB_REPO}",
           "-i", PIP_MIRROR]
    return run(cmd, "安装 human-memory")


def install_from_local(python_exe: str, src_dir: str) -> bool:
    """Fallback: install from local source directory."""
    print("\n[步骤 2/4] 从本地安装...")
    cmd = [python_exe, "-m", "pip", "install", src_dir, "-i", PIP_MIRROR]
    return run(cmd, "本地安装")


def setup_env():
    """Set environment variables for Chinese mirrors."""
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", HF_MIRROR)
    os.environ["PIP_INDEX_URL"] = PIP_MIRROR
    print(f"\n[步骤 1/4] 配置镜像源")
    print(f"  PyPI: {PIP_MIRROR}")
    print(f"  Hugging Face: {HF_MIRROR}")


def verify_install(python_exe: str) -> bool:
    """Smoke test the installation."""
    print("\n[步骤 3/4] 验证安装...")
    test_code = """
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
try:
    from human_memory import MemoryManager, MemoryConfig, DeepSeekLLM
    from human_memory.agent import MemoryAgent
    print("  [OK] 所有模块导入成功")
    # Quick functional test (no LLM)
    mm = MemoryManager()
    mm.remember("安装测试", explicit_signal=True)
    mm.consolidate()
    results = mm.recall("测试", limit=1)
    print(f"  [OK] 记忆系统功能正常 (检索到 {len(results)} 条)")
    mm.close()
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)
"""
    result = subprocess.run(
        [python_exe, "-c", test_code],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "HF_ENDPOINT": HF_MIRROR},
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True


def _validate_key(key: str) -> tuple:
    """Inline key validation — avoids importing human_memory at build time."""
    url = "https://api.deepseek.com/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "密钥有效"
            return False, f"状态码 {resp.status}"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "密钥无效或已过期 (401)"
        if e.code == 403:
            return False, "权限不足 (403)"
        return False, f"API 错误: {e.code}"
    except Exception as e:
        return False, f"网络不通: {e}"


def _save_key(key: str) -> None:
    """Inline key persistence — avoids importing human_memory at build time."""
    config_dir = Path.home() / ".genagent"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "env"
    config_file.write_text(
        f"# genagent config\nDEEPSEEK_API_KEY={key}\n", encoding="utf-8")


def setup_api_key():
    """Prompt for DeepSeek API key, with skip option."""
    print("\n[步骤 4/5] DeepSeek API Key")
    print("  获取 Key: https://platform.deepseek.com/api_keys")
    print("  输入 Key 以启用 AI 对话，或按回车跳过（可之后设置）")

    key = input("  API Key: ").strip()
    if not key:
        print("  已跳过。之后可通过以下方式设置:")
        print("    set DEEPSEEK_API_KEY=sk-xxx")
        print("    或在 memory-agent 启动时输入")
        return

    print("  正在验证...")
    valid, msg = _validate_key(key)
    if valid:
        print(f"  [OK] {msg}")
        _save_key(key)
        print(f"  Key 已保存到 ~/.genagent/env")
    else:
        print(f"  [FAIL] {msg}")
        print("  Key 未保存。启动 memory-agent 时可重新输入。")


def show_usage():
    print("\n[步骤 5/5] 安装完成！")
    print("=" * 56)
    print()
    print("  使用方法:")
    print()
    print("    1. 命令行启动 (需要先设 API Key):")
    print("       set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx")
    print("       memory-agent")
    print()
    print("    2. Python 中使用:")
    print("       from human_memory.agent import MemoryAgent")
    print("       agent = MemoryAgent()")
    print("       agent.run(\"你好\")")
    print()
    print("    3. 获取 DeepSeek API Key:")
    print("       https://platform.deepseek.com/api_keys")
    print()
    print("  文档: https://github.com/fenghai0712/genagent-with-humanmemory")
    print("=" * 56)


def main():
    banner()

    # Step 0: find Python
    python = check_python()
    if not python:
        print("[错误] 未找到 Python 3.11+")
        print("请先安装 Python: https://www.python.org/downloads/")
        print('安装时勾选 "Add Python to PATH"')
        print()
        print("或使用国内镜像下载:")
        print("  https://registry.npmmirror.com/binary.html?path=python/")
        input("\n按回车退出...")
        return 1

    print(f"[OK] Python: {python}")
    subprocess.run([python, "--version"])

    # Step 1: configure mirrors
    setup_env()
    test_mirror(PIP_MIRROR + "/numpy/", "PyPI 清华镜像")
    test_mirror(HF_MIRROR, "HuggingFace 镜像")

    # Step 2: install
    print()
    # Find if we're running from the source directory
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "setup.py").exists() or (script_dir / "pyproject.toml").exists():
        success = install_from_local(python, str(script_dir))
    else:
        success = install_package(python)

    if not success:
        print("\n[错误] 安装失败。请检查网络连接后重试。")
        print("如果持续失败，尝试手动安装:")
        print("  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \\")
        print(f"    git+{GITHUB_REPO}")
        input("\n按回车退出...")
        return 1

    # Step 3: verify
    if not verify_install(python):
        print("\n[警告] 验证未通过，但安装文件已就位。")
        print("请尝试运行: python -m human_memory.agent")
        input("\n按回车退出...")
        return 1

    # Step 4: API key
    setup_api_key()

    # Step 5: done
    show_usage()
    input("\n按回车退出...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
