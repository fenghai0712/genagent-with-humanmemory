@echo off
chcp 65001 >nul
title genagent-with-humanmemory 安装

echo ========================================
echo   genagent-with-humanmemory 安装程序
echo   模拟人类记忆的 AI Agent 系统
echo ========================================
echo.

:: 检查 Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.11+
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/2] 正在安装 genagent-with-humanmemory ...
pip install git+https://github.com/fenghai0712/genagent-with-humanmemory.git
if %errorlevel% neq 0 (
    echo.
    echo [错误] 安装失败。尝试备用方式...
    echo 请手动运行: pip install git+https://github.com/fenghai0712/genagent-with-humanmemory.git
    pause
    exit /b 1
)

echo.
echo [2/2] 验证安装 ...
python -c "from human_memory import MemoryManager; print('安装成功!')"
if %errorlevel% neq 0 (
    echo [错误] 导入失败，请检查 Python 版本是否为 3.11+
    pause
    exit /b 1
)

echo.
echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 使用方法:
echo   1. 命令行对话:   python -m human_memory.agent
echo   2. Python 中使用: from human_memory import MemoryManager
echo.
echo 详细文档: https://github.com/fenghai0712/genagent-with-humanmemory
echo.
pause
