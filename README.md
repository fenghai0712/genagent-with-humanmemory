# genagent-with-humanmemory

模拟人类记忆的 AI Agent 系统——工作记忆、情景记忆、语义记忆、程序记忆、方案记忆，带遗忘曲线、记忆强化和死路避免。

## 安装

### Windows 一键安装器（推荐，无需翻墙）

1. 从 [GitHub Releases](../../releases) 下载 `genagent-installer.exe`（约 8 MB）
2. 双击运行，自动完成安装

安装器全程走国内镜像（清华 PyPI + hf-mirror），无需翻墙，约需下载 2-3 GB 依赖。已安装 Python 3.11+ 即可使用。

### pip（开发者）

```bash
pip install git+https://github.com/fenghai0712/genagent-with-humanmemory.git
```

### npm

```bash
npm install -g git+https://github.com/fenghai0712/genagent-with-humanmemory.git
```

npm 安装会自动触发 pip install 安装 Python 依赖。需要 Python 3.11+ 在 PATH 中。

### Windows 一键安装

下载仓库中的 `install.bat`，双击运行（确保 Python 3.11+ 已安装并勾选 "Add Python to PATH"）。

### 环境变量（可选）

安装后可通过环境变量配置，无需改代码：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `HUMAN_MEMORY_DB_PATH` | 数据库文件路径 | `memory.db` |
| `HUMAN_MEMORY_EPISODIC_CAPACITY` | 情景记忆容量上限 | `5000` |
| `HUMAN_MEMORY_CONSOLIDATION_THRESHOLD` | 编码门槛（0-1） | `0.3` |
| `HUMAN_MEMORY_DEPTH_L2_THRESHOLD` | L2 标准编码强度阈值 | `0.3` |
| `HUMAN_MEMORY_DEPTH_L3_THRESHOLD` | L3 深度编码强度阈值 | `0.7` |
| `HUMAN_MEMORY_EMBEDDING_MODEL` | 嵌入模型名称 | `paraphrase-multilingual-MiniLM-L12-v2` |
| `HUMAN_MEMORY_EMBEDDING_DEVICE` | 推理设备 | `cpu` |

**依赖说明**：`sqlite-vec` 提供 Windows 预编译包，`sentence-transformers` 提供中文嵌入模型。首次运行会自动下载模型（约 470MB）。

## 使用

### 方式一：CLI 命令（安装后直接敲命令）

```bash
memory-agent
```

支持命令：`/recall` `/stats` `/learn` `/success` `/fail` `/quit`

也可用模块方式启动：`python -m human_memory.agent`

### 方式二：3 行代码体验

```python
from human_memory import MemoryManager

mm = MemoryManager()
mm.remember("用户在排查 PostgreSQL 慢查询，EXPLAIN 显示 seq scan", explicit_signal=True)
mm.consolidate()

# 用英文搜索中文记忆（跨语言检索）
for m in mm.recall("query optimization"):
    print(m["summary_text"])
```

### 方式三：在自己的 Agent 中集成

```python
from human_memory.agent import MemoryAgent

agent = MemoryAgent()

# 每轮对话走完整 pipeline：感知 → 回忆 → 思考 → 行动 → 学习
response = agent.run("我的数据库查询很慢，怎么办？")

# 记录死路（以后会主动警告）
agent.record_failure(
    problem_type="慢查询",
    problem_desc="PostgreSQL seq scan",
    failed_approach="盲目调大 work_mem",
    why_failed="内存竞争导致整体更慢",
    wasted_minutes=30,
    lessons="先 EXPLAIN ANALYZE，确认瓶颈再调参"
)

agent.end_session()
```

### 方式四：接入 DeepSeek（内置）

```bash
# 设置 API Key（二选一）
# Windows:
set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx

# macOS / Linux:
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

```python
from human_memory.agent import MemoryAgent

# Agent 自动检测 DEEPSEEK_API_KEY，无需传 llm_fn
agent = MemoryAgent()
response = agent.run("帮我查一下昨天那个 bug")
```

也可显式传入：
```python
from human_memory import DeepSeekLLM

agent = MemoryAgent(llm_fn=DeepSeekLLM(api_key="sk-xxx"))
```

### 方式五：接入自定义 LLM

```python
agent = MemoryAgent(llm_fn=your_api_call)
response = agent.run("今天天气怎么样")
```

## 记忆类型

| 类型 | 存储 | 说明 |
|------|------|------|
| 工作记忆 | 内存（7±2 槽位） | 当前会话的注意力焦点 |
| 情景记忆 | SQLite + 向量索引 | 带时间/情绪/重要性标记的经历 |
| 语义记忆 | 概念图（递归 CTE） | 事实知识和概念层级 |
| 程序记忆 | 步骤序列 | 技能、操作流程 |
| 方案记忆 | 独立表 + 向量索引 | 成功方案 + 死路记录，跨语言检索 |

## 核心机制

- **容量竞争遗忘**：活跃记忆超过 5000 条时，强度最低的被淘汰
- **编码深度**：L1 浅层 → L2 标准（strength≥0.3）→ L3 深度（≥0.7）
- **记忆强化**：每次回忆增强 strength，间隔效应（同日重复收益递减）
- **死路排查**：尝试新方向前，先搜索是否匹配已知死路
- **跨语言**：用英文搜中文记忆、用中文搜英文记忆

## 配置

```python
from human_memory import MemoryConfig, MemoryAgent

config = MemoryConfig(
    episodic_capacity=5000,              # 情景记忆上限
    consolidation_score_threshold=0.25,  # 编码门槛（越低记越多）
    depth_l2_threshold=0.3,              # 升级到标准编码的强度阈值
    depth_l3_threshold=0.7,              # 升级到深度编码的强度阈值
)
agent = MemoryAgent(config=config)
```

## 测试

```bash
python demo.py                          # 8 阶段完整演示
python -m pytest tests/ -v              # 25 个测试
```

## 许可

MIT
