# genagent-with-humanmemory

模拟人类记忆的 AI Agent 系统——工作记忆、情景记忆、语义记忆、程序记忆、方案记忆，带遗忘曲线、记忆强化和死路避免。

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/fenghai0712/genagent-with-humanmemory.git
cd genagent-with-humanmemory

# 2. 安装依赖
pip install -r requirements.txt

# 3. （可选）以开发模式安装，方便 import
pip install -e .
```

首次运行会自动下载多语言嵌入模型 `paraphrase-multilingual-MiniLM-L12-v2`（约 470MB），支持中英文跨语言检索。

## 使用

### 方式一：交互式 Agent（命令行对话）

```bash
python -m human_memory.agent
```

支持命令：

| 命令 | 说明 |
|------|------|
| `/recall <关键词>` | 搜索记忆 |
| `/stats` | 查看记忆统计 |
| `/learn <概念>` | 手动学习概念 |
| `/success` | 标记成功方案 |
| `/fail` | 标记失败方案（死路） |
| `/quit` | 退出 |

### 方式二：3 行代码体验

```python
from human_memory import MemoryManager

mm = MemoryManager()

# 记录事件
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
print(response)

# 解决问题后记录方案
agent.record_success(
    problem_type="慢查询",
    problem_desc="PostgreSQL seq scan",
    approach="添加复合索引覆盖 WHERE + ORDER BY",
    why="索引覆盖了查询的所有过滤和排序字段"
)

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

### 方式四：接入你的 LLM

```python
def my_llm(prompt: str) -> str:
    # prompt 已包含所有记忆上下文
    return your_api_call(prompt)

agent = MemoryAgent(llm_fn=my_llm)
response = agent.run("帮我查一下昨天的那个 bug")
```

### 运行演示和测试

```bash
python demo.py                          # 8 阶段完整演示
python examples/agent_integration.py    # Agent 集成示例
python -m pytest tests/ -v              # 25 个测试
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

- **容量竞争遗忘**：活跃记忆超过 5000 条时，强度最低的被淘汰（不是按天数）
- **编码深度**：L1 浅层 → L2 标准（strength≥0.3）→ L3 深度（≥0.7）
- **记忆强化**：每次回忆增强 strength，间隔效应（同日重复收益递减）
- **死路排查**：尝试新方向前，先搜索是否匹配已知死路
- **跨语言**：用英文搜中文记忆、用中文搜英文记忆

## 配置

```python
from human_memory import MemoryConfig

config = MemoryConfig(
    episodic_capacity=5000,           # 情景记忆上限
    consolidation_score_threshold=0.25,  # 编码门槛（越低记越多）
    depth_l2_threshold=0.3,           # 升级到标准编码的强度阈值
    depth_l3_threshold=0.7,           # 升级到深度编码的强度阈值
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
)

agent = MemoryAgent(config=config)
```

## 项目结构

```
human_memory/
├── agent.py             # Agent 循环（感知→回忆→思考→行动→学习）
├── memory_manager.py    # 中央协调器
├── working_memory.py    # 工作记忆
├── encoding.py          # 编码 + 巩固 + 强化 + 维护
├── retrieval.py         # 检索 + 死路排查
├── database.py          # SQLite + sqlite-vec
├── embedding.py         # 多语言嵌入模型
├── config.py            # 全部可调参数
└── models/__init__.py   # 数据模型
tests/
examples/
```

## 许可

MIT
