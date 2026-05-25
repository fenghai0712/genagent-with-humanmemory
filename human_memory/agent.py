"""Memory-augmented agent with perceive→recall→think→act→learn loop.

Can be used standalone (CLI) or as a base class with an LLM backend.
"""

import os
import json
import time
from typing import Optional, Callable

from human_memory import (
    MemoryManager, MemoryConfig,
    SolutionAttempt, DeadEndRecord,
    AttemptOutcome, FailureType,
)
from human_memory.models import now_iso, new_id


class MemoryAgent:
    """An agent that perceives, recalls, thinks, acts, and learns — backed by
    the human-like memory system at every step."""

    def __init__(self, db_path: Optional[str] = None,
                 config: Optional[MemoryConfig] = None,
                 llm_fn: Optional[Callable] = None):
        if db_path is None:
            db_path = os.environ.get("HUMAN_MEMORY_DB_PATH", "agent_memory.db")
        if config is None:
            config = MemoryConfig(
                db_path=db_path,
                episodic_capacity=5000,
                consolidation_score_threshold=0.25,
            )
        self.memory = MemoryManager(config=config)
        self.llm = llm_fn  # optional LLM backend
        self.session_id = new_id()
        self.session_started_at = now_iso()
        self.turn_count = 0

    # ═══════════════════════════════════════════════
    # Main agent loop
    # ═══════════════════════════════════════════════

    def run(self, user_input: str) -> str:
        """One turn of the agent loop. Returns the agent's response."""
        self.turn_count += 1

        # 1. PERCEIVE — record into working memory
        self._perceive(user_input)

        # 2. RECALL — search all memory types for relevant context
        context = self._gather_context(user_input)

        # 3. THINK — formulate a response (uses LLM if available)
        response = self._think(user_input, context)

        # 4. ACT — execute any implicit actions, record what happened
        self._act(user_input, response)

        # 5. LEARN — extract knowledge, update solutions
        self._learn(user_input, response, context)

        # 6. CONSOLIDATE — periodically flush WM to LTM
        if self.turn_count % 5 == 0:
            self.memory.consolidate()

        return response

    # ═══════════════════════════════════════════════
    # Step 1: Perceive
    # ═══════════════════════════════════════════════

    def _perceive(self, user_input: str):
        """Record the user's input into working memory with metadata."""
        importance = self._assess_importance(user_input)
        emotion = self._detect_emotion(user_input)
        tags = self._extract_tags(user_input)

        self.memory.remember(
            content=f"用户: {user_input}",
            summary=user_input[:200],
            attention_weight=importance,
            emotional_intensity=emotion,
            explicit_signal=importance > 0.7,
            context_tags=tags,
        )

    def _assess_importance(self, text: str) -> float:
        """Heuristic importance scoring. Override with LLM for better results."""
        score = 0.5
        urgent_words = ["紧急", "urgent", "错误", "error", "bug", "崩溃", "crash",
                        "重要", "important", "关键", "critical", "fix", "修复"]
        for w in urgent_words:
            if w.lower() in text.lower():
                score += 0.15
        if "?" in text or "？" in text:
            score += 0.1  # question = higher engagement
        return min(score, 1.0)

    def _detect_emotion(self, text: str) -> float:
        """Heuristic emotion detection."""
        frustration = ["搞不定", "不行", "错了", "又坏了", "frustrated", "stuck",
                       "doesn't work", "broken", "failed", "失败"]
        urgency = ["赶紧", "快", "hurry", "asap", "马上", "立刻"]
        for w in frustration:
            if w.lower() in text.lower():
                return 0.6
        for w in urgency:
            if w.lower() in text.lower():
                return 0.4
        return 0.1

    def _extract_tags(self, text: str) -> list[str]:
        """Simple keyword-based tag extraction."""
        tags = []
        keyword_map = {
            "database": ["数据库", "database", "sql", "postgres", "mysql", "查询", "query"],
            "python": ["python", "async", "await", "asyncio", "fastapi", "django"],
            "debugging": ["bug", "错误", "error", "调试", "debug", "fix", "修复"],
            "performance": ["慢", "slow", "性能", "performance", "卡", "timeout", "超时"],
            "config": ["配置", "config", "设置", "settings", "参数", "parameter"],
        }
        lowered = text.lower()
        for tag, keywords in keyword_map.items():
            if any(k.lower() in lowered for k in keywords):
                tags.append(tag)
        return tags

    # ═══════════════════════════════════════════════
    # Step 2: Recall
    # ═══════════════════════════════════════════════

    def _gather_context(self, user_input: str) -> dict:
        """Gather all relevant context from memory."""
        # Ensure WM is searchable
        self.memory.consolidate()

        return {
            "episodic": self.memory.recall(user_input, limit=5),
            "concepts": self.memory.retrieval.recall_concepts(user_input, limit=5),
            "solutions": self.memory.retrieval.find_solutions(user_input, limit=3),
            "recent": self.memory.recall_recent(limit=5),
        }

    # ═══════════════════════════════════════════════
    # Step 3: Think
    # ═══════════════════════════════════════════════

    def _think(self, user_input: str, context: dict) -> str:
        """Formulate a response. Uses LLM if available, otherwise template-based."""
        if self.llm is not None:
            return self._think_with_llm(user_input, context)
        return self._think_template(user_input, context)

    def _think_with_llm(self, user_input: str, context: dict) -> str:
        """Use an external LLM with memory context injected into the prompt."""
        prompt = self._build_llm_prompt(user_input, context)
        return self.llm(prompt)

    def _build_llm_prompt(self, user_input: str, context: dict) -> str:
        """Build a prompt that injects all relevant memory context."""
        parts = ["你是一个有记忆的 AI 助手。以下是你的相关记忆:\n"]

        if context["episodic"]:
            parts.append("## 相关经历")
            for ep in context["episodic"][:3]:
                parts.append(f"- {ep.get('summary_text', '')[:200]}")
            parts.append("")

        if context["solutions"]:
            parts.append("## 已知解决方案")
            for sol in context["solutions"][:2]:
                parts.append(f"- 问题: {sol.get('problem_type', '')}")
                parts.append(f"  最佳方案: {sol.get('best_approach', '')[:200]}")
                if sol.get("worst_approach"):
                    parts.append(f"  避免: {sol.get('worst_approach', '')[:200]}")
            parts.append("")

        if context["concepts"]:
            parts.append("## 相关知识")
            for c in context["concepts"][:3]:
                parts.append(f"- {c.get('name', '')}: {c.get('description', '')[:100]}")
            parts.append("")

        parts.append(f"## 当前对话\n用户: {user_input}\n\n请根据以上记忆和知识回答用户的问题:")

        return "\n".join(parts)

    def _think_template(self, user_input: str, context: dict) -> str:
        """Template-based reasoning when no LLM is available."""
        lines = []

        # Check if we have relevant solutions
        if context["solutions"]:
            sol = context["solutions"][0]
            lines.append(f"[记忆匹配] 这类似之前解决过的「{sol.get('problem_type', '')}」")
            lines.append(f"  推荐方案: {sol.get('best_approach', '')[:150]}")
            if sol.get("total_wasted_time", 0) > 0:
                lines.append(f"  历史踩坑耗时: {sol['total_wasted_time']} 分钟")
            if sol.get("worst_approach"):
                lines.append(f"  ⚠️ 避免: {sol.get('worst_approach', '')[:150]}")

        # Check recent context
        if context["recent"]:
            lines.append(f"\n[最近记忆] (共 {len(context['recent'])} 条)")
            for ep in context["recent"][:3]:
                lines.append(f"  - {ep.get('summary_text', '')[:100]}")

        # Check concepts
        if context["concepts"]:
            lines.append(f"\n[相关知识]")
            for c in context["concepts"][:3]:
                lines.append(f"  - {c.get('name', '')}: {c.get('description', '')[:80]}")

        if not lines:
            lines.append("[无相关记忆] 这是新类型的问题，我没有找到相关历史。")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════
    # Step 4: Act
    # ═══════════════════════════════════════════════

    def _act(self, user_input: str, response: str):
        """Record the agent's response and any implicit actions."""
        self.memory.remember(
            content=f"助手: {response[:500]}",
            summary=f"回复了关于: {user_input[:100]}",
            attention_weight=0.6,
            context_tags=["agent-response"],
        )

    # ═══════════════════════════════════════════════
    # Step 5: Learn
    # ═══════════════════════════════════════════════

    def _learn(self, user_input: str, response: str, context: dict):
        """Extract patterns and update knowledge from this interaction."""
        # Auto-learn concepts from tags
        tags = self._extract_tags(user_input)
        for tag in tags:
            self.memory.learn_concept(tag, f"话题标签: {tag}")

    def record_success(self, problem_type: str, problem_desc: str,
                       approach: str, why: str):
        """Call this after successfully solving a problem."""
        self.memory.record_solution(
            problem_type=problem_type,
            problem_abstract=problem_desc,
            attempts=[
                SolutionAttempt(
                    approach=approach,
                    outcome=AttemptOutcome.SUCCESS,
                    quality_score=9.0,
                    is_best_known=True,
                    why_succeeded=why,
                ),
            ],
            dead_ends=[],
        )

    def record_failure(self, problem_type: str, problem_desc: str,
                       failed_approach: str, why_failed: str,
                       failure_mode: str = "", wasted_minutes: int = 0,
                       lessons: str = ""):
        """Call this after trying something that didn't work."""
        self.memory.record_solution(
            problem_type=problem_type,
            problem_abstract=problem_desc,
            attempts=[
                SolutionAttempt(
                    approach=failed_approach,
                    outcome=AttemptOutcome.FAILED,
                    quality_score=2.0,
                    is_worst_known=True,
                    why_failed=why_failed,
                ),
            ],
            dead_ends=[
                DeadEndRecord(
                    approach=failed_approach,
                    failure_mode=failure_mode or why_failed,
                    failure_type=FailureType.LOGIC,
                    wasted_time_minutes=wasted_minutes,
                    lessons=lessons or why_failed,
                ),
            ],
        )

    # ═══════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════

    def end_session(self):
        """End the session: final consolidation + maintenance."""
        self.memory.remember(
            content=f"会话结束 (turns={self.turn_count})",
            summary=f"会话 #{self.session_id[:6]} 结束，共 {self.turn_count} 轮",
            attention_weight=0.5,
        )
        self.memory.end_session()

    @property
    def stats(self) -> dict:
        return self.memory.stats()

    def close(self):
        self.memory.close()


# ═══════════════════════════════════════════════════
# CLI interface
# ═══════════════════════════════════════════════════
def cli():
    """Interactive CLI for the memory agent."""
    agent = MemoryAgent()  # db_path from HUMAN_MEMORY_DB_PATH env or default

    print("=" * 50)
    print("  Memory Agent — 有记忆的 AI 助手")
    print("  命令: /stats  /recall <query>  /learn <概念>")
    print("       /success  /fail  /quit")
    print("=" * 50)

    pending_approach = None

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd, _, arg = user_input[1:].partition(" ")
            cmd = cmd.lower()

            if cmd == "quit":
                break
            elif cmd == "stats":
                print("\n--- Memory Stats ---")
                for k, v in agent.stats.items():
                    print(f"  {k}: {v}")
            elif cmd == "recall":
                if not arg:
                    print("用法: /recall <搜索内容>")
                    continue
                print(f"\n--- 搜索记忆: '{arg}' ---")
                for m in agent.memory.recall(arg, limit=5):
                    print(f"  [{m.get('strength', 0):.2f}] {m.get('summary_text', '')[:100]}")
            elif cmd == "learn":
                if not arg:
                    print("用法: /learn <概念名>")
                    continue
                agent.memory.learn_concept(arg, f"用户定义的概念: {arg}")
                print(f"已学习概念: {arg}")
            elif cmd == "success":
                if not arg:
                    arg = "手动标记的成功方案"
                agent.record_success("手动记录", arg, arg, "用户确认成功")
                print("已记录为成功方案")
            elif cmd == "fail":
                agent.record_failure(
                    problem_type="手动记录",
                    problem_desc=arg or "手动标记的失败尝试",
                    failed_approach=arg or "未命名的尝试",
                    why_failed="用户标记为失败",
                    wasted_minutes=10,
                )
                print("已记录为失败方案(死路)")
            else:
                print(f"未知命令: {cmd}")
            continue

        # Normal interaction: run agent loop
        response = agent.run(user_input)
        print(f"\n助手:\n{response}")

    agent.end_session()
    print(f"\n会话结束。统计: {agent.stats}")
    agent.close()


if __name__ == "__main__":
    cli()
