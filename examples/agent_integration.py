"""Example: integrating human-memory into an AI agent loop."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human_memory import (
    MemoryManager, MemoryConfig,
    SolutionAttempt, DeadEndRecord,
    AttemptOutcome, FailureType,
)


class AgentWithMemory:
    """Minimal agent loop that uses the human-like memory system."""

    def __init__(self, db_path: str = "agent_memory.db"):
        self.memory = MemoryManager(config=MemoryConfig(
            db_path=db_path,
            episodic_capacity=5000,
            eviction_batch_size=50,
            consolidation_score_threshold=0.25,
        ))
        self.session_active = False

    # ── session lifecycle ──────────────────────────

    def start_session(self):
        self.session_active = True

    def end_session(self):
        """End session: consolidate WM → LTM, run maintenance."""
        self.memory.end_session()
        self.session_active = False

    # ── perception + memory encoding ───────────────

    def perceive(self, content: str, *,
                 is_important: bool = False,
                 is_error: bool = False,
                 emotion: str | None = None,
                 tags: list[str] | None = None):
        """Record a perception into working memory.

        is_important: user explicitly marked this as important
        is_error:     error / failure event (higher emotional weight)
        emotion:      optional emotion label
        tags:         context tags for later retrieval
        """
        attention = 0.8 if is_important else 0.5
        emotional = 0.7 if is_error else (0.3 if emotion else 0.0)

        self.memory.remember(
            content=content,
            attention_weight=attention,
            emotional_intensity=emotional,
            explicit_signal=is_important,
            context_tags=tags,
        )

    # ── retrieval ──────────────────────────────────

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic recall of past experiences relevant to query.
        Consolidates WM first so recent perceptions are searchable."""
        self.memory.consolidate()
        return self.memory.recall(query, limit=limit)

    def recall_recent(self, limit: int = 10) -> list[dict]:
        """What happened recently?"""
        self.memory.consolidate()
        return self.memory.recall_recent(limit=limit)

    # ── problem solving ────────────────────────────

    def before_trying_approach(self, approach_description: str) -> list[dict]:
        """Check: has this approach failed before? Call BEFORE trying something."""
        self.memory.consolidate()  # flush WM so new dead ends are searchable
        return self.memory.check_approach(approach_description)

    def after_problem_solved(self, *,
                              problem_type: str,
                              problem_description: str,
                              success_approach: str,
                              success_why: str,
                              failed_approaches: list[dict] | None = None,
                              dead_ends: list[dict] | None = None,
                              tags: list[str] | None = None):
        """Record a completed problem-solving experience."""
        attempts = [
            SolutionAttempt(
                approach=success_approach,
                outcome=AttemptOutcome.SUCCESS,
                quality_score=9.0,
                is_best_known=True,
                why_succeeded=success_why,
            ),
        ]
        for fa in (failed_approaches or []):
            attempts.append(SolutionAttempt(
                approach=fa["approach"],
                outcome=AttemptOutcome.FAILED,
                quality_score=fa.get("score", 2.0),
                why_failed=fa.get("why", ""),
            ))

        de_records = []
        for de in (dead_ends or []):
            de_records.append(DeadEndRecord(
                approach=de["approach"],
                failure_mode=de["failure_mode"],
                failure_type=de.get("failure_type", FailureType.LOGIC),
                wasted_time_minutes=de.get("wasted_minutes", 0),
                lessons=de.get("lessons", ""),
            ))

        self.memory.record_solution(
            problem_type=problem_type,
            problem_abstract=problem_description,
            attempts=attempts,
            dead_ends=de_records,
            context_tags=tags,
        )

    # ── knowledge ──────────────────────────────────

    def learn_fact(self, concept: str, description: str = "",
                   parent: str | None = None):
        """Learn a new concept in semantic memory."""
        self.memory.learn_concept(concept, description, parent_name=parent)

    def learn_skill(self, name: str, description: str, steps: list[str],
                    trigger: str = ""):
        """Record a procedural skill."""
        return self.memory.learn_skill(name, description, steps, trigger)

    # ── introspection ──────────────────────────────

    @property
    def stats(self) -> dict:
        return self.memory.stats()

    def close(self):
        self.memory.close()


# ═══════════════════════════════════════════════════
# Demo: a realistic agent session
#
# NOTE: The default model (all-MiniLM-L6-v2) is English-only.
# For Chinese support, switch to:
#   MemoryConfig(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    DB = "demo_agent.db"
    if os.path.exists(DB):
        os.remove(DB)

    agent = AgentWithMemory(db_path=DB)
    agent.start_session()

    # User: "why is this database query slow?"
    agent.perceive("user is investigating a slow PostgreSQL query", is_important=True,
                   tags=["database", "postgresql", "performance"])
    agent.perceive("query uses SELECT * with 3 JOINs, explain shows seq scan",
                   tags=["database", "query-plan"])
    agent.perceive("index was created but not used because WHERE wraps column in a function",
                   tags=["database", "indexing"])
    agent.perceive("mistake: tried FORCE INDEX which slowed down another query", is_error=True,
                   tags=["database", "mistake"])

    # Agent thinks: should I suggest FORCE INDEX?
    warnings = agent.before_trying_approach("use FORCE INDEX to force a specific index")
    if warnings:
        print("Dead-end warning:", warnings[0]["lessons"])
    else:
        print("No dead-end warnings for FORCE INDEX approach")

    # Agent recalls similar past situations
    print("\n--- Recall: 'query optimization' ---")
    for m in agent.recall("query optimization"):
        print(f"  [{m.get('strength', 0):.2f}] {m.get('summary_text', '')[:80]}")

    # Problem solved
    agent.after_problem_solved(
        problem_type="PostgreSQL function-wrapped column breaks index usage",
        problem_description="WHERE date(created_at) = '2026-01-01' prevented index usage on created_at",
        success_approach="Rewrote to WHERE created_at >= '2026-01-01' AND created_at < '2026-01-02' — range scan hits index",
        success_why="removed function wrap on column, index is now usable",
        failed_approaches=[
            {"approach": "FORCE INDEX to override optimizer choice", "score": 2.0,
             "why": "optimizer plans for other queries degraded after forcing this index"},
        ],
        dead_ends=[
            {"approach": "blindly increase work_mem expecting memory to fix everything",
             "failure_mode": "excessive work_mem caused parallel workers to compete for memory, overall slower",
             "failure_type": FailureType.CONFIG,
             "wasted_minutes": 30,
             "lessons": "EXPLAIN ANALYZE first, identify actual bottleneck, then tune"},
        ],
        tags=["postgresql", "indexing", "sargable"],
    )

    # Later: check if the dead-end is recognized
    print("\n--- Checking approach: 'crank up work_mem' ---")
    warnings2 = agent.before_trying_approach("crank up work_mem to maximum")
    if warnings2:
        for w in warnings2:
            print(f"  DEAD-END HIT: {w['failure_mode']}")
            print(f"  Wasted time in past: {w['wasted_time_minutes']} min")
            print(f"  Lesson: {w['lessons']}")

    # Learn from the experience
    agent.learn_fact("Sargable Query", "Search ARGument ABLE — index-friendly query conditions",
                     parent="Database Indexing")
    agent.learn_skill(
        name="SQL Performance Tuning",
        description="Standard slow-query debugging workflow",
        steps=[
            "EXPLAIN ANALYZE to get actual execution plan",
            "Check for unexpected seq scans",
            "Verify index usage and filter selectivity",
            "Rewrite query to remove function-wrapped columns / implicit casts",
            "Benchmark to confirm improvement",
        ],
        trigger="query execution time exceeds threshold",
    )

    # End session
    agent.end_session()

    print("\n--- System Stats ---")
    for k, v in agent.stats.items():
        print(f"  {k}: {v}")

    agent.close()
    os.remove(DB)
    print("\nDone.")
