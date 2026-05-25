"""MemoryManager — the central orchestrator for the human-like memory system.

Usage:
    from human_memory import MemoryManager

    mm = MemoryManager()
    mm.remember("刚刚用户说 PostgreSQL 的递归 CTE 可以解决图遍历问题")
    mm.remember("用户选了 SQLite + sqlite-vec 方案", explicit_signal=True)

    results = mm.recall("数据库查询性能问题")
    warnings = mm.check_approach("我准备用多线程来加速")
"""

import threading
import time
from typing import Optional, Callable

from human_memory.config import MemoryConfig, default_config
from human_memory.database import Database
from human_memory.working_memory import WorkingMemory
from human_memory.encoding import Encoder, MaintenanceCycle
from human_memory.retrieval import RetrievalEngine
from human_memory.embedding import EmbeddingProvider, get_embedding_provider
from human_memory.models import (
    WorkingMemorySlot, EpisodicMemory, SolutionMemory,
    SolutionAttempt, DeadEndRecord, Concept, ConceptEdge, Skill,
    EncodingDepth, FailureType, now_iso, new_id,
)


class MemoryManager:
    """Human-like memory system for AI agents.

    Simulates:
    - Working memory (7±2 slots, attention-weighted)
    - Episodic memory (time/place/emotion-tagged experiences)
    - Semantic memory (concept graph)
    - Procedural memory (skills with trigger conditions)
    - Solution memory (problem-solving patterns + dead-end avoidance)
    - Forgetting curve (Ebbinghaus decay)
    - Reinforcement (spacing effect, depth upgrading)
    - Encoding depth (L1 shallow → L2 standard → L3 deep)
    """

    def __init__(self, config: MemoryConfig = default_config,
                 embed_fn: Optional[Callable[[str], list[float]]] = None,
                 embed_provider: Optional[EmbeddingProvider] = None):
        self.config = config
        self._provider = embed_provider or get_embedding_provider(
            config.embedding_model, config.embedding_device)
        # Sync dimension before Database creates vec tables
        self.config.embedding_dim = self._provider.dim
        self.db = Database(config)
        self.wm = WorkingMemory(config)
        self.encoder = Encoder(self.db, config, embed_fn, embed_provider=self._provider)
        self.retrieval = RetrievalEngine(self.db, config, embed_fn or self.encoder.embed,
                                         embed_provider=self._provider)
        self.maintenance = MaintenanceCycle(self.db, self.encoder, config)

        self._maintenance_thread: Optional[threading.Thread] = None
        self._stop_maintenance = threading.Event()

    # --- Working memory interface ---

    def remember(self, content: str, summary: str = "",
                 attention_weight: float = 0.5, emotional_intensity: float = 0.0,
                 explicit_signal: bool = False, context_tags: Optional[list[str]] = None,
                 novelty_score: float = 0.5) -> WorkingMemorySlot:
        """Record an experience into working memory."""
        return self.wm.enter(
            content=content, summary=summary,
            attention_weight=attention_weight,
            emotional_intensity=emotional_intensity,
            explicit_signal=explicit_signal,
            context_tags=context_tags,
            novelty_score=novelty_score,
        )

    def get_current_thoughts(self) -> list[WorkingMemorySlot]:
        """Get current working memory contents."""
        return self.wm.get_attention_sorted()

    # --- Consolidation ---

    def consolidate(self) -> tuple[list[EpisodicMemory], list[WorkingMemorySlot]]:
        """Consolidate working memory into long-term memory. Call at session end or periodically."""
        slots = self.wm.evict_for_consolidation()
        return self.encoder.consolidate(slots)

    def end_session(self):
        """End a session: consolidate WM, run maintenance cycle."""
        self.consolidate()
        self.run_maintenance()

    # --- Retrieval ---

    def recall(self, query: str, limit: int = 10) -> list[dict]:
        """Recall episodic memories similar to the query."""
        return self.retrieval.recall_similar(query, limit)

    def recall_by_tags(self, tags: list[str], limit: int = 10) -> list[dict]:
        return self.retrieval.recall_by_tags(tags, limit)

    def recall_by_time(self, start_iso: str, end_iso: str, limit: int = 50) -> list[dict]:
        return self.retrieval.recall_by_time(start_iso, end_iso, limit)

    def recall_recent(self, limit: int = 20) -> list[dict]:
        return self.retrieval.recall_recent(limit)

    def recall_all(self, query: str, limit_per_type: int = 5) -> dict:
        """Cross-type recall: search all memory types at once."""
        return self.retrieval.recall_all(query, limit_per_type)

    # --- Reinforcement ---

    def reinforce(self, memory_id: str, context: str = "recall"):
        """Reinforce a memory when it's recalled or referenced."""
        kwargs = {"is_recall": True}
        if context == "association":
            kwargs["is_association"] = True
        elif context == "emotional":
            kwargs["emotional_context"] = True
        elif context == "explicit":
            kwargs["is_explicit"] = True
        self.encoder.reinforce(memory_id, **kwargs)
        self.encoder.check_depth_upgrade(memory_id)

    # --- Solution memory ---

    def record_solution(self, problem_type: str, problem_abstract: str,
                        attempts: list[SolutionAttempt],
                        dead_ends: Optional[list[DeadEndRecord]] = None,
                        context_tags: Optional[list[str]] = None) -> SolutionMemory:
        """Record a problem-solving experience including both successes and dead ends."""
        return self.encoder.encode_solution(
            problem_type=problem_type,
            problem_abstract=problem_abstract,
            attempts=attempts,
            dead_ends=dead_ends or [],
            context_tags=context_tags,
        )

    def solve_problem(self, problem_description: str,
                      proposed_approach: Optional[str] = None) -> dict:
        """Full solution pipeline: check dead ends first, then find solutions."""
        return self.retrieval.solve(problem_description, proposed_approach)

    def check_approach(self, approach: str) -> list[dict]:
        """Check if a proposed approach matches known dead ends."""
        return self.retrieval.check_dead_ends(approach)

    # --- Semantic memory ---

    def learn_concept(self, name: str, description: str = "",
                      parent_name: Optional[str] = None,
                      embedding: Optional[list[float]] = None) -> Concept:
        """Add or update a concept in semantic memory."""
        vec = embedding or self.encoder.embed(name + ": " + description)
        parent_id = None
        if parent_name:
            row = self.db.conn.execute(
                "SELECT id FROM concepts WHERE name = ?", (parent_name,)
            ).fetchone()
            if row:
                parent_id = row["id"]
        c = Concept(
            name=name, description=description, parent_id=parent_id,
            embedding=vec,
        )
        self.db.upsert_concept({
            "id": c.id, "name": c.name, "embedding": c.embedding,
            "parent_id": c.parent_id, "description": c.description,
            "strength": c.strength, "created_at": c.created_at,
        })
        return c

    def relate_concepts(self, source_name: str, target_name: str,
                        relation: str = "related_to", strength: float = 0.5):
        """Create an edge between two concepts."""
        s = self.db.conn.execute("SELECT id FROM concepts WHERE name=?", (source_name,)).fetchone()
        t = self.db.conn.execute("SELECT id FROM concepts WHERE name=?", (target_name,)).fetchone()
        if s and t:
            edge = ConceptEdge(
                source_id=s["id"], target_id=t["id"],
                relation_type=relation, strength=strength,
            )
            self.db.insert_edge({
                "id": edge.id, "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation_type": edge.relation_type, "strength": edge.strength,
            })

    def explore_concept(self, concept_name: str, depth: int = 2) -> dict:
        """Explore a concept and its neighborhood in the semantic graph."""
        row = self.db.conn.execute(
            "SELECT id FROM concepts WHERE name=?", (concept_name,)
        ).fetchone()
        if not row:
            return {"hierarchy": [], "edges": [], "connected_concepts": []}
        return self.retrieval.get_concept_graph(row["id"], depth)

    # --- Procedural memory ---

    def learn_skill(self, name: str, description: str, steps: list[str],
                    trigger_condition: str = "", context_tags: Optional[list[str]] = None
                    ) -> Skill:
        """Record a procedural skill."""
        import json
        steps_json = json.dumps([
            {"order": i, "action": step} for i, step in enumerate(steps)
        ], ensure_ascii=False)
        skill = Skill(
            name=name, description=description,
            trigger_condition=trigger_condition,
            steps_json=steps_json,
            context_tags=json.dumps(context_tags or [], ensure_ascii=False),
        )
        self.db.upsert_skill({
            "id": skill.id, "name": skill.name, "description": skill.description,
            "trigger_condition": skill.trigger_condition, "steps_json": skill.steps_json,
            "proficiency": skill.proficiency, "context_tags": skill.context_tags,
            "created_at": skill.created_at, "last_used_at": skill.last_used_at,
            "use_count": skill.use_count,
        })
        return skill

    def use_skill(self, skill_id: str):
        """Mark a skill as used, increasing proficiency."""
        skill = self.db.get_skill(skill_id)
        if skill:
            new_prof = min(skill["proficiency"] + 0.05, 1.0)
            new_count = skill["use_count"] + 1
            self.db.conn.execute(
                "UPDATE skills SET proficiency=?, use_count=?, last_used_at=? WHERE id=?",
                (new_prof, new_count, now_iso(), skill_id),
            )
            self.db.conn.commit()

    # --- Maintenance ---

    def run_maintenance(self):
        """Run one maintenance cycle: forgetting, depth upgrades, archiving."""
        self.maintenance.run()

    def start_background_maintenance(self, interval_seconds: Optional[float] = None):
        """Start background maintenance thread."""
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            return
        self._stop_maintenance.clear()
        interval = interval_seconds or self.config.maintenance_interval_seconds

        def _loop():
            while not self._stop_maintenance.wait(interval):
                try:
                    self.maintenance.run()
                except Exception:
                    pass

        self._maintenance_thread = threading.Thread(target=_loop, daemon=True)
        self._maintenance_thread.start()

    def stop_background_maintenance(self):
        """Stop background maintenance thread."""
        self._stop_maintenance.set()
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5.0)

    # --- Stats ---

    def stats(self) -> dict:
        """Return memory system statistics."""
        ep_total = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memories"
        ).fetchone()["c"]
        ep_active = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memories WHERE is_forgotten = 0"
        ).fetchone()["c"]
        ep_l1 = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memories WHERE encoding_depth = 1"
        ).fetchone()["c"]
        ep_l2 = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memories WHERE encoding_depth = 2"
        ).fetchone()["c"]
        ep_l3 = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memories WHERE encoding_depth = 3"
        ).fetchone()["c"]
        concepts = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM concepts"
        ).fetchone()["c"]
        solutions = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM solution_memories WHERE status = 'active'"
        ).fetchone()["c"]
        total_dead_ends = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM dead_end_embeddings"
        ).fetchone()["c"]
        skills_count = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM skills"
        ).fetchone()["c"]

        return {
            "working_memory_slots": len(self.wm),
            "episodic_total": ep_total,
            "episodic_active": ep_active,
            "episodic_forgotten": ep_total - ep_active,
            "encoding_depth": {"L1_shallow": ep_l1, "L2_standard": ep_l2, "L3_deep": ep_l3},
            "concepts": concepts,
            "active_solutions": solutions,
            "dead_ends_recorded": total_dead_ends,
            "skills": skills_count,
        }

    def close(self):
        """Clean shutdown."""
        self.stop_background_maintenance()
        self.db.close()
