import json
import math
from typing import Optional, Callable

from human_memory.config import MemoryConfig, default_config
from human_memory.database import Database, _vec_to_blob
from human_memory.models import SolutionAttempt, AttemptOutcome
from human_memory.embedding import EmbeddingProvider, get_embedding_provider


class RetrievalEngine:
    """Memory retrieval with decay-aware search, solution lookup, and dead-end avoidance."""

    def __init__(self, db: Database, config: MemoryConfig = default_config,
                 embed_fn: Optional[Callable[[str], list[float]]] = None,
                 embed_provider: Optional[EmbeddingProvider] = None):
        self.db = db
        self.config = config
        self._provider = embed_provider or get_embedding_provider(
            config.embedding_model, config.embedding_device)
        self.config.embedding_dim = self._provider.dim
        self.embed = embed_fn or self._provider.embed

    # --- Episodic retrieval ---

    def recall_similar(self, query: str, limit: int = 10,
                       include_forgotten: bool = False) -> list[dict]:
        """Vector-based recall of similar episodic memories."""
        query_vec = self.embed(query)
        results = self.db.search_similar_episodes(query_vec, limit, include_forgotten)

        # Apply retention filtering (strength-based)
        filtered = []
        for r in results:
            r["retention"] = self._calculate_retention(r)
            if include_forgotten or r["retention"] >= self.config.forgotten_retrieval_threshold:
                filtered.append(r)

        # Sort by combined score: closer vector distance + higher strength
        filtered.sort(key=lambda r: r.get("distance", 1.0) * (1.0 - r.get("strength", 0.5)))
        return filtered

    def recall_by_tags(self, tags: list[str], limit: int = 10) -> list[dict]:
        """Recall memories matching context tags."""
        placeholders = ", ".join(f"'%{t}%'" for t in tags)
        rows = self.db.conn.execute(f"""
            SELECT * FROM episodic_memories
            WHERE is_forgotten = 0
              AND (context_tags LIKE {placeholders[0] if len(tags) == 1 else placeholders})
            ORDER BY strength DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def recall_by_time(self, start_iso: str, end_iso: str, limit: int = 50) -> list[dict]:
        rows = self.db.conn.execute("""
            SELECT * FROM episodic_memories
            WHERE is_forgotten = 0
              AND created_at BETWEEN ? AND ?
            ORDER BY created_at DESC LIMIT ?
        """, (start_iso, end_iso, limit)).fetchall()
        return [dict(r) for r in rows]

    def recall_recent(self, limit: int = 20) -> list[dict]:
        rows = self.db.conn.execute("""
            SELECT * FROM episodic_memories
            WHERE is_forgotten = 0
            ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def _calculate_retention(self, ep: dict) -> float:
        """Retention = strength. Capacity-based competition replaces time-decay."""
        return ep.get("strength", 0.2)

    # --- Solution retrieval (two-phase: dead-end check first, then solution lookup) ---

    def check_dead_ends(self, proposed_approach: str) -> list[dict]:
        """Phase 0: Check if a proposed approach matches known dead ends."""
        approach_vec = self.embed(proposed_approach)
        query_blob = _vec_to_blob(approach_vec)

        rows = self.db.conn.execute("""
            SELECT d.id, d.failure_type, d.parent_solution_id,
                   vec_distance_L2(?, d.approach_vector) as distance
            FROM dead_end_embeddings d
            ORDER BY distance ASC LIMIT 5
        """, (query_blob,)).fetchall()

        warnings = []
        for r in rows:
            distance = r["distance"] if isinstance(r["distance"], (int, float)) else 1.0
            if distance < self.config.dead_end_similarity_threshold:
                sol = self.db.conn.execute(
                    "SELECT dead_ends_json, problem_type FROM solution_memories WHERE id=?",
                    (r["parent_solution_id"],),
                ).fetchone()
                if sol:
                    try:
                        de_list = json.loads(sol["dead_ends_json"] or "[]")
                    except json.JSONDecodeError:
                        de_list = []
                    for de in de_list:
                        if de.get("dead_end_id") == r["id"]:
                            warnings.append({
                                "dead_end_id": r["id"],
                                "problem_type": sol["problem_type"],
                                "approach": de.get("approach", ""),
                                "failure_mode": de.get("failure_mode", ""),
                                "failure_type": r["failure_type"],
                                "wasted_time_minutes": de.get("wasted_time_minutes", 0),
                                "lessons": de.get("lessons", ""),
                                "distance": distance,
                            })
        return warnings

    def find_solutions(self, problem_description: str, limit: int = 5) -> list[dict]:
        """Phase 1: Find relevant solution memories."""
        query_vec = self.embed(problem_description)
        results = self.db.search_similar_problems(query_vec, limit)

        enriched = []
        for r in results:
            try:
                attempts = json.loads(r.get("attempts_json", "[]"))
            except json.JSONDecodeError:
                attempts = []
            try:
                dead_ends = json.loads(r.get("dead_ends_json", "[]"))
            except json.JSONDecodeError:
                dead_ends = []
            try:
                failure_patterns = json.loads(r.get("failure_patterns_json", "[]"))
            except json.JSONDecodeError:
                failure_patterns = []

            # Sort attempts by quality
            attempts.sort(key=lambda a: a.get("quality_score", 0), reverse=True)
            best_attempts = [a for a in attempts if a.get("outcome") == AttemptOutcome.SUCCESS]
            worst_attempts = [a for a in attempts if a.get("outcome") == AttemptOutcome.FAILED]

            r["parsed_attempts"] = attempts
            r["parsed_dead_ends"] = dead_ends
            r["parsed_failure_patterns"] = failure_patterns
            r["best_attempts"] = best_attempts[:3]
            r["worst_attempts"] = worst_attempts[:3]
            r["total_wasted_time"] = sum(
                de.get("wasted_time_minutes", 0) for de in dead_ends)

            enriched.append(r)

        enriched.sort(key=lambda r: (
            r.get("strength", 0.5) * 0.6 + (1.0 - r.get("distance", 1.0)) * 0.4
        ), reverse=True)

        return enriched

    def solve(self, problem_description: str, proposed_approach: str | None = None
              ) -> dict:
        """Full problem-solving pipeline: check dead ends, then find best solutions.

        Returns a dict with 'warnings', 'solutions', and 'recommendation'.
        """
        result = {"warnings": [], "solutions": [], "recommendation": ""}

        # Phase 0: Dead-end check
        if proposed_approach:
            result["warnings"] = self.check_dead_ends(proposed_approach)

        # Phase 1: Find solutions
        result["solutions"] = self.find_solutions(problem_description)

        # Build recommendation
        if result["warnings"]:
            total_wasted = sum(w.get("wasted_time_minutes", 0) for w in result["warnings"])
            result["recommendation"] = (
                f"警告: 你当前想尝试的方向与 {len(result['warnings'])} 条已知死路高度相似，"
                f"历史总浪费 {total_wasted} 分钟。建议先查看下面的成功方案。"
            )
        elif result["solutions"]:
            best = result["solutions"][0]
            result["recommendation"] = (
                f"推荐方案 (来自「{best.get('problem_type', '')}」): "
                f"{best.get('best_approach', '无记录')}"
            )
        else:
            result["recommendation"] = "没有找到相关历史方案，这是新类型的问题。"

        return result

    # --- Semantic retrieval ---

    def recall_concepts(self, query: str, limit: int = 10) -> list[dict]:
        """Vector search for concepts."""
        query_vec = self.embed(query)
        try:
            rows = self.db.conn.execute("""
                SELECT c.*, vec_distance_L2(?, c.embedding) as distance
                FROM concepts c
                WHERE c.embedding IS NOT NULL
                ORDER BY distance ASC LIMIT ?
            """, (_vec_to_blob(query_vec), limit)).fetchall()
        except Exception:
            # Fallback to name search
            rows = self.db.conn.execute(
                "SELECT * FROM concepts WHERE name LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_concept_graph(self, concept_id: str, depth: int = 2) -> dict:
        """Get a concept and its neighborhood in the semantic graph."""
        hierarchy = self.db.get_concept_hierarchy(concept_id, depth)
        edges = self.db.conn.execute("""
            SELECT * FROM concept_edges
            WHERE source_id = ? OR target_id = ?
        """, (concept_id, concept_id)).fetchall()

        # Get connected concepts
        connected_ids = set()
        for e in edges:
            connected_ids.add(e["source_id"])
            connected_ids.add(e["target_id"])
        connected_ids.discard(concept_id)

        connected = []
        if connected_ids:
            placeholders = ", ".join("?" for _ in connected_ids)
            rows = self.db.conn.execute(
                f"SELECT * FROM concepts WHERE id IN ({placeholders})",
                list(connected_ids),
            ).fetchall()
            connected = [dict(r) for r in rows]

        return {
            "hierarchy": hierarchy,
            "edges": [dict(e) for e in edges],
            "connected_concepts": connected,
        }

    # --- Cross-type recall ---

    def recall_all(self, query: str, limit_per_type: int = 5) -> dict:
        """Cross-type recall: search all memory types with one query."""
        query_vec = self.embed(query)

        return {
            "episodic": self.recall_similar(query, limit=limit_per_type),
            "concepts": self.recall_concepts(query, limit=limit_per_type),
            "solutions": self.find_solutions(query, limit=limit_per_type),
        }
