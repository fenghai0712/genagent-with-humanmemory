import json
import math
import time
from typing import Optional, Callable

from human_memory.config import MemoryConfig, default_config
from human_memory.models import (
    EpisodicMemory, WorkingMemorySlot, SolutionMemory, SolutionAttempt,
    DeadEndRecord, EncodingDepth, AttemptOutcome, FailureType,
    now_iso, new_id,
)
from human_memory.database import Database, _vec_to_blob, _blob_to_vec
from human_memory.embedding import EmbeddingProvider, get_embedding_provider


class ConsolidationScorer:
    """Scores working memory slots for LTM encoding eligibility."""

    def __init__(self, config: MemoryConfig = default_config):
        self.config = config

    def score(self, slot: WorkingMemorySlot) -> float:
        return (
            slot.attention_weight * self.config.attention_weight +
            slot.emotional_intensity * self.config.emotion_weight +
            slot.novelty_score * self.config.novelty_weight +
            (1.0 if slot.explicit_signal else 0.0) * self.config.explicit_signal_weight
        )

    def should_encode(self, slot: WorkingMemorySlot) -> bool:
        return self.score(slot) >= self.config.consolidation_score_threshold


class Encoder:
    """Handles WM→LTM encoding, reinforcement, and depth upgrading."""

    def __init__(self, db: Database, config: MemoryConfig = default_config,
                 embed_fn: Optional[Callable[[str], list[float]]] = None,
                 embed_provider: Optional[EmbeddingProvider] = None):
        self.db = db
        self.config = config
        self._provider = embed_provider or get_embedding_provider(
            config.embedding_model, config.embedding_device)
        # Sync config dimension with actual model dimension
        self.config.embedding_dim = self._provider.dim
        self.embed = embed_fn or self._provider.embed
        self.scorer = ConsolidationScorer(config)

    def consolidate(self, slots: list[WorkingMemorySlot]) -> tuple[list[EpisodicMemory], list[WorkingMemorySlot]]:
        """Consolidate WM slots into episodic memories. Returns (encoded, discarded)."""
        encoded = []
        discarded = []
        for slot in slots:
            if self.scorer.should_encode(slot):
                ep = self._encode_slot(slot)
                encoded.append(ep)
            else:
                discarded.append(slot)
                self._record_discard(slot)
        return encoded, discarded

    def _encode_slot(self, slot: WorkingMemorySlot) -> EpisodicMemory:
        score = self.scorer.score(slot)
        depth = self._initial_depth(score)
        embedding = self.embed(slot.summary or slot.content)

        ep = EpisodicMemory(
            encoding_depth=depth,
            strength=score * 0.5,
            summary_text=slot.summary or slot.content[:500],
            summary_vec=embedding,
            importance_initial=score,
            context_tags=json.dumps(slot.context_tags) if slot.context_tags else None,
        )

        ep_data = {
            "id": ep.id, "encoding_depth": ep.encoding_depth,
            "strength": ep.strength, "summary_text": ep.summary_text,
            "summary_vec": ep.summary_vec, "importance_initial": ep.importance_initial,
            "created_at": ep.created_at,
            "entities_json": ep.entities_json, "emotion_tags": ep.emotion_tags,
            "context_summary": ep.context_summary, "related_memory_ids": ep.related_memory_ids,
            "context_tags": ep.context_tags,
            "full_context_json": ep.full_context_json,
            "narrative_thread": ep.narrative_thread,
            "derived_lessons": ep.derived_lessons,
            "sensory_snapshot": ep.sensory_snapshot,
            "last_recalled_at": ep.last_recalled_at,
            "recall_count": ep.recall_count, "retention": ep.retention,
            "is_forgotten": ep.is_forgotten, "forgotten_at": ep.forgotten_at,
        }
        self.db.insert_episodic(ep_data)
        return ep

    def _initial_depth(self, score: float) -> int:
        if score >= 0.7:
            return EncodingDepth.L2_STANDARD
        return EncodingDepth.L1_SHALLOW

    def _record_discard(self, slot: WorkingMemorySlot):
        score = self.scorer.score(slot)
        reason = "below_threshold"
        if slot.attention_weight < 0.1:
            reason = "low_attention"
        elif slot.explicit_signal:
            reason = "explicit_but_low_score"
        self.db.conn.execute(
            "INSERT INTO discarded_episodes (id, content_summary, consolidation_score, reason, discarded_at) VALUES (?,?,?,?,?)",
            (new_id(), slot.summary or slot.content[:200], score, reason, now_iso()),
        )
        self.db.conn.commit()

    # --- Reinforcement ---

    def reinforce(self, memory_id: str, is_recall: bool = True,
                  is_association: bool = False, emotional_context: bool = False,
                  is_explicit: bool = False, is_rapid_repeat: bool = False):
        """Apply reinforcement increment to a memory."""
        cfg = self.config
        delta = 0.0
        if is_recall:
            delta += cfg.base_recall_increment
        if is_association:
            delta += cfg.association_increment
        if emotional_context:
            delta += cfg.emotional_context_increment
        if is_explicit:
            delta += cfg.explicit_mark_increment
        if is_rapid_repeat:
            delta += cfg.rapid_recall_increment

        ep = self.db.get_episodic(memory_id)
        if ep and ep.get("last_recalled_at"):
            try:
                ts = time.mktime(time.strptime(ep["last_recalled_at"], "%Y-%m-%dT%H:%M:%S"))
                days_since = (time.time() - ts) / 86400.0
                spacing_factor = 1.0 - math.exp(-days_since / cfg.spacing_tau_days)
                delta *= spacing_factor
            except Exception:
                pass

        self.db.reinforce(memory_id, delta)

    def check_depth_upgrade(self, memory_id: str) -> bool:
        """Check if a memory should be upgraded in encoding depth."""
        ep = self.db.get_episodic(memory_id)
        if not ep:
            return False
        strength = ep["strength"]
        current_depth = ep["encoding_depth"]
        if strength >= self.config.depth_l3_threshold and current_depth < EncodingDepth.L3_DEEP:
            return True
        if strength >= self.config.depth_l2_threshold and current_depth < EncodingDepth.L2_STANDARD:
            return True
        return False

    def mark_pending_upgrade(self, memory_id: str):
        """Mark a memory as needing depth upgrade during maintenance cycle."""
        self.db.conn.execute(
            "UPDATE episodic_memories SET encoding_depth = encoding_depth WHERE id = ?",
            (memory_id,),
        )
        self.db.conn.commit()

    def perform_depth_upgrade(self, memory_id: str):
        """Actually perform the depth upgrade. Called during maintenance."""
        ep = self.db.get_episodic(memory_id)
        if not ep:
            return
        current_depth = ep["encoding_depth"]
        strength = ep["strength"]
        new_depth = current_depth

        if strength >= self.config.depth_l3_threshold and current_depth < EncodingDepth.L3_DEEP:
            new_depth = EncodingDepth.L3_DEEP
            self._fill_l3_fields(ep)
        elif strength >= self.config.depth_l2_threshold and current_depth < EncodingDepth.L2_STANDARD:
            new_depth = EncodingDepth.L2_STANDARD
            self._fill_l2_fields(ep)

        if new_depth != current_depth:
            self.db.conn.execute(
                "UPDATE episodic_memories SET encoding_depth = ? WHERE id = ?",
                (new_depth, memory_id),
            )
            self.db.conn.commit()

    def _fill_l2_fields(self, ep: dict):
        """Populate L2 fields. In production, call an LLM to extract entities, emotions, context."""
        updates = {}
        if not ep.get("emotion_tags"):
            updates["emotion_tags"] = "[]"
        if not ep.get("related_memory_ids"):
            updates["related_memory_ids"] = "[]"
        if not ep.get("context_summary"):
            updates["context_summary"] = ep.get("summary_text", "")
        if updates:
            set_clause = ", ".join(f"{k}=?" for k in updates)
            self.db.conn.execute(
                f"UPDATE episodic_memories SET {set_clause} WHERE id=?",
                list(updates.values()) + [ep["id"]],
            )
            self.db.conn.commit()

    def _fill_l3_fields(self, ep: dict):
        """Populate L3 fields."""
        self._fill_l2_fields(ep)
        updates = {}
        if not ep.get("narrative_thread"):
            updates["narrative_thread"] = ""
        if not ep.get("derived_lessons"):
            updates["derived_lessons"] = ""
        if updates:
            set_clause = ", ".join(f"{k}=?" for k in updates)
            self.db.conn.execute(
                f"UPDATE episodic_memories SET {set_clause} WHERE id=?",
                list(updates.values()) + [ep["id"]],
            )
            self.db.conn.commit()

    # --- Solution memory encoding ---

    def encode_solution(self, problem_type: str, problem_abstract: str,
                        attempts: list[SolutionAttempt],
                        dead_ends: list[DeadEndRecord],
                        context_tags: list[str] | None = None) -> SolutionMemory:
        """Encode a solution memory from problem-solving experience."""
        problem_embedding = self.embed(problem_abstract)

        # Find existing solution of same type
        existing = self.db.get_solution_by_type(problem_type)

        # Build attempts JSON
        attempts_json = json.dumps([{
            "attempt_id": a.attempt_id, "approach": a.approach,
            "outcome": a.outcome, "quality_score": a.quality_score,
            "is_best_known": a.is_best_known, "is_worst_known": a.is_worst_known,
            "episodic_memory_ids": a.episodic_memory_ids,
            "procedural_memory_id": a.procedural_memory_id,
            "tried_at": a.tried_at, "duration_minutes": a.duration_minutes,
            "why_succeeded": a.why_succeeded, "why_failed": a.why_failed,
        } for a in attempts], ensure_ascii=False)

        # Build dead ends JSON with approach embeddings
        de_list = []
        for de in dead_ends:
            de_dict = {
                "dead_end_id": de.dead_end_id, "approach": de.approach,
                "failure_mode": de.failure_mode, "failure_type": de.failure_type,
                "wasted_time_minutes": de.wasted_time_minutes,
                "tried_at": de.tried_at, "lessons": de.lessons,
                "searchable_symptoms": de.searchable_symptoms,
                "episodic_memory_ids": de.episodic_memory_ids,
                "was_eventually_solved": de.was_eventually_solved,
                "eventual_solution_type": de.eventual_solution_type,
            }
            de_list.append(de_dict)
        dead_ends_json = json.dumps(de_list, ensure_ascii=False)

        # Find best approach
        best = max((a for a in attempts if a.outcome == AttemptOutcome.SUCCESS),
                   key=lambda a: a.quality_score, default=None)
        best_approach = best.approach if best else ""
        best_approach_embedding = self.embed(best_approach) if best_approach else None

        # Find worst approach
        worst = min((a for a in attempts if a.outcome != AttemptOutcome.SUCCESS),
                    key=lambda a: a.quality_score, default=None)
        worst_approach = worst.approach if worst else ""
        worst_approach_why = worst.why_failed if worst else ""

        # Check for cross-problem failure patterns
        failure_patterns_json = json.dumps(
            self._detect_failure_patterns(dead_ends, problem_type), ensure_ascii=False)

        # Merge with existing if found
        failed_count = sum(1 for a in attempts if a.outcome == AttemptOutcome.FAILED)
        trial_count = len(attempts)

        if existing:
            sol_id = existing["id"]
            existing_attempts = json.loads(existing["attempts_json"] or "[]")
            existing_de = json.loads(existing["dead_ends_json"] or "[]")
            existing_fp = json.loads(existing["failure_patterns_json"] or "[]")
            merged_attempts = existing_attempts + json.loads(attempts_json)
            merged_de = existing_de + de_list
            merged_fp = existing_fp + json.loads(failure_patterns_json)
            new_strength = min(existing["strength"] + 0.1, 1.0)
            new_trial = existing["trial_count"] + trial_count
            new_failed = existing["failed_count"] + failed_count
        else:
            sol_id = new_id()
            merged_attempts = json.loads(attempts_json)
            merged_de = de_list
            merged_fp = json.loads(failure_patterns_json)
            new_strength = 0.5
            new_trial = trial_count
            new_failed = failed_count

        sol_data = {
            "id": sol_id,
            "problem_type": problem_type,
            "problem_abstract": problem_abstract,
            "problem_embedding": problem_embedding,
            "context_tags": json.dumps(context_tags or [], ensure_ascii=False),
            "attempts_json": json.dumps(merged_attempts, ensure_ascii=False),
            "dead_ends_json": json.dumps(merged_de, ensure_ascii=False),
            "failure_patterns_json": json.dumps(merged_fp, ensure_ascii=False),
            "best_approach": best_approach,
            "best_approach_embedding": best_approach_embedding,
            "worst_approach": worst_approach,
            "worst_approach_why": worst_approach_why,
            "status": "active",
            "trial_count": new_trial,
            "failed_count": new_failed,
            "last_encountered_at": now_iso(),
            "strength": new_strength,
            "encoding_depth": EncodingDepth.L2_STANDARD,
        }
        self.db.insert_solution(sol_data)

        # Insert dead end vectors for similarity search
        for de in dead_ends:
            de_vec = self.embed(de.approach)
            self.db.conn.execute(
                "INSERT OR REPLACE INTO dead_end_embeddings (id, approach_vector, failure_type, parent_solution_id) VALUES (?,?,?,?)",
                (de.dead_end_id, _vec_to_blob(de_vec), de.failure_type, sol_id),
            )

        self.db.conn.commit()

        sol = SolutionMemory(**{k: v for k, v in sol_data.items() if k in SolutionMemory.__dataclass_fields__})
        return sol

    def _detect_failure_patterns(self, dead_ends: list[DeadEndRecord],
                                  problem_type: str) -> list[dict]:
        """Detect recurring failure patterns across problems."""
        patterns = []
        for de in dead_ends:
            # Check if this failure type appears in other solution memories
            rows = self.db.conn.execute(
                "SELECT problem_type, dead_ends_json FROM solution_memories WHERE id != (SELECT id FROM solution_memories LIMIT 1)"
            ).fetchall()
            same_pattern = 0
            related_problems = []
            for row in rows:
                try:
                    existing_de = json.loads(row["dead_ends_json"] or "[]")
                except json.JSONDecodeError:
                    continue
                for ed in existing_de:
                    if ed.get("failure_type") == de.failure_type:
                        same_pattern += 1
                        if row["problem_type"] not in related_problems:
                            related_problems.append(row["problem_type"])
            if same_pattern >= 1:
                patterns.append({
                    "pattern": de.failure_mode,
                    "failure_count": same_pattern + 1,
                    "problem_types": related_problems + [problem_type],
                    "why_it_always_fails": de.lessons,
                    "early_warning_signal": f"症状: {de.searchable_symptoms}",
                })
        return patterns


class MaintenanceCycle:
    """Periodic maintenance: forgetting, depth upgrades, archiving."""

    def __init__(self, db: Database, encoder: Encoder, config: MemoryConfig = default_config):
        self.db = db
        self.encoder = encoder
        self.config = config

    def run(self):
        """Execute one full maintenance cycle."""
        # 1. Apply forgetting
        self.db.apply_forgetting()

        # 2. Archive excess forgotten memories (capacity-based, keep last 2000 forgotten)
        self.db.archive_forgotten(keep_forgotten_limit=2000)

        # 3. Check depth upgrades for recently reinforced memories
        rows = self.db.conn.execute(
            "SELECT id FROM episodic_memories WHERE is_forgotten = 0 AND encoding_depth < 3"
        ).fetchall()
        for r in rows:
            if self.encoder.check_depth_upgrade(r["id"]):
                self.encoder.perform_depth_upgrade(r["id"])

        # 4. Compact dead-end vectors: merge similar failure patterns
        self._compact_failure_patterns()

    def _compact_failure_patterns(self):
        """Find dead-end patterns that appear in >=2 problems and upgrade them to global warnings."""
        rows = self.db.conn.execute(
            "SELECT id, problem_type, failure_patterns_json FROM solution_memories WHERE status='active'"
        ).fetchall()
        all_patterns: dict[str, dict] = {}
        for row in rows:
            try:
                fps = json.loads(row["failure_patterns_json"] or "[]")
            except json.JSONDecodeError:
                continue
            for fp in fps:
                key = fp.get("pattern", "")
                if key not in all_patterns:
                    all_patterns[key] = {"pattern": key, "failure_count": 0, "problem_types": [], "why": ""}
                all_patterns[key]["failure_count"] += fp.get("failure_count", 1)
                all_patterns[key]["problem_types"].extend(fp.get("problem_types", []))
                all_patterns[key]["why"] = fp.get("why_it_always_fails", "")
                all_patterns[key]["problem_types"] = list(set(all_patterns[key]["problem_types"]))

        # Update patterns for solutions with >=2 occurrences
        for pattern_key, data in all_patterns.items():
            if data["failure_count"] >= 2:
                for row in rows:
                    try:
                        fps = json.loads(row["failure_patterns_json"] or "[]")
                    except json.JSONDecodeError:
                        continue
                    for fp in fps:
                        if fp.get("pattern") == pattern_key:
                            fp["failure_count"] = data["failure_count"]
                            fp["problem_types"] = data["problem_types"]
                    self.db.conn.execute(
                        "UPDATE solution_memories SET failure_patterns_json=? WHERE id=?",
                        (json.dumps(fps, ensure_ascii=False), row["id"]),
                    )
        self.db.conn.commit()
