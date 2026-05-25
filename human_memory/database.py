import sqlite3
import json
import struct
import os
import uuid
from typing import Optional

from human_memory.config import MemoryConfig, default_config


def _now_iso() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS episodic_memories (
    id TEXT PRIMARY KEY,
    encoding_depth INTEGER NOT NULL DEFAULT 1,
    strength REAL NOT NULL DEFAULT 0.2,

    summary_text TEXT NOT NULL DEFAULT '',
    summary_vec BLOB,
    importance_initial REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,

    entities_json TEXT,
    emotion_tags TEXT,
    context_summary TEXT,
    related_memory_ids TEXT,
    context_tags TEXT,

    full_context_json TEXT,
    narrative_thread TEXT,
    derived_lessons TEXT,
    sensory_snapshot TEXT,

    last_recalled_at TEXT,
    recall_count INTEGER DEFAULT 0,
    retention REAL DEFAULT 1.0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_ep_created ON episodic_memories(created_at);
CREATE INDEX IF NOT EXISTS idx_ep_strength ON episodic_memories(strength);
CREATE INDEX IF NOT EXISTS idx_ep_forgotten ON episodic_memories(is_forgotten);
CREATE INDEX IF NOT EXISTS idx_ep_depth ON episodic_memories(encoding_depth);

CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    embedding BLOB,
    parent_id TEXT,
    description TEXT DEFAULT '',
    strength REAL DEFAULT 0.5,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_concept_name ON concepts(name);
CREATE INDEX IF NOT EXISTS idx_concept_parent ON concepts(parent_id);

CREATE TABLE IF NOT EXISTS concept_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL DEFAULT 'related_to',
    strength REAL DEFAULT 0.5,
    FOREIGN KEY (source_id) REFERENCES concepts(id),
    FOREIGN KEY (target_id) REFERENCES concepts(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_edge_unique
    ON concept_edges(source_id, target_id, relation_type);

CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    trigger_condition TEXT DEFAULT '',
    steps_json TEXT,
    proficiency REAL DEFAULT 0.2,
    context_tags TEXT,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    use_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_skill_name ON skills(name);
CREATE INDEX IF NOT EXISTS idx_skill_proficiency ON skills(proficiency);

CREATE TABLE IF NOT EXISTS solution_memories (
    id TEXT PRIMARY KEY,
    problem_type TEXT NOT NULL,
    problem_abstract TEXT NOT NULL DEFAULT '',
    problem_embedding BLOB,
    context_tags TEXT DEFAULT '[]',

    attempts_json TEXT DEFAULT '[]',
    dead_ends_json TEXT DEFAULT '[]',
    failure_patterns_json TEXT DEFAULT '[]',

    best_approach TEXT DEFAULT '',
    best_approach_embedding BLOB,
    worst_approach TEXT DEFAULT '',
    worst_approach_why TEXT DEFAULT '',

    status TEXT DEFAULT 'active',
    trial_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    last_encountered_at TEXT,
    strength REAL DEFAULT 0.5,
    encoding_depth INTEGER DEFAULT 2
);

CREATE INDEX IF NOT EXISTS idx_sol_problem_type ON solution_memories(problem_type);
CREATE INDEX IF NOT EXISTS idx_sol_status ON solution_memories(status);
CREATE INDEX IF NOT EXISTS idx_sol_strength ON solution_memories(strength);

CREATE TABLE IF NOT EXISTS discarded_episodes (
    id TEXT PRIMARY KEY,
    content_summary TEXT,
    consolidation_score REAL,
    reason TEXT,
    discarded_at TEXT
);

CREATE TABLE IF NOT EXISTS archived_memories (
    id TEXT PRIMARY KEY,
    original_table TEXT NOT NULL,
    original_id TEXT NOT NULL,
    compressed_data BLOB,
    archived_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_arch_table ON archived_memories(original_table);
CREATE INDEX IF NOT EXISTS idx_arch_date ON archived_memories(archived_at);

CREATE TABLE IF NOT EXISTS dead_end_embeddings (
    id TEXT PRIMARY KEY,
    approach_vector BLOB NOT NULL,
    failure_type TEXT,
    parent_solution_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dead_end_parent ON dead_end_embeddings(parent_solution_id);

CREATE TABLE IF NOT EXISTS emotion_entries (
    id TEXT PRIMARY KEY,
    emotion_label TEXT NOT NULL,
    intensity REAL DEFAULT 0.5,
    source_memory_type TEXT NOT NULL,
    source_memory_id TEXT NOT NULL,
    context TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_emotion_label ON emotion_entries(emotion_label);

CREATE TABLE IF NOT EXISTS prospective_items (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    trigger_time TEXT,
    trigger_context_embedding BLOB,
    status TEXT DEFAULT 'pending',
    reminder_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_pros_status ON prospective_items(status);
CREATE INDEX IF NOT EXISTS idx_pros_trigger_time ON prospective_items(trigger_time);
"""


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class Database:
    def __init__(self, config: MemoryConfig = default_config):
        self.config = config
        db_dir = os.path.dirname(os.path.abspath(config.db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self.conn = sqlite3.connect(config.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._init_vec_tables()

    def _init_vec_tables(self):
        try:
            self.conn.execute("SELECT vec_version()")
        except Exception:
            import sqlite_vec
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self.conn.execute("SELECT vec_version()")

        self.conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodic_vec USING vec0(
                id TEXT PRIMARY KEY,
                summary_vec FLOAT[{dim}]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS concept_vec USING vec0(
                id TEXT PRIMARY KEY,
                embedding FLOAT[{dim}]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS problem_vec USING vec0(
                id TEXT PRIMARY KEY,
                problem_embedding FLOAT[{dim}]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS best_approach_vec USING vec0(
                id TEXT PRIMARY KEY,
                best_approach_embedding FLOAT[{dim}]
            );
        """.format(dim=self.config.embedding_dim))

    def close(self):
        self.conn.close()

    # --- episodic ---

    def insert_episodic(self, ep: dict) -> str:
        if ep.get("summary_vec") and isinstance(ep["summary_vec"], list):
            vec_blob = _vec_to_blob(ep["summary_vec"])
            ep["summary_vec"] = vec_blob
        cols = ", ".join(ep.keys())
        placeholders = ", ".join("?" for _ in ep)
        self.conn.execute(
            f"INSERT OR REPLACE INTO episodic_memories ({cols}) VALUES ({placeholders})",
            list(ep.values()),
        )
        if ep.get("summary_vec"):
            self._insert_vec("episodic_vec", "summary_vec", ep["id"],
                             _blob_to_vec(ep["summary_vec"]))
        self.conn.commit()
        return ep["id"]

    def get_episodic(self, memory_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM episodic_memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def search_similar_episodes(self, query_vec: list[float], limit: int = 10,
                                 include_forgotten: bool = False) -> list[dict]:
        forgotten_filter = "" if include_forgotten else "AND e.is_forgotten = 0"
        rows = self.conn.execute(f"""
            SELECT e.*, vec_distance_L2(?, e.summary_vec) as distance
            FROM episodic_memories e
            WHERE e.summary_vec IS NOT NULL {forgotten_filter}
            ORDER BY distance ASC
            LIMIT ?
        """, (_vec_to_blob(query_vec), limit)).fetchall()
        return [dict(r) for r in rows]

    def soft_forget(self, memory_id: str):
        self.conn.execute(
            "UPDATE episodic_memories SET is_forgotten=1, forgotten_at=? WHERE id=?",
            (_now_iso(), memory_id),
        )
        self.conn.commit()

    def archive_forgotten(self, keep_forgotten_limit: int = 2000):
        """Archive soft-forgotten memories when the forgotten pile exceeds the limit.
        Evicts the weakest forgotten memories first, moving them to archive."""
        total_forgotten = self.conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memories WHERE is_forgotten = 1"
        ).fetchone()["c"]
        if total_forgotten <= keep_forgotten_limit:
            return
        excess = total_forgotten - keep_forgotten_limit
        rows = self.conn.execute(
            "SELECT id FROM episodic_memories WHERE is_forgotten = 1 ORDER BY strength ASC LIMIT ?",
            (excess,),
        ).fetchall()
        for r in rows:
            self.conn.execute(
                "INSERT INTO archived_memories (id, original_table, original_id, compressed_data, archived_at) VALUES (?, 'episodic_memories', ?, zeroblob(0), ?)",
                (uuid.uuid4().hex[:12], r["id"], _now_iso()),
            )
            self.conn.execute(
                "DELETE FROM episodic_memories WHERE id = ?", (r["id"],),
            )
        self.conn.commit()

    # --- semantic ---

    def upsert_concept(self, concept: dict) -> str:
        if concept.get("embedding") and isinstance(concept["embedding"], list):
            concept["embedding"] = _vec_to_blob(concept["embedding"])
        self.conn.execute("""
            INSERT INTO concepts (id, name, embedding, parent_id, description, strength, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                embedding=excluded.embedding,
                strength=excluded.strength,
                description=excluded.description
        """, (concept["id"], concept["name"], concept.get("embedding"),
              concept.get("parent_id"), concept.get("description", ""),
              concept.get("strength", 0.5), concept.get("created_at")))
        if concept.get("embedding"):
            self._insert_vec("concept_vec", "embedding", concept["id"],
                             _blob_to_vec(concept["embedding"]))
        self.conn.commit()
        return concept["id"]

    def insert_edge(self, edge: dict):
        self.conn.execute("""
            INSERT OR IGNORE INTO concept_edges (id, source_id, target_id, relation_type, strength)
            VALUES (?, ?, ?, ?, ?)
        """, (edge["id"], edge["source_id"], edge["target_id"],
              edge.get("relation_type", "related_to"), edge.get("strength", 0.5)))
        self.conn.commit()

    def get_concept_hierarchy(self, concept_id: str, max_depth: int = 5) -> list[dict]:
        rows = self.conn.execute("""
            WITH RECURSIVE hierarchy AS (
                SELECT id, name, parent_id, strength, 0 AS depth
                FROM concepts WHERE id = ?
                UNION ALL
                SELECT c.id, c.name, c.parent_id, c.strength, h.depth + 1
                FROM concepts c JOIN hierarchy h ON c.parent_id = h.id
                WHERE h.depth < ?
            )
            SELECT * FROM hierarchy ORDER BY depth
        """, (concept_id, max_depth)).fetchall()
        return [dict(r) for r in rows]

    # --- solution ---

    def insert_solution(self, sol: dict) -> str:
        for vec_field in ["problem_embedding", "best_approach_embedding"]:
            if sol.get(vec_field) and isinstance(sol[vec_field], list):
                sol[vec_field] = _vec_to_blob(sol[vec_field])
        cols = ", ".join(sol.keys())
        placeholders = ", ".join("?" for _ in sol)
        self.conn.execute(
            f"INSERT OR REPLACE INTO solution_memories ({cols}) VALUES ({placeholders})",
            list(sol.values()),
        )
        if sol.get("problem_embedding"):
            self._insert_vec("problem_vec", "problem_embedding", sol["id"],
                             _blob_to_vec(sol["problem_embedding"]))
        if sol.get("best_approach_embedding"):
            self._insert_vec("best_approach_vec", "best_approach_embedding", sol["id"],
                             _blob_to_vec(sol["best_approach_embedding"]))
        self.conn.commit()
        return sol["id"]

    def search_similar_problems(self, query_vec: list[float], limit: int = 5) -> list[dict]:
        rows = self.conn.execute(f"""
            SELECT s.*, vec_distance_L2(?, s.problem_embedding) as distance
            FROM solution_memories s
            WHERE s.problem_embedding IS NOT NULL AND s.status = 'active'
            ORDER BY distance ASC LIMIT ?
        """, (_vec_to_blob(query_vec), limit)).fetchall()
        return [dict(r) for r in rows]

    def search_dead_ends(self, approach_vec: list[float], limit: int = 3) -> list[dict]:
        rows = self.conn.execute(f"""
            SELECT s.id, s.problem_type, s.dead_ends_json,
                   vec_distance_L2(?, s.best_approach_embedding) as distance
            FROM solution_memories s
            WHERE s.dead_ends_json IS NOT NULL AND s.dead_ends_json != '[]'
            ORDER BY distance ASC LIMIT ?
        """, (_vec_to_blob(approach_vec), limit)).fetchall()
        return [dict(r) for r in rows]

    def get_solution_by_type(self, problem_type: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM solution_memories WHERE problem_type = ? AND status = 'active' ORDER BY strength DESC LIMIT 1",
            (problem_type,),
        ).fetchone()
        return dict(row) if row else None

    # --- skills ---

    def upsert_skill(self, skill: dict) -> str:
        cols = ", ".join(skill.keys())
        placeholders = ", ".join("?" for _ in skill)
        self.conn.execute(
            f"INSERT OR REPLACE INTO skills ({cols}) VALUES ({placeholders})",
            list(skill.values()),
        )
        self.conn.commit()
        return skill["id"]

    def get_skill(self, skill_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        return dict(row) if row else None

    # --- helpers ---

    def _insert_vec(self, table: str, col: str, row_id: str, vec: list[float]):
        try:
            self.conn.execute(
                f"INSERT OR REPLACE INTO {table} (id, {col}) VALUES (?, ?)",
                (row_id, _vec_to_blob(vec)),
            )
        except Exception:
            pass  # vec table may not exist yet during first init

    def compute_retention(self, memory_id: str) -> float:
        """Retention is now driven by strength alone (capacity-based competition).
        A memory's 'retention' = its strength. Weaker memories get evicted when
        capacity is exceeded, regardless of age."""
        row = self.conn.execute(
            "SELECT strength FROM episodic_memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        if not row:
            return 0.0
        return row["strength"]

    def apply_forgetting(self):
        """Capacity-based forgetting: when active episodic count exceeds capacity,
        evict the weakest memories. This models interference-based forgetting
        rather than time-decay — new memories push out old/weak ones."""
        active_count = self.conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memories WHERE is_forgotten = 0"
        ).fetchone()["c"]
        capacity = self.config.episodic_capacity
        if active_count <= capacity:
            return

        excess = active_count - capacity
        batch = min(excess, self.config.eviction_batch_size)

        # Mark weakest active memories as forgotten
        self.conn.execute("""
            UPDATE episodic_memories SET is_forgotten = 1, forgotten_at = ?
            WHERE id IN (
                SELECT id FROM episodic_memories
                WHERE is_forgotten = 0
                ORDER BY strength ASC
                LIMIT ?
            )
        """, (_now_iso(), batch))
        self.conn.commit()

    def reinforce(self, memory_id: str, increment: float):
        self.conn.execute("""
            UPDATE episodic_memories
            SET strength = MIN(strength + ?, 1.0),
                recall_count = recall_count + 1,
                last_recalled_at = ?
            WHERE id = ?
        """, (increment, __import__('time').strftime("%Y-%m-%dT%H:%M:%S",
                                                      __import__('time').localtime()), memory_id))
        self.conn.commit()
