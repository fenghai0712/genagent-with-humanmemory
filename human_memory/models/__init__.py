from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import time
import uuid


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class EncodingDepth(IntEnum):
    L1_SHALLOW = 1
    L2_STANDARD = 2
    L3_DEEP = 3


class AttemptOutcome:
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class FailureType:
    ARCHITECTURE = "architecture"
    LOGIC = "logic"
    CONFIG = "config"
    DEPENDENCY = "dependency"
    TIMING = "timing"


@dataclass
class WorkingMemorySlot:
    slot_id: str = field(default_factory=new_id)
    content: str = ""
    summary: str = ""
    attention_weight: float = 0.5
    emotional_intensity: float = 0.0
    novelty_score: float = 0.5
    explicit_signal: bool = False
    context_tags: list[str] = field(default_factory=list)
    entered_at: str = field(default_factory=now_iso)
    last_accessed_at: str = field(default_factory=now_iso)


@dataclass
class EpisodicMemory:
    id: str = field(default_factory=new_id)
    encoding_depth: int = EncodingDepth.L1_SHALLOW
    strength: float = 0.2

    # L1
    summary_text: str = ""
    summary_vec: Optional[list[float]] = None
    importance_initial: float = 0.5
    created_at: str = field(default_factory=now_iso)

    # L2
    entities_json: Optional[str] = None
    emotion_tags: Optional[str] = None
    context_summary: Optional[str] = None
    related_memory_ids: Optional[str] = None
    context_tags: Optional[str] = None

    # L3
    full_context_json: Optional[str] = None
    narrative_thread: Optional[str] = None
    derived_lessons: Optional[str] = None
    sensory_snapshot: Optional[str] = None

    # Forgetting
    last_recalled_at: Optional[str] = None
    recall_count: int = 0
    retention: float = 1.0
    is_forgotten: int = 0
    forgotten_at: Optional[str] = None


@dataclass
class Concept:
    id: str = field(default_factory=new_id)
    name: str = ""
    embedding: Optional[list[float]] = None
    parent_id: Optional[str] = None
    description: str = ""
    strength: float = 0.5
    created_at: str = field(default_factory=now_iso)


@dataclass
class ConceptEdge:
    id: str = field(default_factory=new_id)
    source_id: str = ""
    target_id: str = ""
    relation_type: str = ""  # "is_a", "related_to", "part_of", "example_of"
    strength: float = 0.5


@dataclass
class Skill:
    id: str = field(default_factory=new_id)
    name: str = ""
    description: str = ""
    trigger_condition: str = ""
    steps_json: Optional[str] = None
    proficiency: float = 0.2
    context_tags: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    last_used_at: Optional[str] = None
    use_count: int = 0


@dataclass
class SolutionMemory:
    id: str = field(default_factory=new_id)
    problem_type: str = ""
    problem_abstract: str = ""
    problem_embedding: Optional[list[float]] = None
    context_tags: str = "[]"

    attempts_json: str = "[]"
    dead_ends_json: str = "[]"
    failure_patterns_json: str = "[]"

    best_approach: str = ""
    best_approach_embedding: Optional[list[float]] = None
    worst_approach: str = ""
    worst_approach_why: str = ""

    status: str = "active"
    trial_count: int = 0
    failed_count: int = 0
    last_encountered_at: Optional[str] = None
    strength: float = 0.5
    encoding_depth: int = EncodingDepth.L2_STANDARD


@dataclass
class DeadEndRecord:
    dead_end_id: str = field(default_factory=new_id)
    approach: str = ""
    approach_embedding: Optional[list[float]] = None
    failure_mode: str = ""
    failure_type: str = ""
    wasted_time_minutes: int = 0
    tried_at: str = field(default_factory=now_iso)
    lessons: str = ""
    searchable_symptoms: str = "[]"
    episodic_memory_ids: str = "[]"
    was_eventually_solved: bool = False
    eventual_solution_type: str = ""


@dataclass
class SolutionAttempt:
    attempt_id: str = field(default_factory=new_id)
    approach: str = ""
    outcome: str = ""  # success | partial | failed
    quality_score: float = 5.0
    is_best_known: bool = False
    is_worst_known: bool = False
    episodic_memory_ids: str = "[]"
    procedural_memory_id: str = ""
    tried_at: str = field(default_factory=now_iso)
    duration_minutes: int = 0
    why_succeeded: str = ""
    why_failed: str = ""
