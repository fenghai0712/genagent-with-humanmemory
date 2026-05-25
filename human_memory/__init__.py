"""Human-like memory system for AI agents.

Simulates working memory, episodic memory, semantic memory, procedural memory,
and solution memory — with capacity-based forgetting, reinforcement, and
dead-end avoidance.

Usage:
    from human_memory import MemoryManager

    mm = MemoryManager()
    mm.remember("user asked about database performance")
    mm.consolidate()
    results = mm.recall("database")
    mm.end_session()
"""

from human_memory.memory_manager import MemoryManager
from human_memory.config import MemoryConfig
from human_memory.embedding import EmbeddingProvider
from human_memory.models import (
    SolutionAttempt,
    DeadEndRecord,
    AttemptOutcome,
    FailureType,
)

__all__ = [
    "MemoryManager",
    "MemoryConfig",
    "EmbeddingProvider",
    "SolutionAttempt",
    "DeadEndRecord",
    "AttemptOutcome",
    "FailureType",
]
