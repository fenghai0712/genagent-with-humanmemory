import time
import heapq
from typing import Optional

from human_memory.config import MemoryConfig, default_config
from human_memory.models import WorkingMemorySlot, now_iso, new_id


class WorkingMemory:
    """In-memory system with 7±2 slots, attention-weighted, LRU eviction."""

    def __init__(self, config: MemoryConfig = default_config):
        self.config = config
        self.slots: list[WorkingMemorySlot] = []

    def __len__(self) -> int:
        return len(self.slots)

    @property
    def is_full(self) -> bool:
        return len(self.slots) >= self.config.wm_capacity

    def _score_slot(self, slot: WorkingMemorySlot) -> float:
        """Composite score for eviction priority. Lower = evict first."""
        try:
            ts = time.mktime(time.strptime(slot.last_accessed_at, "%Y-%m-%dT%H:%M:%S"))
            age_seconds = time.time() - ts
        except Exception:
            age_seconds = 0

        # Also check TTL
        if age_seconds > self.config.wm_slot_ttl_seconds:
            return -1.0  # force eviction

        # Higher attention + emotion + explicit = less evictable (higher score)
        stickiness = (slot.attention_weight * 0.4 +
                      slot.emotional_intensity * 0.35 +
                      (1.0 if slot.explicit_signal else 0.0) * 0.25)
        # Recency bonus
        recency = max(0, 1.0 - age_seconds / self.config.wm_slot_ttl_seconds)
        return stickiness * 0.6 + recency * 0.4

    def enter(self, content: str, summary: str = "",
              attention_weight: float = 0.5, emotional_intensity: float = 0.0,
              explicit_signal: bool = False, context_tags: Optional[list[str]] = None,
              novelty_score: float = 0.5) -> WorkingMemorySlot:
        """Push a new item into working memory. Evicts if full and consolidates the evicted."""
        self._evict_expired()

        slot = WorkingMemorySlot(
            content=content,
            summary=summary or content[:200],
            attention_weight=attention_weight,
            emotional_intensity=emotional_intensity,
            novelty_score=novelty_score,
            explicit_signal=explicit_signal,
            context_tags=context_tags or [],
        )

        if self.is_full:
            self._evict_one()

        self.slots.append(slot)
        return slot

    def _evict_expired(self):
        self.slots = [s for s in self.slots if self._score_slot(s) >= 0]

    def _evict_one(self) -> Optional[WorkingMemorySlot]:
        if not self.slots:
            return None
        scored = [(self._score_slot(s), i, s) for i, s in enumerate(self.slots)]
        scored.sort(key=lambda x: x[0])
        _, idx, evicted = scored[0]
        self.slots.pop(idx)
        return evicted

    def evict_for_consolidation(self) -> list[WorkingMemorySlot]:
        """Evict all slots that are candidates for consolidation (session end or manual trigger).
        Returns the list for encoding into LTM."""
        self._evict_expired()
        evicted = list(self.slots)
        self.slots.clear()
        return evicted

    def access(self, slot_id: str) -> Optional[WorkingMemorySlot]:
        for s in self.slots:
            if s.slot_id == slot_id:
                s.last_accessed_at = now_iso()
                s.attention_weight = min(s.attention_weight + 0.1, 1.0)
                return s
        return None

    def get_all(self) -> list[WorkingMemorySlot]:
        self._evict_expired()
        return list(self.slots)

    def get_attention_sorted(self) -> list[WorkingMemorySlot]:
        items = self.get_all()
        items.sort(key=lambda s: s.attention_weight, reverse=True)
        return items

    def clear(self):
        self.slots.clear()
