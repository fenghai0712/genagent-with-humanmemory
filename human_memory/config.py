from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    # Working memory
    wm_capacity: int = 7
    wm_slot_ttl_seconds: float = 300.0

    # Encoding
    consolidation_trigger_wm_full: bool = True
    consolidation_trigger_session_end: bool = True
    consolidation_score_threshold: float = 0.3
    attention_weight: float = 0.3
    emotion_weight: float = 0.25
    novelty_weight: float = 0.25
    explicit_signal_weight: float = 0.2

    # Encoding depth thresholds (strength-based)
    depth_l2_threshold: float = 0.3
    depth_l3_threshold: float = 0.7

    # Reinforcement
    base_recall_increment: float = 0.1
    association_increment: float = 0.05
    emotional_context_increment: float = 0.2
    explicit_mark_increment: float = 0.3
    rapid_recall_increment: float = 0.15
    rapid_recall_window_hours: float = 1.0
    spacing_tau_days: float = 1.0

    # Forgetting — capacity-based (interference model, not time-decay)
    episodic_capacity: int = 5000        # max episodic memories before eviction kicks in
    eviction_batch_size: int = 50        # how many to evict at once when over capacity
    forgotten_retrieval_threshold: float = 0.1  # below this strength = "forgotten" (not recalled by default)

    # Dead-end similarity threshold (L2 on unit-normalized vectors: 0=identical, ~1.4=opposite)
    dead_end_similarity_threshold: float = 1.0

    # Maintenance
    maintenance_interval_seconds: float = 3600.0

    # Database
    db_path: str = "memory.db"

    # Embedding
    embedding_dim: int = 384              # auto-adjusted from loaded model on init
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"


default_config = MemoryConfig()
