"""Integration tests for the complete memory system pipeline."""

import time
import os
import pytest

from human_memory.config import MemoryConfig
from human_memory.database import Database
from human_memory.working_memory import WorkingMemory
from human_memory.encoding import Encoder, MaintenanceCycle, ConsolidationScorer
from human_memory.retrieval import RetrievalEngine
from human_memory.memory_manager import MemoryManager
from human_memory.models import (
    WorkingMemorySlot, EpisodicMemory, Concept, ConceptEdge,
    SolutionMemory, SolutionAttempt, DeadEndRecord,
    EncodingDepth, AttemptOutcome, FailureType,
    now_iso, new_id,
)


TEST_DB = "test_memory.db"


@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def config():
    return MemoryConfig(db_path=TEST_DB)


@pytest.fixture
def db(config):
    d = Database(config)
    yield d
    d.close()


@pytest.fixture
def wm(config):
    return WorkingMemory(config)


@pytest.fixture
def encoder(db, config):
    return Encoder(db, config)


@pytest.fixture
def retrieval(db, config, encoder):
    return RetrievalEngine(db, config, embed_fn=encoder.embed)


@pytest.fixture
def mm(config):
    m = MemoryManager(config)
    yield m
    m.close()


# --- Working Memory Tests ---

class TestWorkingMemory:
    def test_enter_slot(self, wm):
        slot = wm.enter("test content", summary="test")
        assert len(wm) == 1
        assert slot.content == "test content"

    def test_eviction_when_full(self, wm):
        for i in range(wm.config.wm_capacity):
            wm.enter(f"content {i}", attention_weight=0.3)
        assert wm.is_full
        evicted = wm._evict_one()
        assert evicted is not None
        assert len(wm) == wm.config.wm_capacity - 1

    def test_high_attention_protected(self, wm):
        # Fill with low-attention items
        for i in range(wm.config.wm_capacity - 1):
            wm.enter(f"low {i}", attention_weight=0.2)
        # Add high-attention item
        wm.enter("important", attention_weight=1.0, explicit_signal=True)
        assert wm.is_full
        evicted = wm._evict_one()
        # The high-attention item should NOT be evicted
        assert evicted is not None
        assert "important" not in evicted.content

    def test_enter_when_full_auto_evicts(self, wm):
        for i in range(wm.config.wm_capacity):
            wm.enter(f"content {i}", attention_weight=0.1)
        slot = wm.enter("new one", attention_weight=0.5)
        assert len(wm) <= wm.config.wm_capacity
        assert slot.content == "new one"

    def test_evict_for_consolidation(self, wm):
        wm.enter("item 1")
        wm.enter("item 2")
        evicted = wm.evict_for_consolidation()
        assert len(evicted) == 2
        assert len(wm) == 0

    def test_access_updates_timestamp_and_attention(self, wm):
        slot = wm.enter("test", attention_weight=0.3)
        wm.access(slot.slot_id)
        assert slot.attention_weight > 0.3


# --- Consolidation Scorer Tests ---

class TestConsolidationScorer:
    def test_high_explicit_signal_scores_high(self, config):
        scorer = ConsolidationScorer(config)
        slot = WorkingMemorySlot(
            content="important", attention_weight=0.8,
            emotional_intensity=0.5, explicit_signal=True, novelty_score=0.7,
        )
        score = scorer.score(slot)
        assert score > 0.5

    def test_low_attention_does_not_encode(self, config):
        scorer = ConsolidationScorer(config)
        slot = WorkingMemorySlot(
            content="noise", attention_weight=0.05,
            emotional_intensity=0.0, explicit_signal=False, novelty_score=0.1,
        )
        assert not scorer.should_encode(slot)


# --- Encoding Tests ---

class TestEncoding:
    def test_consolidate_encodes_worthy_slots(self, encoder, db):
        slot = WorkingMemorySlot(
            content="crucial discovery", summary="key finding",
            attention_weight=0.9, emotional_intensity=0.8,
            explicit_signal=True, novelty_score=0.9,
        )
        encoded, discarded = encoder.consolidate([slot])
        assert len(encoded) == 1
        assert len(discarded) == 0
        ep = encoded[0]
        assert ep.summary_text == "key finding"
        assert ep.encoding_depth >= EncodingDepth.L1_SHALLOW

    def test_consolidate_discards_low_quality(self, encoder, db):
        slot = WorkingMemorySlot(
            content="meh", summary="whatever",
            attention_weight=0.05, emotional_intensity=0.0,
            novelty_score=0.05,
        )
        encoded, discarded = encoder.consolidate([slot])
        assert len(encoded) == 0
        assert len(discarded) == 1

    def test_depth_upgrade(self, encoder, db):
        slot = WorkingMemorySlot(
            content="significant", summary="important event",
            attention_weight=0.8, emotional_intensity=0.6,
            novelty_score=0.5,
        )
        encoded, _ = encoder.consolidate([slot])
        ep = encoded[0]

        # Simulate many reinforcements
        for _ in range(10):
            encoder.reinforce(ep.id, is_recall=True)

        assert encoder.check_depth_upgrade(ep.id)
        encoder.perform_depth_upgrade(ep.id)

        updated = db.get_episodic(ep.id)
        assert updated["encoding_depth"] >= EncodingDepth.L2_STANDARD

    def test_record_discard(self, encoder, db):
        slot = WorkingMemorySlot(
            content="noise", attention_weight=0.01, novelty_score=0.01,
        )
        encoded, discarded = encoder.consolidate([slot])
        assert len(discarded) == 1
        # Check it was recorded
        row = db.conn.execute(
            "SELECT * FROM discarded_episodes WHERE reason != ''"
        ).fetchone()
        assert row is not None


# --- Solution Memory Tests ---

class TestSolutionMemory:
    def test_encode_solution(self, encoder, db):
        attempts = [
            SolutionAttempt(
                approach="用 asyncio.gather 替代嵌套 await",
                outcome=AttemptOutcome.SUCCESS,
                quality_score=9.0,
                is_best_known=True,
                why_succeeded="避免了事件循环嵌套",
            ),
            SolutionAttempt(
                approach="用 threading + async 混用",
                outcome=AttemptOutcome.FAILED,
                quality_score=1.0,
                is_worst_known=True,
                why_failed="两个事件循环互相抢占",
            ),
        ]
        dead_ends = [
            DeadEndRecord(
                approach="threading + async 混用",
                failure_mode="不可复现的竞态条件",
                failure_type=FailureType.ARCHITECTURE,
                wasted_time_minutes=180,
                lessons="异步和线程不能混用",
                searchable_symptoms="[\"超时不固定\", \"偶发死锁\"]",
            ),
        ]
        sol = encoder.encode_solution(
            problem_type="Python异步死锁",
            problem_abstract="FastAPI 里多个异步任务互相等待导致超时",
            attempts=attempts,
            dead_ends=dead_ends,
            context_tags=["python", "asyncio", "fastapi"],
        )
        assert sol.problem_type == "Python异步死锁"
        assert sol.trial_count == 2
        assert sol.failed_count == 1
        assert "asyncio.gather" in sol.best_approach
        assert "threading" in sol.worst_approach

    def test_solution_merging(self, encoder, db):
        # First solution
        encoder.encode_solution(
            problem_type="PerformanceIssue",
            problem_abstract="Query too slow",
            attempts=[
                SolutionAttempt(approach="Add index", outcome=AttemptOutcome.SUCCESS,
                                quality_score=8.0),
            ],
            dead_ends=[],
        )
        # Second attempt on same type
        sol2 = encoder.encode_solution(
            problem_type="PerformanceIssue",
            problem_abstract="Another slow query",
            attempts=[
                SolutionAttempt(approach="Rewrite query", outcome=AttemptOutcome.SUCCESS,
                                quality_score=9.0),
            ],
            dead_ends=[],
        )
        assert sol2.trial_count >= 2
        assert sol2.strength >= 0.55  # increased from merge


# --- Retrieval Tests ---

class TestRetrieval:
    def test_recall_similar(self, mm):
        mm.remember("PostgreSQL 递归 CTE 可以解决图遍历", attention_weight=0.8,
                     explicit_signal=True)
        mm.remember("今天天气不错", attention_weight=0.2)
        mm.consolidate()

        results = mm.recall("数据库查询")
        assert len(results) > 0

    def test_cross_type_recall(self, mm):
        mm.remember("Python 异步编程调试技巧", attention_weight=0.9,
                     explicit_signal=True)
        mm.consolidate()
        mm.learn_concept("Python异步编程", "与 asyncio 相关的知识")

        results = mm.recall_all("异步")
        assert "episodic" in results
        assert "concepts" in results

    def test_solve_problem_pipeline(self, mm):
        # Record a known solution
        mm.record_solution(
            problem_type="Python异步死锁",
            problem_abstract="嵌套 await 导致死锁",
            attempts=[
                SolutionAttempt(
                    approach="用 asyncio.gather 替代嵌套 await",
                    outcome=AttemptOutcome.SUCCESS,
                    quality_score=9.0,
                    is_best_known=True,
                ),
            ],
            dead_ends=[
                DeadEndRecord(
                    approach="threading + async 混用",
                    failure_mode="竞态条件",
                    failure_type=FailureType.ARCHITECTURE,
                    wasted_time_minutes=120,
                    lessons="不要混用",
                ),
            ],
        )

        # Now solve a similar problem
        result = mm.solve_problem(
            problem_description="asyncio 等待超时",
            proposed_approach="我打算用 threading 配合 async",
        )

        assert "solutions" in result
        assert len(result["solutions"]) > 0


# --- Forgetting Tests ---

class TestForgetting:
    def test_capacity_eviction(self, mm):
        """When episodic count exceeds capacity, weakest memories are evicted."""
        # Temporarily lower capacity to force eviction
        mm.config.episodic_capacity = 3
        mm.config.eviction_batch_size = 10

        # Encode 5 memories with varying strengths
        for i in range(5):
            mm.remember(f"memory {i}", attention_weight=0.6, emotional_intensity=0.4)
        mm.consolidate()

        # Artificially set strengths so some are very weak
        rows = mm.db.conn.execute(
            "SELECT id, summary_text FROM episodic_memories ORDER BY created_at"
        ).fetchall()
        for j, r in enumerate(rows):
            mm.db.conn.execute(
                "UPDATE episodic_memories SET strength = ? WHERE id = ?",
                (0.1 + j * 0.15, r["id"]),  # 0.10, 0.25, 0.40, 0.55, 0.70
            )
        mm.db.conn.commit()

        # Run maintenance — should evict the weakest to get down to capacity (3)
        mm.run_maintenance()
        stats = mm.stats()
        assert stats["episodic_active"] <= 3
        assert stats["episodic_forgotten"] >= 2

        # The strongest should survive
        active = mm.db.conn.execute(
            "SELECT strength FROM episodic_memories WHERE is_forgotten = 0 ORDER BY strength"
        ).fetchall()
        # Survivors should be the top 3 strongest
        assert len(active) == 3
        assert active[0]["strength"] >= 0.4  # weakest survivor is the 3rd strongest

    def test_archive_forgotten(self, mm):
        # Insert many weak forgotten memories
        for _ in range(10):
            mm.db.conn.execute(
                "INSERT INTO episodic_memories (id, summary_text, created_at, is_forgotten, forgotten_at, strength) "
                "VALUES (?, ?, '2020-01-01T00:00:00', 1, '2020-01-02T00:00:00', 0.01)",
                (new_id(), "ancient"),
            )
        mm.db.conn.commit()

        # Archive with limit of 5 — excess should go to archive
        mm.db.archive_forgotten(keep_forgotten_limit=5)

        remaining = mm.db.conn.execute(
            "SELECT COUNT(*) as c FROM episodic_memories WHERE is_forgotten = 1"
        ).fetchone()["c"]
        archived = mm.db.conn.execute(
            "SELECT COUNT(*) as c FROM archived_memories"
        ).fetchone()["c"]
        assert remaining <= 5
        assert archived >= 1


# --- Semantic Memory Tests ---

class TestSemanticMemory:
    def test_learn_and_relate_concepts(self, mm):
        mm.learn_concept("Python", "编程语言")
        mm.learn_concept("asyncio", "Python异步库", parent_name="Python")
        mm.relate_concepts("asyncio", "Python", relation="is_part_of")

        graph = mm.explore_concept("asyncio")
        assert len(graph["hierarchy"]) >= 1
        assert len(graph["edges"]) >= 1

    def test_concept_vector_search(self, mm):
        mm.learn_concept("PostgreSQL", "关系型数据库")
        mm.learn_concept("MongoDB", "文档型数据库")
        mm.learn_concept("React", "前端框架")

        results = mm.retrieval.recall_concepts("数据库")
        assert len(results) > 0


# --- Procedural Memory Tests ---

class TestProceduralMemory:
    def test_learn_and_use_skill(self, mm):
        skill = mm.learn_skill(
            name="Git Rebase",
            description="将分支变基到主分支",
            steps=["git checkout feature", "git rebase main", "解决冲突", "git push --force-with-lease"],
            trigger_condition="需要同步主分支的最新改动",
        )
        assert skill.name == "Git Rebase"
        mm.use_skill(skill.id)
        updated = mm.db.get_skill(skill.id)
        assert updated["use_count"] == 1
        assert updated["proficiency"] > 0.2


# --- MemoryManager Integration Tests ---

class TestMemoryManagerIntegration:
    def test_full_workflow(self, mm):
        # 1. Agent perceives things
        mm.remember("用户问了一个关于数据库性能的问题", attention_weight=0.6)
        mm.remember("提到了慢查询和索引优化", attention_weight=0.7)
        mm.remember("用户用的 PostgreSQL 15", attention_weight=0.6)

        # 2. Session ends, consolidate
        mm.end_session()

        # 3. Agent learns a concept
        mm.learn_concept("数据库索引", "加速查询的数据结构")
        mm.learn_concept("PostgreSQL", "开源关系型数据库")

        # 4. Agent solves a problem and records it
        sol = mm.record_solution(
            problem_type="慢查询优化",
            problem_abstract="PostgreSQL 查询超过 5 秒",
            attempts=[
                SolutionAttempt(
                    approach="添加复合索引",
                    outcome=AttemptOutcome.SUCCESS,
                    quality_score=9.0,
                    is_best_known=True,
                    why_succeeded="索引覆盖了 WHERE + ORDER BY",
                ),
                SolutionAttempt(
                    approach="全表扫描调整参数",
                    outcome=AttemptOutcome.FAILED,
                    quality_score=2.0,
                    why_failed="work_mem 调大后反而更慢",
                ),
            ],
            dead_ends=[
                DeadEndRecord(
                    approach="盲目增大 work_mem",
                    failure_mode="内存竞争导致整体变慢",
                    failure_type=FailureType.CONFIG,
                    wasted_time_minutes=60,
                    lessons="先 EXPLAIN ANALYZE，再调参数",
                ),
            ],
        )
        assert sol is not None

        # 5. Later, agent encounters similar problem
        result = mm.solve_problem("数据库查询要 3 秒才返回")
        assert len(result["solutions"]) > 0

        # 6. Agent tries a risky approach
        warnings = mm.check_approach("我准备把所有内存参数调到最大")
        assert isinstance(warnings, list)

        # 7. Learn a skill from the experience
        mm.learn_skill(
            name="SQL 性能分析",
            description="标准性能排查流程",
            steps=["EXPLAIN ANALYZE", "检查索引使用", "分析统计信息", "优化查询或索引"],
        )

        # 8. Stats
        stats = mm.stats()
        assert stats["episodic_active"] >= 3
        assert stats["concepts"] >= 2
        assert stats["active_solutions"] >= 1

    def test_maintenance_does_not_crash(self, mm):
        mm.remember("test", attention_weight=0.5)
        mm.consolidate()
        mm.run_maintenance()
        # Should complete without error

    def test_background_maintenance(self, mm):
        mm.remember("test", attention_weight=0.5)
        mm.consolidate()
        mm.start_background_maintenance(interval_seconds=60)  # won't fire in test
        mm.stop_background_maintenance()
        # Should stop cleanly
