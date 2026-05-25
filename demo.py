# -*- coding: utf-8 -*-
"""Demo script -- exercises the complete memory system pipeline."""

import os, sys, io

# Fix GBK encoding on Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from human_memory import MemoryManager
from human_memory.models import SolutionAttempt, DeadEndRecord, AttemptOutcome, FailureType

DB_PATH = "demo_memory.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

from human_memory.config import MemoryConfig

cfg = MemoryConfig(
    db_path=DB_PATH,
    episodic_capacity=5,    # small to demonstrate capacity eviction
    eviction_batch_size=5,
)
mm = MemoryManager(config=cfg)

# =========================================
# Phase 1: simulation of a chat session
# =========================================
print("=== Phase 1: Perception into Working Memory ===")
mm.remember("user is debugging a Python asyncio performance issue", attention_weight=0.7)
mm.remember("error log: 'Task was destroyed but it is pending'", attention_weight=0.8)
mm.remember("user already tried increasing timeout, didn't help", attention_weight=0.5)

print(f"  WM slots occupied: {len(mm.wm)}")
for s in mm.wm.get_attention_sorted():
    print(f"    [attn={s.attention_weight:.1f}] {s.content[:60]}")

# =========================================
# Phase 2: consolidate to LTM
# =========================================
print("\n=== Phase 2: Consolidate WM -> LTM ===")
encoded, discarded = mm.consolidate()
print(f"  Encoded: {len(encoded)} episodes")
print(f"  Discarded (below threshold): {len(discarded)} items")
for ep in encoded:
    print(f"    Depth L{ep.encoding_depth} [strength={ep.strength:.2f}] {ep.summary_text[:60]}")

# =========================================
# Phase 3: semantic memory
# =========================================
print("\n=== Phase 3: Build Semantic Memory ===")
mm.learn_concept("Python", "general-purpose programming language")
mm.learn_concept("asyncio", "Python async I/O library", parent_name="Python")
mm.learn_concept("event loop", "asyncio core scheduling mechanism", parent_name="asyncio")
mm.relate_concepts("event loop", "asyncio", relation="is_part_of")
print(f"  Concepts stored: {mm.stats()['concepts']}")

# =========================================
# Phase 4: solve problem, record solution
# =========================================
print("\n=== Phase 4: Record Solution Memory ===")
sol = mm.record_solution(
    problem_type="Python async task leak",
    problem_abstract="asyncio.create_task returns a Task that is never awaited or kept, GC collects it while pending",
    attempts=[
        SolutionAttempt(
            approach="Keep Task references in a list; in cleanup, gather all pending tasks",
            outcome=AttemptOutcome.SUCCESS,
            quality_score=9.0,
            is_best_known=True,
            why_succeeded="Task holds a strong ref; GC won't collect; gather ensures all complete",
        ),
        SolutionAttempt(
            approach="Increase timeout + set _log_destroy_pending=False on asyncio.Task",
            outcome=AttemptOutcome.FAILED,
            quality_score=2.0,
            is_worst_known=True,
            why_failed="Only suppressed the log, didn't fix the leak; tasks still accumulate",
        ),
    ],
    dead_ends=[
        DeadEndRecord(
            approach="blindly increasing timeout parameters",
            failure_mode="symptoms disappeared but leak continued, memory grew until OOM",
            failure_type=FailureType.CONFIG,
            wasted_time_minutes=45,
            lessons="Silence is not a fix. Task leaks must be traced by reference chain.",
            searchable_symptoms='["Task was destroyed", "pending", "memory creeping up"]',
        ),
    ],
    context_tags=["python", "asyncio", "task-leak"],
)
print(f"  Solution '{sol.problem_type}' recorded")
print(f"  Total attempts: {sol.trial_count}, failed: {sol.failed_count}")
print(f"  Best approach: {sol.best_approach[:60]}...")

# =========================================
# Phase 5: retrieval
# =========================================
print("\n=== Phase 5: Retrieval ===")
print("  Searching for 'async task error':")
for m in mm.recall("async task error"):
    print(f"    [strength={m.get('strength', 0):.2f}] {m.get('summary_text', '')[:60]}")

# =========================================
# Phase 6: dead-end check + solution recommendation
# =========================================
print("\n=== Phase 6: Dead-end Check + Solution Recommendation ===")
result = mm.solve_problem(
    problem_description="asyncio task gives 'destroyed but pending' warning",
    proposed_approach="let me just crank all timeout params to max",
)
if result["warnings"]:
    print("  [DEAD-END WARNING]:")
    for w in result["warnings"]:
        print(f"    Matched dead end in '{w['problem_type']}': {w['failure_mode']}")
        print(f"    Wasted time in past: {w['wasted_time_minutes']} min")
        print(f"    Lesson: {w['lessons']}")
else:
    print("  [OK] No known dead ends matched")

if result["solutions"]:
    print(f"\n  [SOLUTIONS FOUND] ({len(result['solutions'])}):")
    for s in result["solutions"]:
        print(f"    '{s['problem_type']}' -> {s.get('best_approach', '')[:60]}")
        if s.get("worst_approach"):
            print(f"    AVOID: {s['worst_approach'][:60]}")
        if s.get("total_wasted_time", 0) > 0:
            print(f"    (total wasted time on this problem: {s['total_wasted_time']} min)")

print(f"\n  Recommendation: {result['recommendation']}")

# =========================================
# Phase 7: capacity pressure -> forgetting
# =========================================
print(f"\n=== Phase 7: Capacity Pressure -> Forgetting (capacity={mm.config.episodic_capacity}) ===")
for i in range(12):
    mm.remember(f"filler memory {i} -- background noise", attention_weight=0.6, emotional_intensity=0.3)
mm.consolidate()
mm.run_maintenance()
stats = mm.stats()
print(f"  Active episodes: {stats['episodic_active']} | Forgotten: {stats['episodic_forgotten']}")
print(f"  Depth distribution: L1={stats['encoding_depth']['L1_shallow']}, "
      f"L2={stats['encoding_depth']['L2_standard']}, L3={stats['encoding_depth']['L3_deep']}")

# =========================================
# Phase 8: full stats
# =========================================
print("\n=== Phase 8: System Stats ===")
for k, v in mm.stats().items():
    print(f"  {k}: {v}")

mm.close()
os.remove(DB_PATH)
print("\nDone.")
