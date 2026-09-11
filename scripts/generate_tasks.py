"""Generate a task set and keep only the instances that are provably solvable.

    python scripts/generate_tasks.py --per-combo 8 --out taskset.json

For each (category, difficulty, seed) it asks the GPU worker whether the built-in
reference solution is correct and clears the tier's speedup target on this
machine, and writes the surviving (category, difficulty, seed) triples plus the
measured reference speedup. Cheap pre-filter so unsolvable tasks never reach a
model. Results per triple are cached in ``scripts/_task_cache.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ENV_ROOT = Path(__file__).resolve().parent.parent
if str(_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENV_ROOT))

from cuda_kernel_opt.taskgen import CATEGORIES, DIFFICULTIES, generate_task  # noqa: E402
from cuda_kernel_opt.worker_client import SyncWorkerClient  # noqa: E402

CACHE_PATH = _ENV_ROOT / "scripts" / "_task_cache.json"


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-combo", type=int, default=8, help="seeds per (category, difficulty)")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    ap.add_argument("--difficulties", nargs="+", default=list(DIFFICULTIES), choices=DIFFICULTIES)
    ap.add_argument("--out", type=Path, default=_ENV_ROOT / "taskset.json")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    cache = {} if args.refresh else _load_cache()
    kept: list[dict] = []
    n_total = 0
    worker = SyncWorkerClient()
    try:
        for cat in args.categories:
            for diff in args.difficulties:
                for seed in range(args.per_combo):
                    n_total += 1
                    key = f"{cat}/{diff}/{seed}"
                    kt = generate_task(cat, diff, seed)
                    entry = cache.get(key)
                    if entry is None:
                        resp = worker.request(
                            {
                                "cmd": "validate",
                                "spec": kt.worker_spec(),
                                "optimized_src": kt.optimized_src,
                                "baseline_src": kt.naive_src,
                            }
                        )
                        entry = {
                            "solvable": bool(resp.get("solvable")),
                            "speedup": round(float(resp.get("speedup", 0.0)), 2),
                            "target": kt.target_speedup,
                            "inefficiency": kt.inefficiency,
                            "op": kt.op.name,
                            "reason": resp.get("reason", ""),
                        }
                        cache[key] = entry
                    status = "keep" if entry["solvable"] else "drop"
                    print(
                        f"  [{status}] {key:24s} {entry['inefficiency']:20s} "
                        f"speedup={entry['speedup']:8.2f}x target={entry['target']:g}x {entry['reason']}"
                    )
                    if entry["solvable"]:
                        kept.append(
                            {
                                "category": cat,
                                "difficulty": diff,
                                "seed": seed,
                                "reference_speedup": entry["speedup"],
                            }
                        )
    finally:
        worker.close()

    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
    args.out.write_text(json.dumps({"tasks": kept}, indent=2))
    print(f"\nkept {len(kept)}/{n_total} tasks -> {args.out}")
    print(f"cache -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
