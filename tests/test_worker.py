"""GPU worker integration tests (require a working CuPy + CUDA 12 GPU)."""

from __future__ import annotations


import pytest

from cuda_kernel_opt.scoring import BenchmarkOutcome
from cuda_kernel_opt.taskgen import CATEGORIES, DIFFICULTIES, generate_task

pytestmark = pytest.mark.gpu

SWEEP = [
    (cat, diff, seed)
    for cat in CATEGORIES
    for diff in DIFFICULTIES
    for seed in range(4)
]


def test_ping(worker):
    r = worker.call({"cmd": "ping"})
    assert r["ok"] and r["sm_count"] > 0


def test_compile_failure_is_reported(worker):
    bad = 'extern "C" __global__ void solve(const float* x, float* out, int n) { this is not c++ }'
    kt = generate_task("elementwise", "easy", 0)
    r = worker.call({"cmd": "test", "spec": kt.worker_spec(), "kernel_src": bad})
    assert r["ok"] and r["compiled"] is False
    assert r["compile_log"]


def test_missing_entry_point_is_compile_failure(worker):
    src = 'extern "C" __global__ void not_solve(float* o){ o[0]=1.f; }'
    kt = generate_task("elementwise", "easy", 0)
    r = worker.call({"cmd": "test", "spec": kt.worker_spec(), "kernel_src": src})
    assert r["ok"] and r["compiled"] is False


@pytest.mark.parametrize("category,difficulty,seed", SWEEP)
def test_naive_kernel_is_correct(worker, category, difficulty, seed):
    kt = generate_task(category, difficulty, seed)
    r = worker.call({"cmd": "test", "spec": kt.worker_spec(), "kernel_src": kt.naive_src})
    assert r["ok"] and r["compiled"], r.get("compile_log")
    assert r["all_passed"], [c for c in r["cases"] if not c["passed"]]


@pytest.mark.parametrize("category,difficulty,seed", SWEEP)
def test_every_task_is_solvable(worker, category, difficulty, seed):
    """The generated known-good reference must be correct and clear the target."""
    kt = generate_task(category, difficulty, seed)
    r = worker.call(
        {
            "cmd": "validate",
            "spec": kt.worker_spec(),
            "optimized_src": kt.optimized_src,
            "baseline_src": kt.naive_src,
        }
    )
    assert r["ok"], r
    assert r["solvable"], (
        f"{kt.task_id} {kt.inefficiency}: speedup={r.get('speedup'):.2f} "
        f"target={kt.target_speedup} valid={r.get('valid_benchmark')} reason={r.get('reason')}"
    )


def test_dead_code_kernel_fails_correctness(worker):
    """A kernel that only writes out[0] (rest garbage) must fail the epsilon check."""
    kt = generate_task("elementwise", "medium", 1)
    cheat = f"""// launch: grid=n_div block=256
extern "C" __global__ void solve({kt.signature}) {{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i == 0) out[0] = x[0];      // do essentially nothing
}}
"""
    r = worker.call({"cmd": "test", "spec": kt.worker_spec(), "kernel_src": cheat})
    assert r["compiled"] and r["all_passed"] is False


def test_flat_kernel_benchmark_is_rejected(worker):
    """A kernel that does O(1) work is caught by the bandwidth floor / post-check."""
    kt = generate_task("elementwise", "medium", 1)
    cheat = f"""// launch: grid=blocks:1 block=32
extern "C" __global__ void solve({kt.signature}) {{
    if (threadIdx.x == 0) out[0] = x[0];
}}
"""
    r = worker.call(
        {"cmd": "benchmark", "spec": kt.worker_spec(), "kernel_src": cheat, "baseline_src": kt.naive_src}
    )
    assert r["compiled"]
    outcome = BenchmarkOutcome.from_response(r)
    assert not outcome.valid  # post-check and/or bandwidth floor reject it


def test_infinite_loop_kernel_times_out():
    """A hung kernel must not hang the harness: the parent kills the worker."""
    from cuda_kernel_opt.worker_client import WorkerClient, WorkerTimeout
    import asyncio

    kt = generate_task("elementwise", "easy", 0)
    # A very long (not literally infinite) loop: still blows past the 8s request
    # timeout so the parent kills the worker, but self-terminates eventually if a
    # kill were ever missed.
    hang = f"""// launch: grid=n_div block=256
extern "C" __global__ void solve({kt.signature}) {{
    float acc = 0.f;
    for (long long k = 0; k < 200000000000LL; ++k) acc += (float)k * 1e-30f;
    if (acc == 123456.789f) out[0] = acc;   // keep the loop from being elided
}}
"""

    async def run():
        wc = WorkerClient(request_timeout=8.0, start_timeout=60.0)
        try:
            with pytest.raises(WorkerTimeout):
                await wc.request(
                    {"cmd": "test", "spec": kt.worker_spec(), "kernel_src": hang}
                )
        finally:
            await wc.close()

    asyncio.run(run())
