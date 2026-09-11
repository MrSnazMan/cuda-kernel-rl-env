"""Environment-level integration tests (require a GPU)."""

from __future__ import annotations

import json

import pytest

from cuda_kernel_opt.environment import load_environment, reward
from cuda_kernel_opt.taskgen import generate_task

pytestmark = pytest.mark.gpu


async def _mk_state(env, row):
    state = {"info": row["info"], "trajectory": []}
    await env.setup_state(state)
    return state


async def _call(env, state, name, **kw):
    state["trajectory"].append({"turn": len(state["trajectory"])})
    return await getattr(env, name)(state=state, **kw)


def test_load_environment_shape():
    env = load_environment(num_tasks_per_combo=2)
    assert len(env.dataset) == 2 * 6  # 2 categories x 3 tiers x 2
    names = {t.__name__ for t in env.tools}
    assert names == {
        "view_kernel",
        "submit_kernel",
        "test_correctness",
        "benchmark",
        "check_score",
    }


@pytest.mark.parametrize(
    "category,difficulty",
    [("elementwise", "easy"), ("elementwise", "medium"), ("reduction", "medium")],
)
def test_reference_solution_scores_full(category, difficulty):
    import asyncio

    async def run():
        env = load_environment(
            categories=(category,),
            difficulties=(difficulty,),
            num_tasks_per_combo=1,
            benchmark_budget=3,
        )
        row = env.dataset[0]
        info = json.loads(row["info"])
        kt = generate_task(info["category"], info["difficulty"], info["seed"])
        state = await _mk_state(env, row)
        try:
            await _call(env, state, "submit_kernel", source=kt.optimized_src)
            corr = await _call(env, state, "test_correctness")
            assert "CORRECTNESS: PASS" in corr
            # Score is derived from a measured speedup, so a single timed sample
            # of the reference kernel can land a hair under the tier target when
            # the GPU is busy (e.g. the rest of the suite running). Scoring keeps
            # the best benchmark, exactly as it does for a real agent, so take a
            # few samples within budget and require the best to clear the bar.
            for _ in range(3):
                bench = await _call(env, state, "benchmark")
                assert "valid benchmark" in bench
                if await reward(state) == pytest.approx(1.0):
                    break
            assert await reward(state) == pytest.approx(1.0)
        finally:
            await env._close_worker(state)

    asyncio.run(run())


def test_cheating_kernel_scores_zero():
    import asyncio

    async def run():
        env = load_environment(
            categories=("elementwise",),
            difficulties=("medium",),
            num_tasks_per_combo=1,
            benchmark_budget=3,
        )
        row = env.dataset[0]
        kt = generate_task(**json.loads(row["info"]))
        state = await _mk_state(env, row)
        cheat = f"""// launch: grid=n_div block=256
extern "C" __global__ void solve({kt.signature}) {{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i == 0) out[0] = x[0];
}}
"""
        try:
            await _call(env, state, "submit_kernel", source=cheat)
            corr = await _call(env, state, "test_correctness")
            assert "CORRECTNESS: FAIL" in corr
            await _call(env, state, "benchmark")
            assert await reward(state) == 0.0
        finally:
            await env._close_worker(state)

    asyncio.run(run())


def test_full_trajectory_budget_and_reject_paths():
    """Drive a realistic multi-turn trajectory through the real tools."""
    import asyncio

    async def run():
        env = load_environment(
            categories=("elementwise",),
            difficulties=("easy",),
            num_tasks_per_combo=1,
            benchmark_budget=2,
            max_turns=30,
        )
        row = env.dataset[0]
        kt = generate_task(**json.loads(row["info"]))
        state = await _mk_state(env, row)
        try:
            # 1. look at the task
            v = await _call(env, state, "view_kernel")
            assert "CUDA kernel optimisation task" in v and "Current kernel source" in v

            # 2. submit something broken -> compile fails, no score
            await _call(env, state, "submit_kernel", source='__global__ void solve() { @#$ }')
            r = await _call(env, state, "test_correctness")
            assert "COMPILE: FAILED" in r

            # 3. submit a fast cheat -> benchmark rejected, hack flag, score stays 0
            cheat = f"""// launch: grid=blocks:1 block=32
extern "C" __global__ void solve({kt.signature}) {{ if (threadIdx.x == 0) out[0] = x[0]; }}
"""
            await _call(env, state, "submit_kernel", source=cheat)
            r = await _call(env, state, "benchmark")
            assert "REJECTED" in r
            assert state["score_state"].hack_flags == 1
            assert await reward(state) == 0.0

            # 4. submit the real fix -> valid benchmark, full score
            await _call(env, state, "submit_kernel", source=kt.optimized_src)
            await _call(env, state, "test_correctness")
            r = await _call(env, state, "benchmark")
            assert "valid benchmark" in r
            assert await reward(state) == pytest.approx(1.0)

            # 5. budget now exhausted (2 calls used)
            r = await _call(env, state, "benchmark")
            assert "budget exhausted" in r

            r = await _call(env, state, "check_score")
            assert "best score        : 1.000" in r
        finally:
            await env._close_worker(state)

    asyncio.run(run())


def test_state_isolation_between_rollouts():
    """Two rollouts on one env instance must not share worker or kernel state."""
    import asyncio

    async def run():
        env = load_environment(
            categories=("elementwise",), difficulties=("easy",), num_tasks_per_combo=1
        )
        row = env.dataset[0]
        kt = generate_task(**json.loads(row["info"]))

        s1 = await _mk_state(env, row)
        await _call(env, s1, "submit_kernel", source=kt.optimized_src)
        assert s1["current_src"] == kt.optimized_src
        await env._close_worker(s1)

        s2 = await _mk_state(env, row)
        assert s2["current_src"] == kt.naive_src  # fresh start
        assert s2["worker"] is None
        assert s2["score_state"].best_score == 0.0
        await env._close_worker(s2)

    asyncio.run(run())
