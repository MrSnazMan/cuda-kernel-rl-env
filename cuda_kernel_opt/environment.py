"""The CUDA kernel optimisation environment.

A ``StatefulToolEnv``: each rollout owns an in-memory ``KernelTask``, the current
kernel source, a turn/benchmark budget, and a GPU worker subprocess. All GPU work
goes through that worker so state resets cleanly between rollouts.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import verifiers as vf

from .scoring import BenchmarkOutcome, RolloutScoreState
from .taskgen import (
    CATEGORIES,
    DIFFICULTIES,
    SYSTEM_PROMPT,
    KernelTask,
    build_dataset,
    generate_task,
)
from .worker_client import WorkerClient, WorkerError

_ENTRY_TOKEN = "__global__"


class CudaKernelOptEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        *,
        dataset,
        rubric: vf.Rubric,
        max_turns: int = 20,
        benchmark_budget: int = 12,
        worker_timeout: float = 45.0,
        max_concurrent_gpu: int = 1,
        max_live_workers: int = 4,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=max_turns,
            system_prompt=SYSTEM_PROMPT,
            **kwargs,
        )
        self.benchmark_budget = benchmark_budget
        self.worker_timeout = worker_timeout
        self._gpu_sem = asyncio.Semaphore(max(1, max_concurrent_gpu))
        self._worker_slots = asyncio.Semaphore(max(1, max_live_workers))

        self.add_tool(self.view_kernel, args_to_skip=["state"])
        self.add_tool(self.submit_kernel, args_to_skip=["state"])
        self.add_tool(self.test_correctness, args_to_skip=["state"])
        self.add_tool(self.benchmark, args_to_skip=["state"])
        self.add_tool(self.check_score, args_to_skip=["state"])

    # -- lifecycle ---------------------------------------------------------
    async def setup_state(self, state: vf.State) -> None:
        info = state.get("info") or {}
        if isinstance(info, str):
            info = json.loads(info)
        kt = generate_task(info["category"], info["difficulty"], int(info["seed"]))
        state["kt"] = kt
        state["current_src"] = kt.naive_src
        state["score_state"] = RolloutScoreState(
            difficulty=kt.difficulty, target_speedup=kt.target_speedup
        )
        state["benchmark_used"] = 0
        state["worker"] = None
        state["worker_slot_held"] = False
        state["last_correctness"] = None
        await super().setup_state(state)

    @vf.cleanup
    async def _close_worker(self, state: vf.State) -> None:
        worker: WorkerClient | None = state.get("worker")
        state["worker"] = None
        if worker is not None:
            try:
                await worker.close()
            finally:
                if state.get("worker_slot_held"):
                    state["worker_slot_held"] = False
                    self._worker_slots.release()

    def update_tool_args(
        self, tool_name: str, tool_args: dict, messages: vf.Messages, state: vf.State, **kwargs
    ) -> dict:
        tool_args["state"] = state
        return tool_args

    # -- helpers ---------------------------------------------------------
    def _footer(self, state: vf.State) -> str:
        turns_left = self.max_turns - len(state["trajectory"])
        bench_left = self.benchmark_budget - state["benchmark_used"]
        return f"\n[turns remaining: {turns_left} | benchmark calls left: {bench_left}]"

    async def _worker_request(self, state: vf.State, req: dict) -> dict:
        if state.get("worker") is None:
            await self._worker_slots.acquire()
            state["worker_slot_held"] = True
            state["worker"] = WorkerClient(request_timeout=self.worker_timeout)
        worker: WorkerClient = state["worker"]
        async with self._gpu_sem:
            return await worker.request(req, timeout=self.worker_timeout)

    # -- tools ---------------------------------------------------------
    async def view_kernel(self, state: vf.State) -> str:
        """Show the current kernel source together with the full task specification."""
        kt: KernelTask = state["kt"]
        body = kt.describe(include_source=False)
        return (
            f"{body}\n\nCurrent kernel source:\n```cuda\n{state['current_src'].strip()}\n```"
            + self._footer(state)
        )

    async def submit_kernel(self, state: vf.State, source: str) -> str:
        """Replace the current kernel with an edited version.

        Args:
            source: The complete CUDA C source for the kernel, including the
                `// launch:` directive and the exact `extern "C" __global__ void
                solve(...)` entry point. This does not compile or test it.
        """
        if _ENTRY_TOKEN not in source or "solve" not in source:
            return (
                "Rejected: the source must define `extern \"C\" __global__ void solve(...)` "
                "with the required signature. Kernel unchanged." + self._footer(state)
            )
        state["current_src"] = source
        n_lines = source.count("\n") + 1
        return (
            f"Stored new kernel ({n_lines} lines). Run test_correctness, then benchmark."
            + self._footer(state)
        )

    async def test_correctness(self, state: vf.State) -> str:
        """Compile the current kernel and check it against several fresh random inputs."""
        kt: KernelTask = state["kt"]
        try:
            resp = await self._worker_request(
                state, {"cmd": "test", "spec": kt.worker_spec(), "kernel_src": state["current_src"]}
            )
        except WorkerError as e:
            return f"GPU worker error: {e}" + self._footer(state)

        score_state: RolloutScoreState = state["score_state"]
        if not resp.get("compiled"):
            score_state.record_compile(False)
            state["last_correctness"] = {"compiled": False}
            return (
                "COMPILE: FAILED\n" + _truncate(resp.get("compile_log", ""), 2500) + self._footer(state)
            )
        score_state.record_compile(True)
        cases = resp["cases"]
        all_passed = resp["all_passed"]
        score_state.record_correctness(all_passed)
        state["last_correctness"] = {"compiled": True, "all_passed": all_passed, "n_cases": len(cases)}

        lines = [
            "COMPILE: ok",
            f"CORRECTNESS: {'PASS' if all_passed else 'FAIL'} ({sum(c['passed'] for c in cases)}/{len(cases)} cases)",
        ]
        for c in cases:
            lines.append(
                f"  n={c['n']:<9d} max_abs_err={c['max_abs_err']:.3e}  "
                f"threshold={c['threshold']:.3e}  {'ok' if c['passed'] else 'FAIL'}"
            )
        if not all_passed:
            lines.append(
                "\nThe kernel is wrong on at least one case. Fast-but-wrong scores 0."
            )
        return "\n".join(lines) + self._footer(state)

    async def benchmark(self, state: vf.State) -> str:
        """Time the current kernel against the naive baseline on fresh inputs.

        Limited number of calls per task. Also re-verifies the kernel output,
        checksums it, and checks that run time scales with input size; a
        benchmark that fails any of those is rejected and cannot raise the score.
        """
        if state["benchmark_used"] >= self.benchmark_budget:
            return (
                "benchmark budget exhausted for this task. Use check_score / finish."
                + self._footer(state)
            )
        state["benchmark_used"] += 1
        kt: KernelTask = state["kt"]
        try:
            resp = await self._worker_request(
                state,
                {
                    "cmd": "benchmark",
                    "spec": kt.worker_spec(),
                    "kernel_src": state["current_src"],
                    "baseline_src": kt.naive_src,
                },
            )
        except WorkerError as e:
            return f"GPU worker error: {e}" + self._footer(state)

        score_state: RolloutScoreState = state["score_state"]
        if not resp.get("compiled"):
            score_state.record_compile(False)
            return "COMPILE: FAILED\n" + _truncate(resp.get("compile_log", ""), 2500) + self._footer(state)
        score_state.record_compile(True)
        if resp.get("launch_error"):
            return f"LAUNCH ERROR: {resp['launch_error']}" + self._footer(state)

        outcome = BenchmarkOutcome.from_response(resp)
        run_score = score_state.record_benchmark(outcome)

        agent, base = resp["agent"], resp["baseline"]
        lc = resp["launch_config"]
        pc = resp["post_check"]
        sc = resp["scaling"]
        lines = [
            "COMPILE: ok",
            f"TIMING (median of {resp['timed_iters']} trials):",
            f"  your kernel : {agent['median_ms']:.4f} ms   "
            f"(grid={lc['agent']['grid']} block={lc['agent']['block']} {lc['agent']['mode']})",
            f"  naive base  : {base['median_ms']:.4f} ms   "
            f"(grid={lc['baseline']['grid']} block={lc['baseline']['block']} {lc['baseline']['mode']})",
            f"  speedup     : {resp['speedup']:.2f}x",
            "VALIDATION:",
            f"  post-benchmark correctness : {'PASS' if pc['passed'] else 'FAIL'}  "
            f"(max_abs_err {pc['max_abs_err']:.3e} vs threshold {pc['threshold']:.3e})",
            f"  output checksum            : {'ok' if pc['checksum_ok'] else 'MISMATCH'}",
            f"  finite outputs             : {'ok' if pc['finite'] else 'NON-FINITE'}",
            f"  run-time >= bandwidth floor: {'ok' if pc.get('bandwidth_floor_ok') else 'BELOW FLOOR'}  "
            f"({agent['median_ms']:.4f} ms vs {pc.get('bandwidth_floor_ms', 0):.4f} ms)",
            f"  time grows with input size : {'ok' if sc['monotonic'] else 'FAIL'}  "
            f"(n/4 -> n : {sc['agent_ms'][0]:.4f} -> {sc['agent_ms'][1]:.4f} ms)",
        ]
        if outcome.valid:
            lines.append(
                f"RESULT: valid benchmark. score this run = {run_score:.3f}  "
                f"(speedup {resp['speedup']:.2f}x vs target {kt.target_speedup:g}x)"
            )
        else:
            lines.append(
                f"RESULT: benchmark REJECTED ({score_state.reject_reasons[-1]}). "
                f"score unchanged."
            )
        lines.append(f"best score so far: {score_state.best_score:.3f}")
        return "\n".join(lines) + self._footer(state)

    async def check_score(self, state: vf.State) -> str:
        """Report the best score, best speedup and remaining budget for this task."""
        s: RolloutScoreState = state["score_state"]
        kt: KernelTask = state["kt"]
        lc = state.get("last_correctness")
        if not lc:
            corr = "not run yet"
        elif not lc.get("compiled"):
            corr = "last compile FAILED"
        else:
            corr = f"{'PASS' if lc['all_passed'] else 'FAIL'} ({lc['n_cases']} cases)"
        return "\n".join(
            [
                f"best score        : {s.gated_score:.3f}",
                f"best speedup      : {s.best_speedup:.2f}x  (target {kt.target_speedup:g}x for {kt.difficulty})",
                f"last correctness  : {corr}",
                f"valid benchmarks  : {s.num_valid_benchmarks} / {state['benchmark_used']} used"
                f" (rejected: {s.hack_flags})",
            ]
        ) + self._footer(state)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------
async def reward(state: vf.State) -> float:
    return float(state["score_state"].gated_score)


async def best_speedup(state: vf.State) -> float:
    return float(state["score_state"].best_speedup)


async def correctness_passed(state: vf.State) -> float:
    return 1.0 if state["score_state"].correctness_passed else 0.0


async def valid_benchmarks(state: vf.State) -> float:
    return float(state["score_state"].num_valid_benchmarks)


async def num_benchmarks(state: vf.State) -> float:
    return float(state["score_state"].num_benchmarks)


async def num_compiles(state: vf.State) -> float:
    return float(state["score_state"].num_compiles)


async def compiled_ok(state: vf.State) -> float:
    return 1.0 if state["score_state"].compiled_ok_ever else 0.0


async def rejected_benchmarks(state: vf.State) -> float:
    return float(state["score_state"].hack_flags)


def _build_rubric() -> vf.Rubric:
    rubric = vf.Rubric(funcs=[reward], weights=[1.0])
    for fn in (
        best_speedup,
        correctness_passed,
        valid_benchmarks,
        num_benchmarks,
        num_compiles,
        compiled_ok,
        rejected_benchmarks,
    ):
        rubric.add_metric(fn)
    return rubric


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_environment(
    categories: tuple[str, ...] | list[str] = CATEGORIES,
    difficulties: tuple[str, ...] | list[str] = DIFFICULTIES,
    num_tasks_per_combo: int = 2,
    seed: int = 0,
    max_turns: int = 20,
    benchmark_budget: int = 12,
    worker_timeout: float = 45.0,
    max_concurrent_gpu: int = 1,
    max_live_workers: int = 4,
    prevalidate: bool = False,
    **kwargs: Any,
) -> vf.Environment:
    """Build the CUDA-kernel-optimisation environment.

    Args:
        categories: kernel categories to include (`elementwise`, `reduction`).
        difficulties: tiers to include (`easy`, `medium`, `hard`).
        num_tasks_per_combo: task instances per (category, difficulty).
        seed: base seed; instance k uses seed + k.
        max_turns: assistant turns per rollout.
        benchmark_budget: `benchmark` tool calls per rollout.
        worker_timeout: per GPU-request wall-clock limit (kills hung kernels).
        max_concurrent_gpu: GPU requests allowed to run at once across rollouts.
        max_live_workers: cap on simultaneously live GPU worker subprocesses.
        prevalidate: if True, drop tasks whose known-good reference solution does
            not actually clear the tier's speedup target on this machine.
    """
    from datasets import Dataset

    categories = tuple(categories)
    difficulties = tuple(difficulties)
    rows = build_dataset(categories, difficulties, num_tasks_per_combo, seed)

    if prevalidate:
        rows = _prevalidate_rows(rows, worker_timeout)

    dataset = Dataset.from_list(rows)
    rubric = _build_rubric()
    return CudaKernelOptEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        benchmark_budget=benchmark_budget,
        worker_timeout=worker_timeout,
        max_concurrent_gpu=max_concurrent_gpu,
        max_live_workers=max_live_workers,
        **kwargs,
    )


def _prevalidate_rows(rows: list[dict], worker_timeout: float) -> list[dict]:
    """Synchronously filter tasks to those with a demonstrable fast+correct solution."""
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "cuda_kernel_opt.gpu_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin and proc.stdout
    kept: list[dict] = []
    try:
        for row in rows:
            info = json.loads(row["info"])
            kt = generate_task(info["category"], info["difficulty"], int(info["seed"]))
            req = {
                "cmd": "validate",
                "spec": kt.worker_spec(),
                "optimized_src": kt.optimized_src,
                "baseline_src": kt.naive_src,
            }
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())
            if resp.get("solvable"):
                kept.append(row)
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    if not kept:
        raise RuntimeError("prevalidate dropped every task; check the GPU worker")
    return kept
