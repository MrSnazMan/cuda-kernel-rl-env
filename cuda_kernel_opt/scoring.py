"""Scoring: the hard correctness gate and the speedup-to-reward mapping.

Kept deliberately small and pure so it can be unit-tested without a GPU. All the
"did the kernel actually do the work" judgement lives in the GPU worker and is
handed to :func:`score_from_benchmark` as a already-decided ``valid`` flag plus
the measured speedup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DIFFICULTIES = ("easy", "medium", "hard")

# Speedup over the naive baseline that maps to a full reward of 1.0, keyed by
# [category][difficulty]. Calibrated on the RTX 3060 Laptop (see
# scripts/generate_tasks.py) so that:
#   * the generator's known-good reference solution clears the target with margin
#     (the solvability pre-filter needs reference_speedup >= target), and
#   * a partial fix lands mid-ramp rather than instantly at 1.0 -- the targets
#     sit well below the reference ceiling so weak kernels get partial credit.
# Reduction targets are much higher than elementwise because a per-thread-atomic
# reduction still has a real ~7-14x of headroom to a proper block reduction,
# whereas the elementwise inefficiencies here top out around 3-4x.
TARGET_SPEEDUP: dict[str, dict[str, float]] = {
    "elementwise": {"easy": 1.7, "medium": 1.7, "hard": 1.9},
    "reduction": {"easy": 3.0, "medium": 5.0, "hard": 6.0},
}

# Correctness tolerances per tier: pass iff
#   max|agent - ref| <= atol + rtol * max|ref|
# Reductions widen rtol (see ``reduction_rtol``) to absorb float32 accumulation
# error over ``n`` summands.
TOLERANCES: dict[str, dict[str, float]] = {
    "easy": {"atol": 1e-4, "rtol": 1e-4, "reduction_rtol": 3e-3},
    "medium": {"atol": 1e-4, "rtol": 1e-4, "reduction_rtol": 5e-3},
    "hard": {"atol": 2e-4, "rtol": 2e-4, "reduction_rtol": 1e-2},
}

# Anti-"compiled-away kernel" defenses, in order of strength:
#   1. the benchmark's own post-check re-verifies the agent output on fresh
#      random inputs + a checksum (you can't fake the result of work you skipped);
#   2. a *physical* lower bound on run time from memory bandwidth: a kernel that
#      truly moves N elements cannot beat ``bytes / peak_bw`` by much;
#   3. run time must not *decrease* as the input grows (monotonic).
# The old size-ratio heuristic was dropped: genuinely fast kernels have flat
# timing at these sizes and it produced false positives on honest solutions.
PEAK_BW_GBPS = 336.0          # RTX 3060 Laptop GDDR6 theoretical
BANDWIDTH_FLOOR_FRACTION = 0.18  # below 18% of peak BW => not actually moving the data


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def speedup_to_score(speedup: float, target: float) -> float:
    """Map a measured speedup to [0, 1], normalised against the task's target.

    0.0 at baseline speed (speedup == 1), 1.0 at ``target``, clamped above.
    Comparable across tasks and tiers by construction because ``target`` is
    itself calibrated per (category, tier) against the achievable ceiling.
    """
    if speedup <= 1.0:
        return 0.0
    return clamp((speedup - 1.0) / (target - 1.0))


def bandwidth_floor_ms(bytes_moved: int) -> float:
    """Fastest run time physically plausible for moving ``bytes_moved`` bytes."""
    return (bytes_moved / (PEAK_BW_GBPS * 1e9)) * 1e3 * BANDWIDTH_FLOOR_FRACTION


@dataclass
class BenchmarkOutcome:
    """Normalised view of one ``benchmark`` worker response, used for scoring."""

    compiled: bool
    speedup: float = 0.0
    post_check_passed: bool = False
    checksum_ok: bool = False
    finite: bool = False
    bandwidth_floor_ok: bool = True
    scaling_monotonic: bool = True
    scaling_ratio: float = 0.0
    scaling_baseline_ratio: float = 0.0

    @property
    def valid(self) -> bool:
        return (
            self.compiled
            and self.post_check_passed
            and self.checksum_ok
            and self.finite
            and self.bandwidth_floor_ok
            and self.scaling_monotonic
        )

    @classmethod
    def from_response(cls, resp: dict) -> "BenchmarkOutcome":
        if not resp.get("compiled"):
            return cls(compiled=False)
        pc = resp.get("post_check", {}) or {}
        sc = resp.get("scaling", {}) or {}
        return cls(
            compiled=True,
            speedup=float(resp.get("speedup", 0.0) or 0.0),
            post_check_passed=bool(pc.get("passed")),
            checksum_ok=bool(pc.get("checksum_ok")),
            finite=bool(pc.get("finite")),
            bandwidth_floor_ok=bool(pc.get("bandwidth_floor_ok", True)),
            scaling_monotonic=bool(sc.get("monotonic", True)),
            scaling_ratio=float(sc.get("agent_ratio", 0.0) or 0.0),
            scaling_baseline_ratio=float(sc.get("baseline_ratio", 0.0) or 0.0),
        )


def score_from_benchmark(outcome: BenchmarkOutcome, target: float) -> float:
    """Score for a single benchmark. 0 unless it is valid *and* faster."""
    if not outcome.valid:
        return 0.0
    return speedup_to_score(outcome.speedup, target)


@dataclass
class RolloutScoreState:
    """Accumulates the best honest result seen during a rollout."""

    difficulty: str
    target_speedup: float
    best_score: float = 0.0
    best_speedup: float = 0.0
    num_benchmarks: int = 0
    num_valid_benchmarks: int = 0
    num_compiles: int = 0
    compiled_ok_ever: bool = False
    correctness_passed: bool = False
    hack_flags: int = 0
    reject_reasons: list[str] = field(default_factory=list)

    def record_compile(self, ok: bool) -> None:
        self.num_compiles += 1
        self.compiled_ok_ever = self.compiled_ok_ever or ok

    def record_correctness(self, all_passed: bool) -> None:
        # Latest correctness result wins; the gate below also requires a valid
        # benchmark so a lucky earlier pass cannot carry a broken final kernel.
        self.correctness_passed = all_passed

    def record_benchmark(self, outcome: BenchmarkOutcome) -> float:
        self.num_benchmarks += 1
        if outcome.compiled and not outcome.valid:
            self.hack_flags += 1
            self.reject_reasons.append(_reject_reason(outcome))
        s = score_from_benchmark(outcome, self.target_speedup)
        if outcome.valid:
            self.num_valid_benchmarks += 1
            if s > self.best_score:
                self.best_score = s
            if outcome.speedup > self.best_speedup:
                self.best_speedup = outcome.speedup
        return s

    @property
    def gated_score(self) -> float:
        """Final reward: the hard gate applied to the best valid benchmark."""
        if not self.correctness_passed:
            return 0.0
        if self.num_valid_benchmarks == 0:
            return 0.0
        return self.best_score


def _reject_reason(o: BenchmarkOutcome) -> str:
    if not o.post_check_passed:
        return "post_benchmark_correctness_failed"
    if not o.checksum_ok:
        return "output_checksum_mismatch"
    if not o.finite:
        return "output_not_finite"
    if not o.bandwidth_floor_ok:
        return "run_time_below_memory_bandwidth_floor"
    if not o.scaling_monotonic:
        return "run_time_decreases_as_input_grows"
    return "benchmark_invalid"
