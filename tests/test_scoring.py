"""Unit tests for the scoring gate and speedup->reward mapping (no GPU)."""

from __future__ import annotations

import pytest

from cuda_kernel_opt.scoring import (
    BenchmarkOutcome,
    RolloutScoreState,
    TARGET_SPEEDUP,
    bandwidth_floor_ms,
    speedup_to_score,
)


def _good(speedup: float) -> BenchmarkOutcome:
    return BenchmarkOutcome(
        compiled=True,
        speedup=speedup,
        post_check_passed=True,
        checksum_ok=True,
        finite=True,
        bandwidth_floor_ok=True,
        scaling_monotonic=True,
    )


_TARGETS = [t for tiers in TARGET_SPEEDUP.values() for t in tiers.values()]


@pytest.mark.parametrize("target", _TARGETS)
def test_speedup_curve(target):
    assert speedup_to_score(1.0, target) == 0.0
    assert speedup_to_score(0.5, target) == 0.0
    assert speedup_to_score(target, target) == pytest.approx(1.0)
    assert speedup_to_score(target * 5, target) == 1.0  # clamped
    mid = 1.0 + (target - 1.0) / 2
    assert speedup_to_score(mid, target) == pytest.approx(0.5)


def test_target_table_shape():
    assert set(TARGET_SPEEDUP) == {"elementwise", "reduction"}
    for tiers in TARGET_SPEEDUP.values():
        assert set(tiers) == {"easy", "medium", "hard"}
        assert all(t > 1.0 for t in tiers.values())


def test_valid_requires_every_check():
    assert _good(3.0).valid
    for field in (
        "post_check_passed",
        "checksum_ok",
        "finite",
        "bandwidth_floor_ok",
        "scaling_monotonic",
    ):
        o = _good(3.0)
        setattr(o, field, False)
        assert not o.valid, field
    assert not BenchmarkOutcome(compiled=False).valid


def test_hard_gate():
    s = RolloutScoreState(difficulty="easy", target_speedup=1.5)
    s.record_compile(True)

    # correct but never a valid benchmark -> 0
    s.record_correctness(True)
    assert s.gated_score == 0.0

    # a valid fast benchmark -> scored
    run = s.record_benchmark(_good(1.5))
    assert run == pytest.approx(1.0)
    assert s.gated_score == pytest.approx(1.0)
    assert s.num_valid_benchmarks == 1

    # correctness later fails -> gate slams to 0 despite the good benchmark
    s.record_correctness(False)
    assert s.gated_score == 0.0


def test_rejected_benchmark_counts_as_hack_flag():
    s = RolloutScoreState(difficulty="medium", target_speedup=2.0)
    s.record_compile(True)
    s.record_correctness(True)
    o = _good(50.0)
    o.post_check_passed = False  # looks fast, but output is wrong on fresh inputs
    run = s.record_benchmark(o)
    assert run == 0.0
    assert s.hack_flags == 1
    assert s.gated_score == 0.0
    assert s.reject_reasons == ["post_benchmark_correctness_failed"]


def test_best_score_is_monotone():
    s = RolloutScoreState(difficulty="medium", target_speedup=2.0)
    s.record_compile(True)
    s.record_correctness(True)
    s.record_benchmark(_good(1.5))  # score 0.5
    s.record_benchmark(_good(3.0))  # score 1.0
    s.record_benchmark(_good(1.2))  # score 0.2 -- must not lower best
    assert s.best_score == pytest.approx(1.0)
    assert s.best_speedup == pytest.approx(3.0)


def test_bandwidth_floor_monotonic_in_size():
    assert bandwidth_floor_ms(8_000_000) > bandwidth_floor_ms(2_000_000)
    assert bandwidth_floor_ms(0) == 0.0
