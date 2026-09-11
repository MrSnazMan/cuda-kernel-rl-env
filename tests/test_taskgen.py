"""Unit tests for task generation — no GPU required."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from cuda_kernel_opt import taskgen as tg
from cuda_kernel_opt.taskgen import (
    CATEGORIES,
    DIFFICULTIES,
    generate_task,
    parse_launch_directive,
)

ALL_COMBOS = [
    (cat, diff, seed)
    for cat in CATEGORIES
    for diff in DIFFICULTIES
    for seed in range(6)
]


@pytest.mark.parametrize("category,difficulty,seed", ALL_COMBOS)
def test_task_shape(category, difficulty, seed):
    kt = generate_task(category, difficulty, seed)
    assert kt.task_id == f"{category}/{difficulty}/{seed}"
    assert kt.size_min <= kt.size <= kt.size_max
    assert (kt.size & (kt.size - 1)) == 0  # power of two
    assert kt.target_speedup > 1.0
    assert kt.atol > 0 and kt.rtol > 0
    for src in (kt.naive_src, kt.optimized_src):
        assert 'extern "C" __global__ void solve(' in src
        assert kt.signature in src
        cfg = parse_launch_directive(src)  # must parse + validate
        cfg.validate()


@pytest.mark.parametrize("category,difficulty,seed", ALL_COMBOS)
def test_deterministic(category, difficulty, seed):
    a = generate_task(category, difficulty, seed)
    b = generate_task(category, difficulty, seed)
    assert a.naive_src == b.naive_src
    assert a.optimized_src == b.optimized_src
    assert a.op == b.op


def _fingerprint_in_subprocess(hashseed: str) -> str:
    """Fingerprint every task in a fresh interpreter with a chosen PYTHONHASHSEED."""
    code = textwrap.dedent(
        """
        from cuda_kernel_opt.taskgen import CATEGORIES, DIFFICULTIES, generate_task
        out = []
        for cat in CATEGORIES:
            for diff in DIFFICULTIES:
                for seed in range(6):
                    kt = generate_task(cat, diff, seed)
                    out.append(f"{kt.task_id}|{kt.inefficiency}|{kt.op.name}|{kt.size}|{kt.naive_src}")
        print(chr(10).join(out))
        """
    )
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    return res.stdout


def test_deterministic_across_processes():
    """The (category, difficulty, seed) triple must map to the same task in any
    process. Regression test for the ``hash()``-based seed, which varied with
    PYTHONHASHSEED and desynced the eval prompt from the graded reference."""
    a = _fingerprint_in_subprocess("0")
    b = _fingerprint_in_subprocess("1")
    c = _fingerprint_in_subprocess("random")
    assert a == b == c


def test_directive_parser():
    assert parse_launch_directive("// launch: grid=stride block=256").grid_mode == "stride"
    c = parse_launch_directive("//launch:grid=blocks:8 block=128")
    assert c.grid_mode == "blocks" and c.grid_blocks == 8 and c.block == 128
    assert parse_launch_directive("no directive here").grid_mode == "n_div"  # default
    with pytest.raises(ValueError):
        parse_launch_directive("// launch: grid=stride block=100").validate()
    with pytest.raises(ValueError):
        parse_launch_directive("// launch: grid=stride block=2048").validate()


def test_grid_resolution():
    c = tg.LaunchConfig("n_div", 256)
    assert c.resolve_grid(1000, 30) == math.ceil(1000 / 256)
    c = tg.LaunchConfig("blocks", 64, 4)
    assert c.resolve_grid(10**9, 30) == 4
    c = tg.LaunchConfig("stride", 256)
    assert c.resolve_grid(10**9, 30) == tg.STRIDE_MULT * 30
    assert c.resolve_grid(512, 30) == 2  # capped at one thread per element


@pytest.mark.parametrize("category,difficulty,seed", ALL_COMBOS)
def test_reference_matches_numpy(category, difficulty, seed):
    """reference_impl must agree with a hand-rolled float64 numpy computation."""
    kt = generate_task(category, difficulty, seed)
    rng = np.random.default_rng(seed)
    n = 4096
    xs = [rng.uniform(kt.input_lo, kt.input_hi, n).astype(np.float32)]
    if kt.n_inputs == 2:
        xs.append(rng.uniform(kt.input_lo, kt.input_hi, n).astype(np.float32))
    ref = tg.reference_impl(kt.reference_spec(), np, xs)

    x = xs[0].astype(np.float64)
    op = kt.op
    if category == "reduction":
        if op.reduce_kind == "sumsq":
            want = float(np.sum(x * x))
        else:
            want = float(np.sum(x))
        if op.reduce_kind == "mean":
            want /= n
        assert abs(ref - want) <= 1e-6 * max(1.0, abs(want))
    else:
        y = xs[1].astype(np.float64) if kt.n_inputs == 2 else None
        table = {
            "saxpy": lambda: op.a * x + op.b,
            "square": lambda: op.a * x * x + op.b * x + op.c,
            "gelu_tanh": lambda: 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3))),
            "scale_clamp": lambda: np.minimum(np.maximum(op.a * x, 0.0), op.b),
            "axpby": lambda: op.a * x + op.b * y,
            "muladd": lambda: x * y + op.a * x,
            "poly": lambda: _poly(x, op.poly),
        }
        want = table[op.name]()
        assert np.max(np.abs(ref - want)) <= 1e-9 * (1 + np.max(np.abs(want)))


def _poly(x, coeffs):
    acc = np.zeros_like(x)
    for k in coeffs:
        acc = acc * x + k
    return acc


def test_build_dataset():
    rows = tg.build_dataset(num_per_combo=2, seed0=0)
    assert len(rows) == len(CATEGORIES) * len(DIFFICULTIES) * 2
    for r in rows:
        assert r["prompt"][0]["role"] == "user"
        info = json.loads(r["info"])
        assert set(info) == {"category", "difficulty", "seed"}
        # round-trips back to the same task
        generate_task(**info)
