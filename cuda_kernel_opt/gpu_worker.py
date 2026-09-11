"""GPU worker subprocess: the only place CUDA is touched.

Runs as ``python -m cuda_kernel_opt.gpu_worker``. Speaks newline-delimited JSON on
stdin/stdout: one request object in, one response object out, flushed. Every
request is fully self-contained (sources, seeds, sizes) so the worker keeps no
computation state between calls — fresh device buffers and freshly regenerated
inputs each time. This is the state-isolation boundary: the parent kills and
respawns this process between rollouts and on any hang.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

# Must happen before cupy is imported: CuPy's path finder would otherwise pick up
# the system CUDA 13.x toolkit via these vars and load an NVRTC the installed
# driver cannot JIT. We want the pip-installed CUDA 12.x NVRTC instead.
for _v in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_PATH_V13_3"):
    os.environ.pop(_v, None)
warnings.filterwarnings("ignore", message="CUDA path could not be detected")

import numpy as np  # noqa: E402

try:
    import cupy as cp  # noqa: E402

    _CUPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    cp = None
    _CUPY_IMPORT_ERROR = repr(exc)

from cuda_kernel_opt.scoring import bandwidth_floor_ms  # noqa: E402
from cuda_kernel_opt.taskgen import parse_launch_directive, reference_impl  # noqa: E402

COMPILE_OPTIONS = ("--std=c++17", "--gpu-architecture=compute_86")
WARMUP = 3
TIMED = 25
CORRECTNESS_CASES = 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sm_count() -> int:
    return int(cp.cuda.Device(0).attributes["MultiProcessorCount"])


def _make_inputs(spec: dict, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    lo, hi = spec["input_lo"], spec["input_hi"]
    arrs = [rng.uniform(lo, hi, n).astype(np.float32)]
    if spec["n_inputs"] == 2:
        arrs.append(rng.uniform(lo, hi, n).astype(np.float32))
    return [cp.asarray(a) for a in arrs]


class CompileError(Exception):
    pass


def _compile(src: str):
    try:
        cfg = parse_launch_directive(src)
    except ValueError as e:
        raise CompileError(f"launch directive error: {e}") from e
    try:
        mod = cp.RawModule(code=src, backend="nvrtc", options=COMPILE_OPTIONS)
        fn = mod.get_function("solve")
    except cp.cuda.compiler.CompileException as e:  # type: ignore[attr-defined]
        raise CompileError(str(e)) from e
    except Exception as e:  # get_function raises generic on missing symbol
        raise CompileError(f"could not load kernel 'solve': {e}") from e
    return fn, cfg


def _launch(fn, cfg, arrays: list, out, n: int, sm: int) -> None:
    grid = cfg.resolve_grid(n, sm)
    args = (*arrays, out, np.int32(n))
    fn((grid,), (cfg.block,), args)


def _time_kernel(fn, cfg, arrays, out, n, sm, warmup, timed, is_reduction) -> list[float]:
    for _ in range(warmup):
        if is_reduction:
            out.fill(0)
        _launch(fn, cfg, arrays, out, n, sm)
    cp.cuda.Device().synchronize()
    samples: list[float] = []
    for _ in range(timed):
        if is_reduction:
            out.fill(0)
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        _launch(fn, cfg, arrays, out, n, sm)
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)))
    return samples


def _stats(samples: list[float]) -> dict:
    s = sorted(samples)
    k = max(0, len(s) // 5)
    trimmed = s[k : len(s) - k] or s
    n = len(s)
    median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {
        "median_ms": median,
        "trimmed_mean_ms": sum(trimmed) / len(trimmed),
        "min_ms": s[0],
        "raw_ms": [round(v, 5) for v in samples],
    }


def _adaptive_timed(first_ms: float) -> int:
    if first_ms > 50:
        return 5
    if first_ms > 5:
        return 12
    return TIMED


def _run_out(n: int, is_reduction: bool):
    return cp.zeros(1 if is_reduction else n, dtype=cp.float32)


def _correctness_one(fn, cfg, spec, n, seed, sm, atol, rtol, is_reduction) -> dict:
    arrays = _make_inputs(spec, n, seed)
    out = _run_out(n, is_reduction)
    if is_reduction:
        out.fill(0)
    _launch(fn, cfg, arrays, out, n, sm)
    cp.cuda.Device().synchronize()
    ref = reference_impl(spec, cp, arrays)
    if is_reduction:
        agent_val = float(out[0])
        ref_val = float(ref)
        abs_err = abs(agent_val - ref_val)
        thresh = atol + rtol * abs(ref_val)
        return {
            "n": n,
            "seed": seed,
            "max_abs_err": abs_err,
            "max_rel_err": abs_err / (abs(ref_val) + 1e-30),
            "threshold": thresh,
            "passed": abs_err <= thresh,
        }
    diff = cp.abs(out.astype(cp.float64) - ref)
    max_abs = float(diff.max())
    ref_mag = float(cp.abs(ref).max())
    thresh = atol + rtol * ref_mag
    rel = float((diff / (cp.abs(ref) + 1e-30)).max())
    return {
        "n": n,
        "seed": seed,
        "max_abs_err": max_abs,
        "max_rel_err": rel,
        "threshold": thresh,
        "passed": max_abs <= thresh,
    }


def _checksum(arr) -> float:
    return float(cp.abs(arr.astype(cp.float64)).sum())


# ---------------------------------------------------------------------------
# command handlers
# ---------------------------------------------------------------------------
def handle_ping(_req: dict) -> dict:
    return {"ok": True, "cupy": cp.__version__, "sm_count": _sm_count()}


def handle_test(req: dict) -> dict:
    spec = req["spec"]
    is_reduction = spec["kind"] == "reduction"
    try:
        fn, cfg = _compile(req["kernel_src"])
    except CompileError as e:
        return {"ok": True, "compiled": False, "compile_log": str(e)[:4000]}
    sm = _sm_count()
    atol, rtol = spec["atol"], spec["rtol"]
    sizes = _spread_sizes(spec, req.get("n_cases", CORRECTNESS_CASES))
    cases = []
    for i, n in enumerate(sizes):
        cases.append(
            _correctness_one(
                fn, cfg, spec, n, spec["seed"] + 1000 + i, sm, atol, rtol, is_reduction
            )
        )
    return {
        "ok": True,
        "compiled": True,
        "cases": cases,
        "all_passed": all(c["passed"] for c in cases),
    }


def _spread_sizes(spec: dict, k: int) -> list[int]:
    """k sizes spanning the tier range; when the range is a single power of two,
    k repeats of it (each correctness case still gets fresh random inputs)."""
    k = max(1, k)
    lo, hi = spec["size_min"], spec["size_max"]
    if lo == hi:
        return [spec["size"]] * k
    lo_log, hi_log = lo.bit_length() - 1, hi.bit_length() - 1
    span = hi_log - lo_log
    logs = [lo_log + round(i * span / max(1, k - 1)) for i in range(k)]
    return [1 << L for L in logs]


def handle_benchmark(req: dict) -> dict:
    spec = req["spec"]
    is_reduction = spec["kind"] == "reduction"
    try:
        agent_fn, agent_cfg = _compile(req["kernel_src"])
        base_fn, base_cfg = _compile(req["baseline_src"])
    except CompileError as e:
        return {"ok": True, "compiled": False, "compile_log": str(e)[:4000]}

    sm = _sm_count()
    n = spec["size"]
    atol, rtol = spec["atol"], spec["rtol"]

    bench_inputs = _make_inputs(spec, n, spec["seed"] + 2000)
    out = _run_out(n, is_reduction)

    # one warmup launch to size the timed loop, and to surface launch errors early
    try:
        if is_reduction:
            out.fill(0)
        _launch(agent_fn, agent_cfg, bench_inputs, out, n, sm)
        _launch(base_fn, base_cfg, bench_inputs, out, n, sm)
        cp.cuda.Device().synchronize()
    except cp.cuda.runtime.CUDARuntimeError as e:  # type: ignore[attr-defined]
        return {"ok": True, "compiled": True, "launch_error": str(e)}

    first = _time_kernel(base_fn, base_cfg, bench_inputs, out, n, sm, 1, 3, is_reduction)
    timed = _adaptive_timed(max(first))

    agent_samples = _time_kernel(
        agent_fn, agent_cfg, bench_inputs, out, n, sm, WARMUP, timed, is_reduction
    )
    base_samples = _time_kernel(
        base_fn, base_cfg, bench_inputs, out, n, sm, WARMUP, timed, is_reduction
    )
    agent_stats, base_stats = _stats(agent_samples), _stats(base_samples)
    speedup = base_stats["median_ms"] / max(agent_stats["median_ms"], 1e-9)

    post = _post_check(agent_fn, agent_cfg, spec, n, spec["seed"] + 3000, sm, atol, rtol, is_reduction)
    bytes_moved = n * 4 * ((spec["n_inputs"] + 1) if not is_reduction else 1)
    floor_ms = bandwidth_floor_ms(bytes_moved)
    post["bandwidth_floor_ms"] = floor_ms
    post["bandwidth_floor_ok"] = bool(agent_stats["median_ms"] >= floor_ms)
    scaling = _scaling_check(
        agent_fn, agent_cfg, base_fn, base_cfg, spec, n, sm, is_reduction
    )

    return {
        "ok": True,
        "compiled": True,
        "n": n,
        "timed_iters": timed,
        "agent": agent_stats,
        "baseline": base_stats,
        "speedup": speedup,
        "post_check": post,
        "scaling": scaling,
        "launch_config": {
            "agent": _cfg_repr(agent_cfg, n, sm),
            "baseline": _cfg_repr(base_cfg, n, sm),
        },
    }


def _cfg_repr(cfg, n, sm) -> dict:
    return {"grid": cfg.resolve_grid(n, sm), "block": cfg.block, "mode": cfg.grid_mode}


def _post_check(fn, cfg, spec, n, seed, sm, atol, rtol, is_reduction) -> dict:
    arrays = _make_inputs(spec, n, seed)
    out = _run_out(n, is_reduction)
    if is_reduction:
        out.fill(0)
    _launch(fn, cfg, arrays, out, n, sm)
    cp.cuda.Device().synchronize()
    ref = reference_impl(spec, cp, arrays)
    finite = bool(cp.isfinite(out).all())
    if is_reduction:
        agent_val, ref_val = float(out[0]), float(ref)
        abs_err = abs(agent_val - ref_val)
        thresh = atol + rtol * abs(ref_val)
        chk_a, chk_b = abs(agent_val), abs(ref_val)
    else:
        diff = cp.abs(out.astype(cp.float64) - ref)
        abs_err = float(diff.max())
        thresh = atol + rtol * float(cp.abs(ref).max())
        chk_a, chk_b = _checksum(out), _checksum(ref)
    checksum_ok = abs(chk_a - chk_b) <= 1e-3 * (abs(chk_b) + 1.0)
    return {
        "passed": bool(abs_err <= thresh) and finite,
        "max_abs_err": abs_err,
        "threshold": thresh,
        "finite": finite,
        "checksum": chk_a,
        "ref_checksum": chk_b,
        "checksum_ok": bool(checksum_ok),
    }


def _scaling_check(agent_fn, agent_cfg, base_fn, base_cfg, spec, n, sm, is_reduction) -> dict:
    n_small = max(n // 4, 4096)
    big_in = _make_inputs(spec, n, spec["seed"] + 4000)
    small_in = _make_inputs(spec, n_small, spec["seed"] + 4001)
    out_big = _run_out(n, is_reduction)
    out_small = _run_out(n_small, is_reduction)

    def med(fn, cfg, arrays, out, size):
        s = _time_kernel(fn, cfg, arrays, out, size, sm, 3, 15, is_reduction)
        return _stats(s)["median_ms"]

    a_big = med(agent_fn, agent_cfg, big_in, out_big, n)
    a_small = med(agent_fn, agent_cfg, small_in, out_small, n_small)
    b_big = med(base_fn, base_cfg, big_in, out_big, n)
    b_small = med(base_fn, base_cfg, small_in, out_small, n_small)
    # Only a real drop in run time as the input grows 4x is suspicious. Below
    # ~50 us both measurements are launch-overhead noise and the check is muted.
    noisy = a_big < 0.05 and a_small < 0.05
    monotonic = bool(noisy or a_big >= a_small * 0.7)
    return {
        "sizes": [n_small, n],
        "agent_ms": [a_small, a_big],
        "baseline_ms": [b_small, b_big],
        "agent_ratio": a_big / max(a_small, 1e-9),
        "baseline_ratio": b_big / max(b_small, 1e-9),
        "monotonic": monotonic,
    }


def handle_validate(req: dict) -> dict:
    """Deterministic solvability check: does the reference optimised kernel pass
    correctness and beat the naive baseline by >= target?"""
    spec = req["spec"]
    test_resp = handle_test({"spec": spec, "kernel_src": req["optimized_src"]})
    if not test_resp.get("compiled") or not test_resp.get("all_passed"):
        return {"ok": True, "solvable": False, "reason": "reference_optimized_incorrect", "test": test_resp}
    bench = handle_benchmark(
        {"spec": spec, "kernel_src": req["optimized_src"], "baseline_src": req["baseline_src"]}
    )
    from cuda_kernel_opt.scoring import BenchmarkOutcome

    outcome = BenchmarkOutcome.from_response(bench)
    solvable = outcome.valid and outcome.speedup >= spec["target_speedup"]
    return {
        "ok": True,
        "solvable": bool(solvable),
        "speedup": outcome.speedup,
        "target": spec["target_speedup"],
        "valid_benchmark": outcome.valid,
        "reason": "" if solvable else "reference_speedup_below_target",
    }


HANDLERS = {
    "ping": handle_ping,
    "test": handle_test,
    "benchmark": handle_benchmark,
    "validate": handle_validate,
}


def main() -> None:
    if cp is None:
        # Still answer requests so the parent gets a clean error instead of a hang.
        for _line in sys.stdin:
            sys.stdout.write(
                json.dumps({"ok": False, "fatal": True, "error": f"cupy import failed: {_CUPY_IMPORT_ERROR}"})
                + "\n"
            )
            sys.stdout.flush()
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            handler = HANDLERS[req["cmd"]]
        except Exception as e:  # malformed request — not fatal
            _emit({"ok": False, "error": f"bad request: {e}"})
            continue
        try:
            _emit(handler(req))
        except cp.cuda.memory.OutOfMemoryError as e:  # type: ignore[attr-defined]
            _emit({"ok": False, "error": f"out of GPU memory: {e}"})
        except Exception as e:  # device likely wedged — ask parent to respawn
            _emit({"ok": False, "fatal": True, "error": f"{type(e).__name__}: {e}"})
            return


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
