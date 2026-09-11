"""Deterministic CUDA-kernel-optimisation task generation.

A task is fully determined by ``(category, difficulty, seed)``. From those three
values we derive the concrete op, its constants, the input size, which
inefficiency is seeded into the starting kernel, and a known-good reference
solution that fixes it. Everything here is pure Python + NumPy so it imports
cheaply in the environment process and in the GPU worker alike.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .scoring import TARGET_SPEEDUP, TOLERANCES

ENTRY = "solve"
CATEGORIES = ("elementwise", "reduction")
DIFFICULTIES = ("easy", "medium", "hard")

# Fixed grid for grid-stride ("stride") kernels: STRIDE_MULT * SM count, capped
# at one thread per element. 32 * 30 SMs = 960 blocks of 256 on the dev card.
STRIDE_MULT = 32

# log2(size) range per tier. Kept modest so (a) timing scales measurably with
# size for the anti-hack check, and (b) even the pathological naive kernels
# (single warp, per-element global atomics) finish a timed loop in well under the
# worker timeout. Difficulty is carried mainly by the seeded inefficiency and the
# correctness tolerance, not by raw size.
SIZE_LOG2_RANGE: dict[str, tuple[int, int]] = {
    "easy": (20, 21),
    "medium": (21, 22),
    "hard": (22, 23),
}
# Inefficiencies whose *naive* kernel serialises hard (one 32-thread warp does
# the whole array) get their size clamped so a benchmark stays fast. Only
# ``poor_block_grid`` still qualifies, and it is currently unassigned (see
# ``_TIER_OPTIONS``); the entry keeps the clamp correct if it is ever restored.
_SLOW_NAIVE = {"poor_block_grid"}
_SLOW_NAIVE_MAX_LOG2 = 21

# ``no_shared_memory_reuse``: the naive does one global atomicAdd per *thread*
# (~grid*block of them), a fixed cost that amortises as the array grows, so the
# reference block-reduction's speedup over it shrinks from ~11x at 2^20 to ~4x at
# 2^23. Clamp these to 2^20-2^21 so every reduction tier keeps a real ~6-12x of
# headroom to spread the reward curve over; tiers are separated by target,
# tolerance and op, not raw size.
_HEADROOM_SENSITIVE = {"no_shared_memory_reuse"}
_HEADROOM_SENSITIVE_MAX_LOG2 = 21

# ---------------------------------------------------------------------------
# Launch-config directive:  // launch: grid=<mode> block=<pow2>
#   grid=stride       fixed grid, kernel must use a grid-stride loop
#   grid=n_div        grid = ceil(n / block), one element per thread
#   grid=blocks:<k>   explicit block count
# ---------------------------------------------------------------------------
_DIRECTIVE_RE = re.compile(
    r"//\s*launch:\s*grid=(stride|n_div|blocks:\d+)\s+block=(\d+)", re.IGNORECASE
)
DEFAULT_DIRECTIVE = "// launch: grid=n_div block=256"


@dataclass(frozen=True)
class LaunchConfig:
    grid_mode: str  # "stride" | "n_div" | "blocks"
    block: int
    grid_blocks: int | None = None  # set when grid_mode == "blocks"

    def validate(self) -> None:
        if self.block < 32 or self.block > 1024 or (self.block & (self.block - 1)) != 0:
            raise ValueError(
                f"launch directive: block must be a power of two in [32, 1024], got {self.block}"
            )
        if self.grid_mode == "blocks" and (self.grid_blocks or 0) < 1:
            raise ValueError("launch directive: grid=blocks:<k> needs k >= 1")

    def resolve_grid(self, n: int, sm_count: int) -> int:
        if self.grid_mode == "n_div":
            return max(1, math.ceil(n / self.block))
        if self.grid_mode == "blocks":
            assert self.grid_blocks is not None
            return self.grid_blocks
        # stride
        per_elem = max(1, math.ceil(n / self.block))
        return max(1, min(STRIDE_MULT * sm_count, per_elem))


def parse_launch_directive(src: str) -> LaunchConfig:
    """Parse the first ``// launch:`` directive; fall back to the default."""
    m = _DIRECTIVE_RE.search(src) or _DIRECTIVE_RE.search(DEFAULT_DIRECTIVE)
    assert m is not None
    grid_raw, block_raw = m.group(1).lower(), int(m.group(2))
    if grid_raw.startswith("blocks:"):
        cfg = LaunchConfig("blocks", block_raw, int(grid_raw.split(":")[1]))
    else:
        cfg = LaunchConfig(grid_raw, block_raw)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Op:
    name: str
    category: str
    n_inputs: int
    a: float
    b: float
    c: float
    reduce_kind: str = ""  # "sum" | "sumsq" | "mean"
    poly: tuple[float, ...] = ()  # for the warp-divergence poly family
    rtol_override: float | None = None

    # -- device-side C expression for one element (elementwise ops) --
    def device_expr(self, x: str, y: str) -> str:
        a, b, c = f"{self.a:.8f}f", f"{self.b:.8f}f", f"{self.c:.8f}f"
        if self.name == "saxpy":
            return f"fmaf({a}, {x}, {b})"
        if self.name == "square":
            return f"fmaf(fmaf({a}, {x}, {b}), {x}, {c})"
        if self.name == "gelu_tanh":
            return (
                f"0.5f*{x}*(1.f + tanhf(0.7978845608028654f*"
                f"({x} + 0.044715f*{x}*{x}*{x})))"
            )
        if self.name == "scale_clamp":
            return f"fminf(fmaxf({a}*{x}, 0.f), {b})"
        if self.name == "axpby":
            return f"fmaf({a}, {x}, {b}*{y})"
        if self.name == "muladd":
            return f"fmaf({x}, {y}, {a}*{x})"
        if self.name == "poly":  # Horner, fma form
            terms = ", ".join(f"{k:.8f}f" for k in self.poly)
            return f"__cko_poly_fma({x}, (const float[]){{{terms}}}, {len(self.poly)})"
        raise KeyError(self.name)

    # -- device-side C expression for one reduction contribution --
    def device_contrib(self, v: str) -> str:
        if self.reduce_kind in ("sum", "mean"):
            return v
        if self.reduce_kind == "sumsq":
            return f"({v}*{v})"
        raise KeyError(self.reduce_kind)

    @property
    def effective_rtol_key(self) -> float | None:
        return self.rtol_override

    def summary(self) -> str:
        if self.category == "reduction":
            base = {
                "sum": "out[0] = sum(x[i])",
                "sumsq": "out[0] = sum(x[i] * x[i])",
                "mean": "out[0] = sum(x[i]) / n",
            }[self.reduce_kind]
            return f"{base}   (reduction over the whole array)"
        if self.name == "saxpy":
            return f"out[i] = {self.a:.4f} * x[i] + {self.b:.4f}"
        if self.name == "square":
            return f"out[i] = {self.a:.4f}*x[i]^2 + {self.b:.4f}*x[i] + {self.c:.4f}"
        if self.name == "gelu_tanh":
            return "out[i] = gelu(x[i])   (tanh approximation)"
        if self.name == "scale_clamp":
            return f"out[i] = clamp({self.a:.4f} * x[i], 0, {self.b:.4f})"
        if self.name == "axpby":
            return f"out[i] = {self.a:.4f} * x[i] + {self.b:.4f} * y[i]"
        if self.name == "muladd":
            return f"out[i] = x[i] * y[i] + {self.a:.4f} * x[i]"
        if self.name == "poly":
            return f"out[i] = P(x[i])   (degree-{len(self.poly) - 1} polynomial, fixed coefficients)"
        raise KeyError(self.name)


_ELEMENTWISE_OPS_1 = ("saxpy", "square", "gelu_tanh", "scale_clamp")
_ELEMENTWISE_OPS_2 = ("axpby", "muladd")
_REDUCE_KINDS = ("sum", "sumsq", "mean")


def reference_impl(spec: dict, xp, arrays: list) -> Any:
    """Compute the trusted reference. ``xp`` is numpy or cupy; ``arrays`` are the
    input arrays already on the right device. Elementwise -> array; reduction ->
    python float. Accumulates in float64."""
    x = arrays[0].astype(xp.float64)
    y = arrays[1].astype(xp.float64) if len(arrays) > 1 else None
    name = spec["op"]
    a, b, c = spec.get("a", 0.0), spec.get("b", 0.0), spec.get("c", 0.0)

    if spec["kind"] == "reduction":
        rk = spec["reduce_kind"]
        if rk == "sumsq":
            total = xp.sum(x * x)
        else:
            total = xp.sum(x)
        total = float(total)
        if rk == "mean":
            total /= x.size
        return total

    if name == "saxpy":
        return a * x + b
    if name == "square":
        return a * x * x + b * x + c
    if name == "gelu_tanh":
        return 0.5 * x * (1.0 + xp.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))
    if name == "scale_clamp":
        return xp.minimum(xp.maximum(a * x, 0.0), b)
    if name == "axpby":
        return a * x + b * y
    if name == "muladd":
        return x * y + a * x
    if name == "poly":
        coeffs = spec["poly"]
        acc = xp.zeros_like(x)
        for k in coeffs:
            acc = acc * x + k
        return acc
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Kernel source assembly
# ---------------------------------------------------------------------------
_POLY_HELPER = """
__device__ __forceinline__ float __cko_poly_fma(float x, const float* k, int m) {
    float acc = 0.f;
#pragma unroll 1
    for (int t = 0; t < m; ++t) acc = fmaf(acc, x, k[t]);
    return acc;
}
__device__ __forceinline__ float __cko_poly_mul(float x, const float* k, int m) {
    float acc = 0.f;
#pragma unroll 1
    for (int t = 0; t < m; ++t) acc = acc * x + k[t];
    return acc;
}
"""


def _sig(op: Op) -> str:
    if op.category == "reduction":
        return "const float* __restrict__ x, float* __restrict__ out, int n"
    if op.n_inputs == 2:
        return (
            "const float* __restrict__ x, const float* __restrict__ y, "
            "float* __restrict__ out, int n"
        )
    return "const float* __restrict__ x, float* __restrict__ out, int n"


def _load_y(op: Op, idx: str) -> str:
    return f"        float yv = y[{idx}];\n" if op.n_inputs == 2 else ""


def _prelude(op: Op) -> str:
    return _POLY_HELPER if op.name == "poly" else ""


def _elw_clean(op: Op, directive: str = "// launch: grid=stride block=256") -> str:
    body_y = _load_y(op, "i")
    return f"""{directive}
{_prelude(op)}
extern "C" __global__ void {ENTRY}({_sig(op)}) {{
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {{
        float xv = x[i];
{body_y}        out[i] = {op.device_expr("xv", "yv")};
    }}
}}
"""


def _reduce_clean(op: Op) -> str:
    scale = " * (1.f / (float)n)" if op.reduce_kind == "mean" else ""
    return f"""// launch: grid=stride block=256
extern "C" __global__ void {ENTRY}({_sig(op)}) {{
    __shared__ float sdata[256];
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        acc += {op.device_contrib("x[i]")};
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {{
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }}
    if (threadIdx.x == 0) atomicAdd(&out[0], sdata[0]{scale});
}}
"""


# -- naive builders (one per inefficiency) --
def _naive_poor_block_grid(op: Op) -> str:
    if op.category == "reduction":
        scale = " * (1.f / (float)n)" if op.reduce_kind == "mean" else ""
        return f"""// launch: grid=blocks:1 block=32
extern "C" __global__ void {ENTRY}({_sig(op)}) {{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&out[0], {op.device_contrib("x[i]")}{scale});
}}
"""
    return _elw_clean(op, "// launch: grid=blocks:1 block=32")


def _naive_unnecessary_syncthreads(op: Op) -> str:
    y_decl = "    float yv = 0.f;\n" if op.n_inputs == 2 else ""
    y_ptr = (
        "    const volatile float* vy = y;\n" if op.n_inputs == 2 else ""
    )
    y_reload = "            yv = vy[i];\n" if op.n_inputs == 2 else ""
    return f"""// launch: grid=n_div block=256
{_prelude(op)}
extern "C" __global__ void {ENTRY}({_sig(op)}) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const volatile float* vx = x;   // volatile: every iteration re-reads global memory
{y_ptr}    float xv = 0.f, r = 0.f;
{y_decl}#pragma unroll 1
    for (int k = 0; k < 8; ++k) {{
        if (i < n) {{
            xv = vx[i];
{y_reload}        }}
        __syncthreads();            // unnecessary: threads share no data
        r = {op.device_expr("xv", "yv")};
        __syncthreads();            // unnecessary
    }}
    if (i < n) out[i] = r;
}}
"""


def _naive_uncoalesced(op: Op, logn: int) -> str:
    body_y = "        float yv = y[j];\n" if op.n_inputs == 2 else ""
    return f"""// launch: grid=n_div block=256
{_prelude(op)}
extern "C" __global__ void {ENTRY}({_sig(op)}) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (unsigned)n) {{
        int lg = 31 - __clz((unsigned)n);          // log2(n), n is a power of two
        int k = lg >> 1;
        unsigned int m = (unsigned)n - 1u;
        unsigned int j = ((i << k) | (i >> (lg - k))) & m;   // rotate index bits: fully uncoalesced
        float xv = x[j];
{body_y}        out[j] = {op.device_expr("xv", "yv")};
    }}
}}
"""


def _naive_warp_divergence(op: Op) -> str:
    # op is forced to the "poly" family for this inefficiency
    m = len(op.poly)
    terms = ", ".join(f"{c:.8f}f" for c in op.poly)
    return f"""// launch: grid=n_div block=256
{_POLY_HELPER}
extern "C" __global__ void {ENTRY}({_sig(op)}) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        float xv = x[i];
        const float kk[{m}] = {{{terms}}};
        float r;
        if ((threadIdx.x & 31) < 16) r = __cko_poly_fma(xv, kk, {m});   // half the lanes ...
        else                         r = __cko_poly_mul(xv, kk, {m});   // ... the other half: same math
        out[i] = r;
    }}
}}
"""


def _opt_warp_divergence(op: Op) -> str:
    m = len(op.poly)
    terms = ", ".join(f"{c:.8f}f" for c in op.poly)
    return f"""// launch: grid=stride block=256
{_POLY_HELPER}
extern "C" __global__ void {ENTRY}({_sig(op)}) {{
    const float kk[{m}] = {{{terms}}};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {{
        out[i] = __cko_poly_fma(x[i], kk, {m});
    }}
}}
"""


# ---------------------------------------------------------------------------
# Inefficiency taxonomy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Inefficiency:
    key: str
    categories: tuple[str, ...]
    tiers: tuple[str, ...]
    description: str
    fix_hint: str


INEFFICIENCIES: dict[str, Inefficiency] = {
    "poor_block_grid": Inefficiency(
        "poor_block_grid",
        ("elementwise", "reduction"),
        (),  # defined for docs/TUI; not tier-assigned (see _TIER_OPTIONS note)
        "The kernel is launched with a single 32-thread block, so one warp does "
        "the entire array while the other ~29 SMs sit idle.",
        "Declare a real launch config (e.g. `grid=stride block=256`) and keep the "
        "grid-stride loop so every SM is filled.",
    ),
    "unnecessary_syncthreads": Inefficiency(
        "unnecessary_syncthreads",
        ("elementwise",),
        (),  # defined for docs/TUI; not tier-assigned (see _TIER_OPTIONS note)
        "Each element is re-read from global memory 8 times (a volatile pointer "
        "defeats caching) inside a loop that calls __syncthreads() twice per "
        "iteration, even though threads share no data.",
        "Load each input once, compute once, drop every __syncthreads(); a plain "
        "grid-stride loop is enough.",
    ),
    "uncoalesced": Inefficiency(
        "uncoalesced",
        ("elementwise",),
        ("easy", "medium", "hard"),
        "Thread i accesses element `rotate_bits(i)` instead of `i`, so adjacent "
        "threads touch addresses ~n/2 apart and every memory transaction is a "
        "near-cache-line-sized waste.",
        "Access element `i` directly (a coalesced, unit-stride grid-stride loop); "
        "the permutation is a red herring — reading and writing the same index in "
        "natural order is equivalent.",
    ),
    "warp_divergence": Inefficiency(
        "warp_divergence",
        ("elementwise",),
        ("medium", "hard"),
        "Lanes 0-15 of every warp evaluate the polynomial with an fma chain while "
        "lanes 16-31 evaluate the identical polynomial with a mul/add chain, so "
        "both code paths run for every warp.",
        "Pick one evaluation for all lanes — the two chains compute the same "
        "function — and drop the lane-id branch entirely.",
    ),
    "no_shared_memory_reuse": Inefficiency(
        "no_shared_memory_reuse",
        ("reduction",),
        ("easy", "medium", "hard"),
        "Every thread does a global atomicAdd into out[0], so the whole reduction "
        "serialises on one memory location even though each thread only adds once.",
        "Reduce within each block in shared memory (tree reduction), then do a "
        "single atomicAdd per block.",
    ),
}

# category -> tier -> allowed inefficiency keys
_TIER_OPTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "elementwise": {
        "easy": ("uncoalesced",),
        "medium": ("uncoalesced", "warp_divergence"),
        "hard": ("uncoalesced", "warp_divergence"),
    },
    "reduction": {
        "easy": ("no_shared_memory_reuse",),
        "medium": ("no_shared_memory_reuse",),
        "hard": ("no_shared_memory_reuse",),
    },
}

# ``unnecessary_syncthreads`` and ``poor_block_grid`` are defined above and shown
# in the TUI/docs but are not currently assigned to a tier:
#   * ``unnecessary_syncthreads`` -- re-reading a cached element and an
#     uncontended block barrier are both too cheap on the dev GPU (~1.3x).
#   * ``poor_block_grid`` -- on this GPU it is all-or-nothing: a single 32-thread
#     warp is pathological (70-250x, so any competent rewrite instantly maxes the
#     score with no gradient), while a merely under-subscribed grid is worth
#     under 1.5x because the elementwise kernels are already bandwidth-bound.
# Both are kept for a faster host or a per-inefficiency sub-target.


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
@dataclass
class KernelTask:
    task_id: str
    category: str
    difficulty: str
    seed: int
    op: Op
    inefficiency: str
    inefficiency_desc: str
    fix_hint: str
    size: int
    size_min: int
    size_max: int
    n_inputs: int
    entry: str
    signature: str
    atol: float
    rtol: float
    target_speedup: float
    input_lo: float
    input_hi: float
    naive_src: str
    optimized_src: str = field(repr=False)

    def reference_spec(self) -> dict:
        return {
            "kind": "reduction" if self.category == "reduction" else "elementwise",
            "op": self.op.name,
            "a": self.op.a,
            "b": self.op.b,
            "c": self.op.c,
            "reduce_kind": self.op.reduce_kind,
            "n_inputs": self.n_inputs,
            "poly": list(self.op.poly),
        }

    def worker_spec(self) -> dict:
        """Everything the GPU worker needs: reference math + sizes + tolerances."""
        spec = self.reference_spec()
        spec.update(
            task_id=self.task_id,
            seed=self.seed,
            size=self.size,
            size_min=self.size_min,
            size_max=self.size_max,
            atol=self.atol,
            rtol=self.rtol,
            target_speedup=self.target_speedup,
            input_lo=self.input_lo,
            input_hi=self.input_hi,
        )
        return spec

    def describe(self, include_source: bool = True) -> str:
        kind = (
            "reduction over the whole array"
            if self.category == "reduction"
            else (
                "element-wise, two input arrays"
                if self.n_inputs == 2
                else "element-wise, single input array"
            )
        )
        lines = [
            f"CUDA kernel optimisation task  [{self.task_id}]",
            "",
            "Function to implement (must stay numerically correct vs the reference):",
            f"  {self.op.summary()}    ({kind})",
            "",
            "Entry point (exact signature required):",
            f'  extern "C" __global__ void {self.entry}({self.signature})',
            "",
            "Launch configuration — declare it in a comment the harness reads:",
            "  // launch: grid=stride block=256     (fixed grid, use a grid-stride loop)",
            "  // launch: grid=n_div  block=256     (grid = ceil(n/block), one element per thread)",
            "  // launch: grid=blocks:8 block=128   (explicit block count)",
            "  block must be a power of two in [32, 1024]. Default: grid=n_div block=256.",
        ]
        if self.category == "reduction":
            lines.append("  The harness zeroes out[0] before every launch.")
        lines += [
            "",
            f"Seeded inefficiency in the starting kernel: {self.inefficiency}",
            f"  {self.inefficiency_desc}",
            "",
            f"Input: n in [2^{int(math.log2(self.size_min))}, 2^{int(math.log2(self.size_max))}], "
            f"values ~ U({self.input_lo:g}, {self.input_hi:g}), float32.",
            "Correctness is checked on several freshly-randomised inputs at different "
            "sizes; the benchmark re-randomises again. Passing means",
            f"  max|agent - ref| <= {self.atol:g} + {self.rtol:g} * max|ref|.",
            "",
            f"Target speedup vs the naive baseline for full score: {self.target_speedup:g}x",
        ]
        if include_source:
            lines += ["", "Current kernel:", "```cuda", self.current_source_placeholder(), "```"]
        return "\n".join(lines)

    def current_source_placeholder(self) -> str:
        return self.naive_src.strip()


def _rng_params(rng: np.random.Generator) -> tuple[float, float, float]:
    a = float(rng.uniform(0.5, 2.0)) * (1.0 if rng.random() < 0.5 else -1.0)
    b = float(rng.uniform(-1.0, 1.0))
    c = float(rng.uniform(-0.5, 0.5))
    return round(a, 4), round(b, 4), round(c, 4)


def _make_op(category: str, inefficiency: str, seed: int) -> Op:
    rng = np.random.default_rng(seed * 2_654_435_761 % (2**32))
    a, b, c = _rng_params(rng)
    if category == "reduction":
        rk = _REDUCE_KINDS[int(rng.integers(len(_REDUCE_KINDS)))]
        return Op("sum" if rk == "sum" else rk, "reduction", 1, a, b, c, reduce_kind=rk)
    if inefficiency == "warp_divergence":
        deg = 12
        # Shrink high-order coefficients so |P(x)| stays O(1) for |x| < 2 (keeps
        # the absolute correctness threshold tight instead of ballooning).
        poly = tuple(
            round(float(rng.uniform(-0.4, 0.4)) / (2.0**i), 8) for i in range(deg + 1)
        )
        return Op("poly", "elementwise", 1, a, b, c, poly=poly, rtol_override=1e-3)
    # generic elementwise op family
    if rng.random() < 0.35:
        name = _ELEMENTWISE_OPS_2[int(rng.integers(len(_ELEMENTWISE_OPS_2)))]
        n_inputs = 2
    else:
        name = _ELEMENTWISE_OPS_1[int(rng.integers(len(_ELEMENTWISE_OPS_1)))]
        n_inputs = 1
    if name == "scale_clamp":  # b must be a positive clamp ceiling
        b = abs(b) + 0.5
    return Op(name, "elementwise", n_inputs, round(a, 4), round(b, 4), round(c, 4))


def _build_sources(task_category: str, inefficiency: str, op: Op, logn: int) -> tuple[str, str]:
    if inefficiency == "poor_block_grid":
        naive = _naive_poor_block_grid(op)
        opt = _reduce_clean(op) if op.category == "reduction" else _elw_clean(op)
    elif inefficiency == "unnecessary_syncthreads":
        naive = _naive_unnecessary_syncthreads(op)
        opt = _elw_clean(op)
    elif inefficiency == "uncoalesced":
        naive = _naive_uncoalesced(op, logn)
        opt = _elw_clean(op)
    elif inefficiency == "warp_divergence":
        naive = _naive_warp_divergence(op)
        opt = _opt_warp_divergence(op)
    elif inefficiency == "no_shared_memory_reuse":
        scale = " * (1.f / (float)n)" if op.reduce_kind == "mean" else ""
        # Each thread grid-stride accumulates into a register (so the memory
        # traffic is already coalesced and minimal) and then does ONE global
        # atomicAdd into out[0]. The whole reduction still serialises on that one
        # address across ~grid*block threads; a block-level shared-memory tree
        # reduction with a single atomicAdd per block is ~7-14x faster. This is a
        # "mediocre first attempt", not the pathological per-element-atomic form.
        naive = f"""// launch: grid=stride block=256
extern "C" __global__ void {ENTRY}({_sig(op)}) {{
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        acc += {op.device_contrib("x[i]")};
    atomicAdd(&out[0], acc{scale});
}}
"""
        opt = _reduce_clean(op)
    else:
        raise KeyError(inefficiency)
    return naive.strip() + "\n", opt.strip() + "\n"


def _stable_seed(category: str, difficulty: str, seed: int) -> int:
    """A process-independent 32-bit seed for ``(category, difficulty, seed)``.

    The builtin ``hash()`` randomises string hashing per process (PEP 456 /
    ``PYTHONHASHSEED``), so ``hash((category, difficulty, seed))`` returned a
    different value in the eval client that builds the prompt than in the env
    worker that runs the rollout. That made ``generate_task`` pick a different
    inefficiency, op family and input size in each: the model was shown one task
    and graded against a different reference. A fixed digest keeps the triple
    deterministic across processes, as the design doc promises.
    """
    blob = f"{category}\x1f{difficulty}\x1f{int(seed)}".encode()
    return int.from_bytes(hashlib.blake2b(blob, digest_size=4).digest(), "big")


def generate_task(category: str, difficulty: str, seed: int) -> KernelTask:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty!r}")

    rng = np.random.default_rng(_stable_seed(category, difficulty, seed))
    options = _TIER_OPTIONS[category][difficulty]
    inefficiency = options[int(rng.integers(len(options)))]

    lo_log, hi_log = SIZE_LOG2_RANGE[difficulty]
    if inefficiency in _SLOW_NAIVE:
        hi_log = min(hi_log, _SLOW_NAIVE_MAX_LOG2)
        lo_log = min(lo_log, hi_log)
    if inefficiency in _HEADROOM_SENSITIVE:
        hi_log = min(hi_log, _HEADROOM_SENSITIVE_MAX_LOG2)
        lo_log = min(lo_log, hi_log)
    logn = int(rng.integers(lo_log, hi_log + 1))
    size = 1 << logn
    size_min, size_max = 1 << lo_log, 1 << hi_log

    op = _make_op(category, inefficiency, seed)
    naive_src, optimized_src = _build_sources(category, inefficiency, op, logn)

    tol = TOLERANCES[difficulty]
    if category == "reduction":
        rtol = tol["reduction_rtol"]
        # All-positive, mean ~ 1.0: keeps the reference far from zero so the
        # float32 accumulation error stays comfortably inside a relative tolerance
        # (a signed distribution summing near zero would make correctness flaky).
        input_lo, input_hi = 0.25, 1.75
    else:
        rtol = op.rtol_override or tol["rtol"]
        input_lo, input_hi = -2.0, 2.0
    atol = tol["atol"]

    ineff = INEFFICIENCIES[inefficiency]
    return KernelTask(
        task_id=f"{category}/{difficulty}/{seed}",
        category=category,
        difficulty=difficulty,
        seed=seed,
        op=op,
        inefficiency=inefficiency,
        inefficiency_desc=ineff.description,
        fix_hint=ineff.fix_hint,
        size=size,
        size_min=size_min,
        size_max=size_max,
        n_inputs=op.n_inputs,
        entry=ENTRY,
        signature=_sig(op),
        atol=atol,
        rtol=rtol,
        target_speedup=TARGET_SPEEDUP[category][difficulty],
        input_lo=input_lo,
        input_hi=input_hi,
        naive_src=naive_src,
        optimized_src=optimized_src,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You optimise CUDA kernels. Each task hands you a correct-but-slow __global__ kernel \
with one seeded inefficiency. Rewrite it to run faster on the GPU while staying \
numerically correct against a reference implementation.

Workflow with the tools you have:
  1. view_kernel        - re-read the task spec and the current kernel source.
  2. submit_kernel      - replace the current kernel with your edited version.
  3. test_correctness   - compile and check against several fresh random inputs
                          (unlimited; use it freely).
  4. benchmark          - time your kernel and the naive baseline on fresh inputs
                          and report the speedup (LIMITED number of calls).
  5. check_score        - see your best score, best speedup and remaining budget.

Rules and anti-gaming notes:
  - Score is 0 unless the kernel is correct AND a benchmark is valid. There is no
    partial credit for fast-but-wrong.
  - Correctness, benchmark and the benchmark's own post-check each use different
    randomised inputs, so precomputing or memorising an answer does not work.
  - The benchmark re-verifies your kernel's output, checksums it, and checks that
    run time grows with input size. A kernel that skips work is rejected.
  - Keep the exact entry-point signature. Declare your launch config in the
    `// launch:` comment. Finish by simply replying without a tool call.
"""


def build_dataset(
    categories: tuple[str, ...] = CATEGORIES,
    difficulties: tuple[str, ...] = DIFFICULTIES,
    num_per_combo: int = 2,
    seed0: int = 0,
) -> list[dict]:
    rows: list[dict] = []
    for category in categories:
        for difficulty in difficulties:
            for k in range(num_per_combo):
                seed = seed0 + k
                kt = generate_task(category, difficulty, seed)
                rows.append(
                    {
                        "prompt": [{"role": "user", "content": kt.describe(include_source=True)}],
                        "info": json.dumps(
                            {"category": category, "difficulty": difficulty, "seed": seed}
                        ),
                    }
                )
    return rows
