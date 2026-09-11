# CUDA Kernel Optimization Environment — Design Doc & Test Plan

Status: **living document**. Revised after each milestone (see Changelog at bottom).

---

## 1. Goal

A `verifiers` environment where an agent is handed a deliberately inefficient CUDA
kernel and must iteratively rewrite it to run faster on real hardware while staying
numerically correct against a trusted reference. Score is a hard-gated function of
measured speedup over the naive baseline.

The environment must be **robust to reward hacking**, which here is a bigger risk
than in a QA/math checker because the agent ships code that actually runs and gets
timed, and that code can behave differently depending on how it is measured.

---

## 2. Hardware / toolchain reality (this workspace)

- GPU: **NVIDIA GeForce RTX 3060 Laptop**, compute capability **sm_86**, 6 GB, 30 SMs.
- Driver: 576.88 → supports CUDA **12.9** runtime. (`nvidia-smi` reports "CUDA 12.9".)
- Installed CUDA Toolkit: **13.3** only. A binary linked against the 13.3 runtime
  **crashes** on this driver (`cudaGetDeviceCount` → access violation), and NVRTC
  13.3 emits PTX the 12.9 driver cannot JIT (module load → access violation).
- MSVC 14.51 (VS 2026 BuildTools) is present but `nvcc` + `cl.exe` is a dead end
  because of the runtime/driver mismatch above.

### Chosen compile/run path: **CuPy + NVRTC 12.x, no system toolkit**

- `cupy-cuda12x==14.2` for device memory, launch, events, high-level reference ops.
- `nvidia-cuda-nvrtc-cu12==12.9` + `nvidia-cuda-runtime-cu12==12.9` pip wheels
  provide a **CUDA 12.9 NVRTC** that the driver can JIT.
- CuPy's path finder otherwise picks up the system 13.3 install via `CUDA_PATH`,
  so the GPU worker **scrubs `CUDA_PATH` / `CUDA_HOME` / `CUDA_ROOT`** from its
  environment *before* importing cupy. Verified working: RawModule compiles,
  kernel runs, events time correctly, NVRTC reports `(12, 9)`.
- No `nvcc`, no host compiler, no `.exe` build step. The agent's code only ever
  runs as a `__global__` kernel on the GPU — NVRTC cannot do host I/O or
  `system()`, so the compile step is inherently sandboxed.

---

## 3. Anti-reward-hacking design (the core of this environment)

Threats from the brief and the defenses implemented:

| # | Threat | Defense |
|---|--------|---------|
| 1 | **Dead-code elimination**: timed kernel does nothing; compiler optimizes the computation away; a separate correctness pass reads cached/precomputed output. | The **benchmark itself produces the verified output**. After the timed iterations, one extra launch runs on a dedicated freshly-randomized `check` input set and its output is epsilon-compared to the reference for *those* inputs. A benchmark whose post-check fails is `invalid` and cannot raise the score. Because the kernel writes to a `float*` output param, NVRTC must treat those global stores as observable and cannot elide them; doing "less work" is what we actually defend against. |
| 2 | **Caching across calls / no state reset**: kernel replays a stored result. | The GPU **worker is a separate subprocess with its own CUDA context**. It holds no output buffers between calls — every `test`/`benchmark`/`validate` request allocates fresh device buffers and regenerates inputs from seeds in the request. The worker is killed + respawned between rollouts (and on any hang). No result cache anywhere in the harness. |
| 3 | **Passing one test ≠ solving the task**: output memorized for the tested case. | Inputs are **re-randomized with a different seed for every phase**: correctness uses `seed + 1000 + case_idx`, benchmark uses `seed + 2000`, the benchmark post-check uses `seed + 3000`, scaling uses `seed + 4000`. Correctness runs `n_cases` (default 5) cases at different sizes within the task's size range. Nothing can be precomputed at submit time. |
| 4 | **Compiled-away / short-circuited kernel shows suspiciously fast/flat timing.** | Two checks. (a) **Bandwidth floor**: a kernel that really moves `bytes` of data cannot beat `bytes / peak_bw` by much — if the benchmark median is below `0.18 * bytes/peak_bw` the kernel is not touching the array and the benchmark is rejected. (b) **Monotonic**: the agent kernel is timed at `size` and `size//4`; run time must not *drop* as the input grows (muted below ~50 us where launch overhead dominates and honest kernels look flat too). The earlier "size ratio must be >= 1.5" heuristic was removed — genuinely fast kernels have flat timing at these sizes and it false-positived on honest solutions. |
| 5 | **Checksum / "did real computation happen"**. | The worker computes a device-side reduction (`sum(|out|)` in float64) of the benchmark output and compares to the same reduction of the reference output (`checksum_ok`), and asserts the output is all-finite (`finite`). Both must hold for the benchmark to be valid. |
| 6 | **Tuning against the timer**: brute-forcing many benchmark calls. | `benchmark` has a **per-rollout call budget** (default 12). `test_correctness` is unlimited (cheap, no timing signal beyond pass/fail). Score is taken from the best *valid* benchmark. |
| 7 | **Baseline drift**: comparing against a stale/other baseline. | Every `benchmark` call **re-times the naive baseline in the same call, on the same inputs, through the same harness path** as the agent kernel. Speedup = `median(baseline) / median(agent)` from that call. |
| 8 | **Nonsense "kernel" that isn't the entry point.** | The worker requires an `extern "C" __global__ void <entry>(...)` with the exact expected signature; a missing/rename ⇒ compile-invalid, not a pass. |

### Correctness comparison

Epsilon comparison, never exact equality (reordered reductions + fast-math shift
bits legitimately). Per task: `atol`, `rtol` from the difficulty tier. Pass iff
`max(|agent - ref|) <= atol + rtol * max(|ref|)` elementwise, or for reductions
the scalar analog with `rtol` widened to absorb float32 accumulation error over
`n` terms. Reference is computed by CuPy high-level ops in **float64** then cast.

### Scoring

```
target      = TARGET_SPEEDUP[category][difficulty]
              # elementwise  easy 1.7  medium 1.7  hard 1.9
              # reduction    easy 3.0  medium 5.0  hard 6.0
correct_ok  = compiled AND all correctness cases pass
valid_bench = compiled AND post_check.passed AND checksum_ok AND finite
              AND bandwidth_floor_ok AND scaling_monotonic
if not (correct_ok AND at least one valid_bench):  score = 0.0          # HARD GATE
else:
    s     = best valid  median(baseline)/median(agent)
    score = clamp((s - 1.0) / (target - 1.0), 0.0, 1.0)
```

`target` is per **(category, tier)** because the two categories have very
different achievable ceilings on this GPU: a `no_shared_memory_reuse` reduction
(one global `atomicAdd` per thread) has ~8-14x of headroom to a proper
shared-memory block reduction, while the elementwise inefficiencies top out
around 2.2-5x. A single shared target would either saturate reduction at 1.0 with
no gradient (the pre-2026-09 behaviour: pathological single-warp / per-element
atomic baselines gave 45-250x, always clamped to 1.0) or be unreachable for the
weaker elementwise fixes.

Targets are set to ~30-60% of the *reference* speedup so that (a) the known-good
reference clears with margin — `scripts/generate_tasks.py` keeps 48/48 of an
8-seed sweep — and (b) a partial fix (divergent/interleaved reduction, badly
configured launch) lands mid-ramp instead of at 0 or 1. `speedup_to_score` and
`score_from_benchmark` take the resolved `target` float; `RolloutScoreState`
carries `target_speedup` from the task.

`score = 0` at baseline speed, `1.0` at the tier target, capped at 1.0 above it.
`best_score` over the rollout is what `check_score` reports and what the rubric
returns as reward.

---

## 4. Task generation

Parameterized by **`(category, difficulty, seed)`** — everything else derives
deterministically.

### Categories (start)
- `elementwise` — `out[i] = f(x[i])` or `out[i] = f(x[i], y[i])`, `f` an
  affine + light-nonlinear combo chosen by seed (`saxpy`, `square`, `gelu_tanh`,
  `scale_clamp`, `axpby`).
- `reduction` — `out[0] = reduce(x)`; reduce ∈ {`sum`, `sumsq`, `mean`}
  (sum-family only: `atomicAdd(float)` is native on sm_86; `max`/`argmax` deferred
  with transpose to the later extension).

### Difficulty controls (as implemented)
| tier | elementwise inefficiencies | reduction inefficiencies | size range (elw / red) | `atol` | `rtol` elw / red | target elw / red |
|------|---------------------------|--------------------------|-----------|--------|------------------|--------|
| easy   | `uncoalesced` | `no_shared_memory_reuse` | 2^20–2^21 / 2^20–2^21 | 1e-4 | 1e-4 / 3e-3 | 1.7 / 3.0 |
| medium | `uncoalesced`, `warp_divergence` | `no_shared_memory_reuse` | 2^21–2^22 / 2^21 | 1e-4 | 1e-4 / 5e-3 | 1.7 / 5.0 |
| hard   | `uncoalesced`, `warp_divergence` | `no_shared_memory_reuse` | 2^22–2^23 / 2^21 | 2e-4 | poly override 1e-3 / 1e-2 | 1.9 / 6.0 |

`reduction` is clamped to ≤2^21 (`_HEADROOM_SENSITIVE`): the naive's one-atomic-
per-thread cost is fixed, so the reference's speedup over it decays from ~11x at
2^20 to ~4x at 2^23 — small inputs keep a wide reward ramp. Reduction tiers are
separated by target / tolerance / op rather than size.

`poor_block_grid` was **dropped from all active tiers** (kept in the taxonomy for
a faster host). On the RTX 3060 it is all-or-nothing: a single 32-thread warp is
pathological (70-250x — any competent rewrite instantly maxes the score), and a
merely under-subscribed grid is worth <1.5x because these kernels are already
bandwidth-bound. This removes the last source of reward saturation.

Notes on what changed from the first sketch, and why:
- **`no_vectorized_loads` dropped.** On Ampere, re-issuing loads for the same
  cache line and scalar-vs-`float4` access on a bandwidth-bound kernel are both
  worth well under 1.5x, so its reference fix can't clear even the easy target.
  Kept in the taxonomy/TUI for documentation and a future host.
- **`unnecessary_syncthreads` defined but unassigned** for the same reason
  (uncontended block barrier + `volatile` re-reads ≈ 1.2–1.4x here).
- **Sizes clamped to 2^20–2^23** (and 2^21 for the single-warp / single-atomic
  naive kernels): big enough that timing scales, small enough that the
  pathological baselines finish a timed loop well inside the worker timeout.
- **Reduction inputs are `U(0.25, 1.75)`** (all positive, mean ≈ 1) so the
  reference is far from zero and float32 accumulation error stays inside a
  relative tolerance.
- **Reductions fully reduce into `out[0]`** (harness zeroes it before each
  launch); `sum` / `sumsq` / `mean` only (`atomicAdd(float)` is native on sm_86).
  `max` / `argmax` deferred with transpose to the later extension.
- **`warp_divergence`** uses its own degree-12 polynomial op family (coeffs shrink
  as `c_k / 2^k` to keep `|P(x)|` ~ O(1)); the naive splits each warp's lanes
  between an fma chain and a mul/add chain computing the same polynomial.

### Launch-config convention

NVRTC has no host code, so each kernel **declares its launch config in a directive
comment** the harness parses (first match wins, defaults applied if absent):

```
// launch: grid=stride block=256      // fixed grid = 32*SM, grid-stride loop
// launch: grid=n_div  block=256      // grid = ceil(n / block), 1 elem/thread
// launch: grid=blocks:1 block=32     // explicit block count
```

- `block` ∈ powers of 2, 32…1024.
- Naive and agent kernels are each launched per **their own** declared config;
  the baseline's bad config is part of the seeded inefficiency
  (`poor_block_grid` naive declares `grid=blocks:1 block=32`).
- Reduction entries always fully reduce into `out[0]`; harness zeroes `out[0]`
  before every launch. Naive `global_atomics` form: every thread
  `atomicAdd(&out[0], x[i])`. Optimized: shared-mem tree reduction, one
  `atomicAdd` per block.

### Known-good reference solution + cheap pre-filtering

The generator emits, alongside `naive_src`, an **`optimized_src`** that fixes the
seeded inefficiency. `validate(task)` (deterministic given seeds):
1. compile `naive_src` and `optimized_src`,
2. run the full correctness suite on `optimized_src`,
3. benchmark `optimized_src` vs `naive_src`,
4. return `solvable = correctness_pass AND speedup >= target`.

`load_environment(prevalidate=True)` and `scripts/generate_tasks.py` use this to
drop unsolvable `(category, difficulty, seed)` triples before they reach a model.
Results cache to `scripts/_task_cache.json` keyed by `(cat, diff, seed, srchash)`.

---

## 5. Environment implementation

`StatefulToolEnv` subclass, in-memory per-rollout state only.

### State (per rollout, built in `setup_state` from `state["info"]`)
- `kt`: the generated `KernelTask` (naive src, optimized src [hidden], op spec,
  size range, entry name/signature, atol/rtol, target).
- `current_src`: `str`, starts = `kt.naive_src`.
- `benchmark_calls_used`, `benchmark_budget`.
- `best_score`, `best_speedup`, `last_correctness` (dict or None),
  `num_compiles`, `compiled_ok_ever`.
- `worker`: `WorkerClient | None` (lazy spawn on first GPU tool call; killed in
  `@vf.cleanup`).

`max_turns` = assistant turns; `turns_remaining = max_turns - len(state["trajectory"])`.
Every tool result ends with `\n[turns remaining: k | benchmark calls left: b]`.

### Tools (all `async`, `state` hidden via `args_to_skip=["state"]`)

| tool | args | effect |
|------|------|--------|
| `view_kernel` | – | returns the current kernel source + full task spec (category, difficulty, seeded-inefficiency name + description, size range, entry signature, launch directive help, atol/rtol, target speedup) + budgets. |
| `submit_kernel` | `source: str` | replaces `current_src` in state (no compile). Basic guard: must contain `__global__` and the entry name. Returns confirmation + a diff-ish size note. |
| `test_correctness` | – | worker `test`: compile `current_src`; run `n_cases` cases at spread sizes/seeds; return per-case `passed`, `max_abs_err`, `max_rel_err`; on compile failure return the NVRTC log. Updates `last_correctness`. Unlimited. |
| `benchmark` | – | costs 1 budget. worker `benchmark`: compile; warmup+timed agent & baseline on `seed+2000`; post-check on `seed+3000`; scaling on `size` & `size//4`. Returns median/trimmed-mean ms for both, speedup, validity breakdown, and — if valid — the updated `best_score`. Refuses when budget exhausted or last compile failed-known. |
| `check_score` | – | current `best_score`, `best_speedup`, whether a valid benchmark exists, `last_correctness` summary, budgets, turns. |

### Rubric
- `reward_score` (weight 1.0) → `state["best_score"]`.
- metrics (weight 0): `best_speedup`, `correctness_passed` (0/1),
  `num_benchmarks`, `num_compiles`, `compiled_ok` (0/1), `hack_flags`
  (count of times a benchmark was rejected for post-check/scaling/checksum).

### GPU worker (`gpu_worker.py`, run as `python -m cuda_kernel_opt.gpu_worker`)
- scrub CUDA_PATH/HOME/ROOT; `import cupy`; warn-filter the path warning.
- stdin/stdout **JSON lines**, one request → one response, `flush` after each.
- commands: `ping`, `test`, `benchmark`, `validate`. Each fully self-contained
  (all seeds/sizes/sources in the request).
- compile via `cp.RawModule(code=src, backend="nvrtc", options=("--std=c++17", f"--gpu-architecture=compute_86"))`; catch `CompileException` → structured `{compiled:false, log}`.
- timing: `cp.cuda.Event` pairs, per-iteration ms into a list; report `median`,
  `trimmed_mean` (drop 20% each end), `min`; `W` warmup + `T` timed
  (defaults W=5, T=30, scaled down for the largest sizes).
- reference: cupy high-level in float64.
- hard wall: parent enforces `worker_timeout` per request; on timeout it
  `kill()`s and respawns, tool returns a timeout error (covers infinite-loop
  kernels; WDDM TDR also recovers the device).

### `WorkerClient` (`worker_client.py`)
async: `start()`, `request(dict, timeout) -> dict`, `close()`. Uses
`asyncio.create_subprocess_exec`, `readline` with `asyncio.wait_for`. On timeout
or dead pipe: kill, mark stale; next `request` respawns.

---

## 6. Standalone TUI (`scripts/tui.py`)

Rich (static render; `--watch` re-renders). Design language mirrors
`prime eval tui` / the calendar-scheduling example: rounded panels, `box.SIMPLE_HEAD`
tables, dim labels + bold values, green/yellow/red status.

`python scripts/tui.py --category reduction --difficulty medium --seed 3`

Renders:
- header: env id, task id (`cat/diff/seed`), size range, entry signature, atol/rtol, target.
- panel **Seeded inefficiency**: name + human description + what a good fix does.
- panel **Naive kernel** (`Syntax`, `cuda`), with its launch directive highlighted.
- panel **Reference / optimized kernel** (`Syntax`).
- panel **Reference semantics**: the CuPy expression as text.
- **Status** table (runs the worker): naive compiles ✓/✗ · naive correct vs ref ✓/✗
  · baseline median ms · optimized median ms · **speedup** · target · scaling ratio
  · **solvable** verdict.
- `--list` prints the full taxonomy; `--sweep` runs every (cat,diff,seed<N) and
  prints a solvability table.

---

## 7. File layout

```
cuda_kernel_opt/
    __init__.py         # load_environment, CudaKernelOptEnv, tools, rubric
    taskgen.py          # KernelTask, kernel templates, taxonomy, dataset builder
    scoring.py          # constants, score(), rubric reward/metric fns
    gpu_worker.py       # cupy subprocess worker
    worker_client.py    # async client
  scripts/
    tui.py
    generate_tasks.py
  tests/
    test_taskgen.py     # templates parse, directives valid, dataset shape
    test_worker.py      # ping, compile ok/fail, correctness, benchmark, timeout  [GPU]
    test_scoring.py     # gate + formula edge cases
    test_env.py         # load_environment, setup_state, one scripted rollout     [GPU]
  pyproject.toml
  README.md
  PLAN.md
```

---

## 8. Test plan

### Unit (no GPU)
- `taskgen`: every `(category, difficulty, seed in 0..30)` produces a `KernelTask`
  whose `naive_src` & `optimized_src` contain `extern "C" __global__ void <entry>`,
  a parseable `// launch:` directive, and matching signatures. Size within range.
  Op spec round-trips. Dataset rows have `prompt` + JSON `info` with the triple.
- `scoring`: gate returns 0 on any failed correctness / invalid benchmark;
  formula hits 0 at s=1, 1 at s=target, clamps; `best_score` is monotone.
- directive parser: valid/invalid strings, defaults.

### Integration (GPU, marked `@pytest.mark.gpu`, skipped if `cupy` import/gpu fails)
- worker `ping` round-trips.
- compile failure returns `compiled:false` + non-empty log (feed a syntax error).
- naive kernel of each template compiles and is **correct vs reference**.
- `optimized` of each template compiles, is correct, and beats naive by
  `>= target` for its tier → i.e. every generated task is solvable
  (this is also the pre-filter).
- dead-code-elimination probe: a "kernel" that does `if(i==0) out[0]=ref;` (writes
  one element) → correctness FAILS (other cases/elements wrong) → score 0.
- flat-timing probe: a kernel that early-returns for `i>0` → scaling ratio ~1 →
  benchmark `invalid` → score 0 even though a subset matches.
- timeout probe: `while(true);` kernel → worker killed, tool returns timeout,
  rollout survives, score 0.
- state isolation: two sequential rollouts on the same env instance don't share
  worker/buffers; second starts from naive again.

### End-to-end eval
- `prime eval run cuda-kernel-opt -n 2 -r 1` against a Prime Inference model from
  `configs/` once the above pass. Inspect with `prime eval tui`. Expect: some
  rollouts reach score > 0, no crashes, metrics populated, `hack_flags` == 0 for
  honest runs.
- Small sweep across tiers for a sanity gradient (easy easier than hard).

### Commands
```
# unit
uv run pytest tests -k "not gpu" -q
# gpu
uv run pytest tests -q
# tui smoke
uv run python scripts/tui.py --category elementwise --difficulty easy --seed 0
uv run python scripts/tui.py --sweep 4
# solvability pre-filter
uv run python scripts/generate_tasks.py --out taskset.json
# eval
prime env install cuda-kernel-opt
prime eval run cuda-kernel-opt -n 2 -r 1
```

---

## 9. Milestones

- [x] **M1 — skeleton**: package layout, `pyproject` deps, `taskgen`
      (elementwise + reduction templates, taxonomy, directive parser, dataset
      builder), `scoring`. Unit tests green (`test_taskgen`, `test_scoring`).
- [x] **M2 — GPU worker**: `gpu_worker` + `worker_client` (async) +
      `SyncWorkerClient`; `ping` / `test` / `benchmark` / `validate`; adaptive
      timed-iteration counts; timeout → kill + respawn. `test_worker` green;
      every generated task solvable (48/48 on an 8-seed sweep).
- [x] **M3 — environment**: `CudaKernelOptEnv` (`StatefulToolEnv`), 5 tools,
      rubric (`reward` + 7 metrics), per-rollout worker with `max_concurrent_gpu`
      / `max_live_workers` semaphores, `@vf.cleanup` teardown, `load_environment`
      (+ `prevalidate`). `test_env` green incl. cheat→0, budget exhaustion,
      hack-flag path, state isolation.
- [x] **M4 — TUI + generate script**: `scripts/tui.py` (single task / `--sweep` /
      `--list` / `--no-gpu`), `scripts/generate_tasks.py` (cache +
      `taskset.json`). README written.
- [~] **M5 — evals**: `prime env install` OK; `prime eval run` reaches inference
      then fails **`Payment required — insufficient balance`** on the Prime
      account, and no other provider key is present. Full end-to-end is instead
      covered by scripted-policy tests (`test_env.py::test_full_trajectory_*`,
      `test_reference_solution_scores_full`). Needs a funded key to run a real
      model; budgets/targets already tuned from the `generate_tasks` sweep.

## 10. Open questions / risks

- **Live eval is blocked on inference credit.** `prime eval run cuda-kernel-opt
  -m <model>` is the last step; it will work once the Prime balance is topped up
  (or `OPENAI_API_KEY` / `-p openai` etc. is available). Env loading + dataset
  build + env-server startup were all exercised by that run before it hit the
  billing error.
- Concurrency vs. 6 GB VRAM: each rollout spawns its own GPU worker (state
  isolation). `max_live_workers` (default 4) caps live CUDA contexts and
  `max_concurrent_gpu` (default 1) serialises GPU compute across rollouts. Run
  `prime eval run` with `--num-workers 1` and a modest `-c`; the worker returns
  OOM as a normal tool error rather than crashing.
- WDDM TDR (~2 s) also resets genuinely slow legit kernels; the size clamps keep
  the pathological baselines short enough that this hasn't triggered.
- NVRTC 12.9 + `--gpu-architecture=compute_86` JIT is ~100–300 ms per compile,
  paid every `test_correctness` / `benchmark` (not cached across rollouts by
  design). First tool call also pays ~1–2 s CUDA context init.
- `warp_divergence` / `uncoalesced` elementwise fixes are "spot the trick":
  either the agent sees that one branch / the identity permutation suffices (full
  ~2.5–5x) or it does not (0). Little *intermediate* gradient — the reward is
  close to bimodal on elementwise. Reduction has a real ramp (interleaved ≈4x <
  sequential ≈5x < grid-stride reference ≈10x). Genuinely graded elementwise
  variants (compound inefficiencies, larger f) are a follow-up.
- `no_vectorized_loads` / `unnecessary_syncthreads` can't hit a ≥1.5x target on
  this GPU; revisit with a per-inefficiency sub-1.5x target or a faster card.

## Changelog
- _init_: doc created from `prompt.md` + toolchain investigation (CuPy/NVRTC 12.9
  path chosen after the system nvcc/driver mismatch was ruled out — CUDA 13.3
  toolkit vs. a 12.9-capable driver).
- _M1–M4_: implemented in full. Key deviations from the sketch, all forced by
  what actually produces a ≥target speedup on the RTX 3060 Laptop: dropped
  `no_vectorized_loads`, shelved `unnecessary_syncthreads`, clamped sizes to
  2^20–2^23, targets set to 1.5 / 2.0 / 2.3, and the timing-scaling gate replaced
  by a physical memory-bandwidth floor + a lenient monotonicity check (the ratio
  heuristic false-positived on honest fast kernels). 180 tests pass
  (120 unit + 60 GPU).
- _M5_: blocked on Prime inference balance; scripted-policy coverage stands in.
- _2026-09 determinism + saturation pass_:
  * **Cross-process task mismatch fixed.** `generate_task` seeded its RNG with
    `hash((category, difficulty, seed))`, which PEP 456 randomises per process, so
    the eval client (builds the prompt) and the rollout worker (builds the graded
    reference) picked different inefficiency / op / size for the same triple —
    unwinnable tasks. Replaced with a `blake2b` digest (`_stable_seed`);
    regression test `test_deterministic_across_processes` spawns interpreters with
    different `PYTHONHASHSEED`. Audited the package: no other `hash()` or
    process-varying source feeds seeding or IDs.
  * **Reward de-saturated.** Baselines were pathological (single-warp launch;
    per-element global atomics) so every competent kernel scored exactly 1.0 with
    20–250x margin and zero gradient. Reduction now uses one
    `no_shared_memory_reuse` baseline (grid-stride register accumulate + one
    `atomicAdd`/thread, ~8–14x achievable) clamped to ≤2^21; `poor_block_grid`
    dropped from active tiers. `TARGET_SPEEDUP` is now per `[category][difficulty]`
    (elementwise 1.7/1.7/1.9, reduction 3.0/5.0/6.0), set to ~⅓–½ of the reference
    ceiling. 185 tests pass; 48/48 solvability.
