"""Terminal viewer for a generated CUDA-kernel-optimisation task.

    python scripts/tui.py --category reduction --difficulty medium --seed 3
    python scripts/tui.py --sweep 4          # solvability table for seeds 0..3
    python scripts/tui.py --list             # the inefficiency taxonomy

Rich static render; design language follows `prime eval tui`: rounded panels,
dim labels + bold values, green/yellow/red status. Runs the GPU worker to fill in
the compile / correctness / timing / solvable columns (skipped with --no-gpu).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ENV_ROOT = Path(__file__).resolve().parent.parent
if str(_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENV_ROOT))

from rich import box  # noqa: E402
from rich.columns import Columns  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from cuda_kernel_opt.taskgen import (  # noqa: E402
    CATEGORIES,
    DIFFICULTIES,
    INEFFICIENCIES,
    KernelTask,
    generate_task,
)
from cuda_kernel_opt.worker_client import SyncWorkerClient, WorkerError  # noqa: E402

console = Console()

ACCENT = "#6cb6ff"
OK = "bold green"
BAD = "bold red"
WARN = "bold yellow"


def _kv(label: str, value: str, value_style: str = "bold") -> Text:
    t = Text()
    t.append(f"{label:<22}", style="dim")
    t.append(value, style=value_style)
    return t


def _task_panel(kt: KernelTask) -> Panel:
    kind = (
        "reduction"
        if kt.category == "reduction"
        else ("elementwise (2 inputs)" if kt.n_inputs == 2 else "elementwise (1 input)")
    )
    body = Group(
        _kv("category", kt.category),
        _kv("difficulty", kt.difficulty),
        _kv("seed", str(kt.seed)),
        _kv("kind", kind),
        _kv("function", kt.op.summary()),
        _kv("size (this instance)", f"{kt.size:,}  (2^{kt.size.bit_length() - 1})"),
        _kv("size range", f"2^{kt.size_min.bit_length() - 1} .. 2^{kt.size_max.bit_length() - 1}"),
        _kv("tolerance", f"atol {kt.atol:g}   rtol {kt.rtol:g}"),
        _kv("target speedup", f"{kt.target_speedup:g}x", value_style=f"bold {ACCENT}"),
        _kv("entry", f"solve({kt.signature})", value_style="cyan"),
    )
    return Panel(body, title="Task", border_style=ACCENT, box=box.ROUNDED)


def _inefficiency_panel(kt: KernelTask) -> Panel:
    info = INEFFICIENCIES[kt.inefficiency]
    body = Group(
        Text(kt.inefficiency, style=f"bold {WARN}"),
        Text(),
        Text(info.description, style="none"),
        Text(),
        Text("What a good fix does:", style="dim"),
        Text(info.fix_hint, style="italic"),
    )
    return Panel(body, title="Seeded inefficiency", border_style=WARN, box=box.ROUNDED)


def _code_panel(title: str, src: str, border: str) -> Panel:
    syntax = Syntax(src.strip(), "cuda", theme="ansi_dark", line_numbers=False, word_wrap=False)
    return Panel(syntax, title=title, border_style=border, box=box.ROUNDED)


def _status_panel(kt: KernelTask, worker: SyncWorkerClient | None) -> Panel:
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False, padding=(0, 1))
    table.add_column("Check", style="dim", no_wrap=True)
    table.add_column("Result")

    if worker is None:
        table.add_row("gpu", Text("skipped (--no-gpu)", style="dim"))
        return Panel(table, title="Status", border_style="dim", box=box.ROUNDED)

    try:
        naive = worker.request(
            {"cmd": "test", "spec": kt.worker_spec(), "kernel_src": kt.naive_src}
        )
        if not naive.get("compiled"):
            table.add_row("naive compiles", Text("NO", style=BAD))
            table.add_row("nvrtc log", Text(naive.get("compile_log", "")[:400], style=BAD))
            return Panel(table, title="Status", border_style=BAD, box=box.ROUNDED)
        table.add_row("naive compiles", Text("yes", style=OK))
        n_pass = sum(c["passed"] for c in naive["cases"])
        table.add_row(
            "naive correct vs ref",
            Text(f"{n_pass}/{len(naive['cases'])} cases", style=OK if naive["all_passed"] else BAD),
        )

        val = worker.request(
            {
                "cmd": "validate",
                "spec": kt.worker_spec(),
                "optimized_src": kt.optimized_src,
                "baseline_src": kt.naive_src,
            }
        )
        bench = worker.request(
            {
                "cmd": "benchmark",
                "spec": kt.worker_spec(),
                "kernel_src": kt.optimized_src,
                "baseline_src": kt.naive_src,
            }
        )
        agent_ms = bench["agent"]["median_ms"]
        base_ms = bench["baseline"]["median_ms"]
        speedup = bench["speedup"]
        table.add_row("naive baseline time", Text(f"{base_ms:.4f} ms", style="bold"))
        table.add_row("reference opt time", Text(f"{agent_ms:.4f} ms", style="bold"))
        table.add_row(
            "reference speedup",
            Text(f"{speedup:.2f}x  (target {kt.target_speedup:g}x)",
                 style=OK if speedup >= kt.target_speedup else WARN),
        )
        pc = bench["post_check"]
        sc = bench["scaling"]
        table.add_row("post-check correct", Text("yes" if pc["passed"] else "NO", style=OK if pc["passed"] else BAD))
        table.add_row("checksum / finite", Text(
            f"{'ok' if pc['checksum_ok'] else 'MISMATCH'} / {'ok' if pc['finite'] else 'NON-FINITE'}",
            style=OK if pc["checksum_ok"] and pc["finite"] else BAD))
        table.add_row("bandwidth floor", Text(
            f"{'ok' if pc.get('bandwidth_floor_ok') else 'BELOW'}  "
            f"({agent_ms:.4f} >= {pc.get('bandwidth_floor_ms', 0):.4f} ms)",
            style=OK if pc.get("bandwidth_floor_ok") else BAD))
        table.add_row("time scales w/ size", Text(
            f"{'ok' if sc['monotonic'] else 'FAIL'}  "
            f"(n/4 {sc['agent_ms'][0]:.4f} -> n {sc['agent_ms'][1]:.4f} ms)",
            style=OK if sc["monotonic"] else BAD))
        solvable = bool(val.get("solvable"))
        table.add_section()
        table.add_row(
            "SOLVABLE",
            Text("YES — a correct, fast enough solution exists" if solvable
                 else f"NO — {val.get('reason', 'unknown')}",
                 style=OK if solvable else BAD),
        )
        border = ACCENT if solvable else BAD
    except WorkerError as e:
        table.add_row("worker error", Text(str(e), style=BAD))
        border = BAD
    return Panel(table, title="Status", border_style=border, box=box.ROUNDED)


def render_task(kt: KernelTask, worker: SyncWorkerClient | None) -> None:
    console.print()
    console.print(Rule(Text(f" CUDA Kernel Opt · {kt.task_id} ", style=f"bold {ACCENT}"), style=ACCENT))
    console.print(Columns([_task_panel(kt), _inefficiency_panel(kt)], expand=True, equal=True))
    console.print(_code_panel("Naive kernel (given to the agent)", kt.naive_src, "yellow"))
    console.print(_code_panel("Reference solution (hidden from the agent)", kt.optimized_src, "green"))
    console.print(_status_panel(kt, worker))
    console.print(Rule(style=ACCENT))


def render_sweep(seeds: int, use_gpu: bool) -> None:
    table = Table(title=f"Solvability sweep (seeds 0..{seeds - 1})", box=box.SIMPLE_HEAD, expand=True)
    for col in ("task", "inefficiency", "op", "size", "naive ok", "ref speedup", "target", "solvable"):
        table.add_column(col, no_wrap=True)
    worker = SyncWorkerClient() if use_gpu else None
    try:
        for cat in CATEGORIES:
            for diff in DIFFICULTIES:
                for seed in range(seeds):
                    kt = generate_task(cat, diff, seed)
                    row = [kt.task_id, kt.inefficiency, kt.op.name, f"2^{kt.size.bit_length() - 1}"]
                    if worker is None:
                        table.add_row(*row, "-", "-", f"{kt.target_speedup:g}x", "-")
                        continue
                    naive = worker.request({"cmd": "test", "spec": kt.worker_spec(), "kernel_src": kt.naive_src})
                    val = worker.request({
                        "cmd": "validate", "spec": kt.worker_spec(),
                        "optimized_src": kt.optimized_src, "baseline_src": kt.naive_src,
                    })
                    naive_ok = naive.get("compiled") and naive.get("all_passed")
                    solv = bool(val.get("solvable"))
                    table.add_row(
                        *row,
                        Text("yes" if naive_ok else "NO", style=OK if naive_ok else BAD),
                        f"{val.get('speedup', 0):.2f}x",
                        f"{kt.target_speedup:g}x",
                        Text("YES" if solv else "no", style=OK if solv else BAD),
                    )
    finally:
        if worker is not None:
            worker.close()
    console.print(table)


def render_list() -> None:
    table = Table(title="Seeded-inefficiency taxonomy", box=box.SIMPLE_HEAD, expand=True)
    for col in ("key", "categories", "tiers", "description"):
        table.add_column(col, no_wrap=(col != "description"))
    for key, info in INEFFICIENCIES.items():
        table.add_row(key, ", ".join(info.categories), ", ".join(info.tiers) or "(unassigned)", info.description)
    console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", choices=CATEGORIES, default="elementwise")
    ap.add_argument("--difficulty", choices=DIFFICULTIES, default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", type=int, metavar="N", help="render a solvability table for seeds 0..N-1")
    ap.add_argument("--list", action="store_true", help="print the inefficiency taxonomy and exit")
    ap.add_argument("--no-gpu", action="store_true", help="skip compiling/timing on the GPU")
    args = ap.parse_args()

    if args.list:
        render_list()
        return
    if args.sweep:
        render_sweep(args.sweep, use_gpu=not args.no_gpu)
        return

    kt = generate_task(args.category, args.difficulty, args.seed)
    worker = None if args.no_gpu else SyncWorkerClient()
    try:
        render_task(kt, worker)
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
