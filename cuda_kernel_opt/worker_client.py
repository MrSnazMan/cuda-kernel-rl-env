"""Async client for the GPU worker subprocess.

One :class:`WorkerClient` per rollout. It spawns ``python -m
cuda_kernel_opt.gpu_worker``, sends one JSON request per line and reads one JSON
response per line, enforcing a wall-clock timeout. On timeout, a dead pipe, or a
response flagged ``fatal`` (device wedged), the process is killed; the next
request transparently respawns it.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

WORKER_MODULE = "cuda_kernel_opt.gpu_worker"


class WorkerError(RuntimeError):
    pass


class WorkerTimeout(WorkerError):
    pass


class WorkerClient:
    def __init__(self, request_timeout: float = 40.0, start_timeout: float = 60.0):
        self.request_timeout = request_timeout
        self.start_timeout = start_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._stderr_tail: list[str] = []
        self._stderr_task: asyncio.Task | None = None

    async def _spawn(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            WORKER_MODULE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_tail = []
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        # first request pays CUDA context init; give it a longer leash
        await self._request_raw({"cmd": "ping"}, timeout=self.start_timeout)

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            async for raw in proc.stderr:
                text = raw.decode(errors="replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
                    del self._stderr_tail[:-20]
        except Exception:
            pass

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        task, self._stderr_task = self._stderr_task, None
        if task is not None:
            task.cancel()
        if proc is None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        await asyncio.sleep(0)  # let the loop run the transport close callbacks

    async def _request_raw(self, req: dict, timeout: float) -> dict:
        proc = self._proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        proc.stdin.write((json.dumps(req) + "\n").encode())
        await proc.stdin.drain()
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError as e:
            await self._kill()
            raise WorkerTimeout(
                f"GPU worker exceeded {timeout:g}s on {req.get('cmd')!r} "
                f"(likely an infinite-loop kernel); it was killed."
            ) from e
        if not line:
            await self._kill()
            tail = " | ".join(self._stderr_tail[-5:])
            raise WorkerError(f"GPU worker exited unexpectedly. stderr: {tail}")
        return json.loads(line.decode())

    async def request(self, req: dict, timeout: float | None = None) -> dict:
        """Send a request, respawning the worker first if needed."""
        async with self._lock:
            if self._proc is None or self._proc.returncode is not None:
                await self._spawn()
            try:
                resp = await self._request_raw(req, timeout or self.request_timeout)
            except WorkerError:
                raise
            if resp.get("fatal"):
                await self._kill()
                raise WorkerError(resp.get("error", "worker reported a fatal error"))
            if not resp.get("ok", False):
                raise WorkerError(resp.get("error", "worker returned ok=false"))
            return resp

    async def close(self) -> None:
        async with self._lock:
            await self._kill()


class SyncWorkerClient:
    """Blocking JSON-lines client — for the TUI and the task-generation script,
    which run outside an event loop."""

    def __init__(self) -> None:
        self._p = subprocess.Popen(
            [sys.executable, "-u", "-m", WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def request(self, req: dict) -> dict:
        assert self._p.stdin and self._p.stdout
        self._p.stdin.write(json.dumps(req) + "\n")
        self._p.stdin.flush()
        line = self._p.stdout.readline()
        if not line:
            raise WorkerError("GPU worker exited unexpectedly")
        resp = json.loads(line)
        if resp.get("fatal"):
            raise WorkerError(resp.get("error", "fatal worker error"))
        return resp

    def close(self) -> None:
        try:
            if self._p.stdin:
                self._p.stdin.close()
            self._p.wait(timeout=10)
        except Exception:
            self._p.kill()

    def __enter__(self) -> "SyncWorkerClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
