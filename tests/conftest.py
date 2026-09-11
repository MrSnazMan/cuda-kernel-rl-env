"""Shared fixtures. GPU tests are skipped automatically when CuPy or the device
is unavailable so the unit suite still runs anywhere."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def _gpu_available() -> bool:
    env = dict(os.environ)
    for v in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        env.pop(v, None)
    code = (
        "import cupy as cp; cp.cuda.runtime.getDeviceCount(); "
        "m=cp.RawModule(code='extern \"C\" __global__ void k(float* o){o[0]=1.f;}'); "
        "m.get_function('k')((1,),(1,),(cp.zeros(1,dtype=cp.float32),)); "
        "cp.cuda.Device().synchronize()"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, timeout=90)
        return r.returncode == 0
    except Exception:
        return False


GPU_OK = _gpu_available()


def pytest_collection_modifyitems(config, items):
    if GPU_OK:
        return
    skip = pytest.mark.skip(reason="no working CuPy + CUDA 12 GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


class WorkerProc:
    """Blocking JSON-lines client for the GPU worker, for tests."""

    def __init__(self):
        self.p = subprocess.Popen(
            [sys.executable, "-u", "-m", "cuda_kernel_opt.gpu_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def call(self, obj: dict) -> dict:
        assert self.p.stdin and self.p.stdout
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        try:
            if self.p.stdin:
                self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


@pytest.fixture(scope="module")
def worker():
    w = WorkerProc()
    yield w
    w.close()
