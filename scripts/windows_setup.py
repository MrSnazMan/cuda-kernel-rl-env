"""Make the verifiers / prime toolchain work in this venv on Windows.

The ``cuda-kernel-opt`` wheel already ships ``_vf_win_compat.py`` +
``_vf_win_compat.pth`` (see ``_win_compat/`` and the ``force-include`` block in
``pyproject.toml``), which is enough for a normal ``prime env install``. Run this
script when that shim did not land -- e.g. an editable install layout that drops
force-included data, or a venv where ``verifiers`` was installed before the
environment. It:

  * copies the shim pair into this interpreter's site-packages,
  * removes the older stand-alone ``sitecustomize.py`` fcntl shim if present,
  * reports whether ``tornado`` (needed for the Windows env server) is importable.

It is idempotent and safe to re-run. No-op message on non-Windows.

    python scripts/windows_setup.py
"""

from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path

_SHIM_DIR = Path(__file__).resolve().parent.parent / "_win_compat"
_STALE_SITECUSTOMIZE_MARKER = "cuda_kernel_opt windows fcntl shim"


def _remove_stale_sitecustomize(site_dir: Path) -> None:
    target = site_dir / "sitecustomize.py"
    if not target.is_file():
        return
    text = target.read_text(encoding="utf-8")
    if _STALE_SITECUSTOMIZE_MARKER not in text:
        return
    # Strip our old block; drop the file entirely if nothing else remains.
    lines = text.splitlines()
    kept, skipping = [], False
    for line in lines:
        if line.strip().startswith("# >>> cuda_kernel_opt"):
            skipping = True
        if not skipping:
            kept.append(line)
        if line.strip().startswith("# <<< cuda_kernel_opt"):
            skipping = False
    remainder = "\n".join(kept).strip()
    if remainder:
        target.write_text(remainder + "\n", encoding="utf-8")
        print(f"Trimmed stale fcntl shim from {target}")
    else:
        target.unlink()
        print(f"Removed superseded {target}")


def main() -> None:
    if sys.platform != "win32":
        print("Not Windows - nothing to do.")
        return

    site_dir = Path(sysconfig.get_paths()["purelib"])
    for name in ("_vf_win_compat.py", "_vf_win_compat.pth"):
        src = _SHIM_DIR / name
        dst = site_dir / name
        shutil.copyfile(src, dst)
        print(f"Installed {dst}")

    _remove_stale_sitecustomize(site_dir)

    try:
        import tornado  # noqa: F401

        print(f"tornado {tornado.version} present (Windows env server OK).")
    except Exception:
        print(
            "WARNING: tornado is not importable. The verifiers env server will "
            "not start on Windows without it. Install with:\n"
            "    uv pip install 'tornado>=6.1'\n"
            "If that fails with 'os error 32' (OneDrive/Defender file lock), "
            "unpack the wheel by hand:\n"
            "    python -m zipfile -e <tornado-*.whl> "
            + str(site_dir).replace("\\", "/")
        )

    print("\nDone. New Python processes in this venv now load the shim at startup.")


if __name__ == "__main__":
    main()
