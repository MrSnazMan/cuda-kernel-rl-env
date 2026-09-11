"""Windows compatibility shims for the verifiers / prime toolchain.

This module is imported automatically at interpreter startup by the companion
``_vf_win_compat.pth`` file, which the ``cuda-kernel-opt`` wheel installs into
site-packages. It is a no-op on non-Windows platforms and is written to never
raise out of interpreter start-up.

It papers over three places where ``verifiers`` 0.1.x (the release ``prime``
pins on Python 3.10) assumes a POSIX host:

1. **``import fcntl``** -- ``verifiers`` does an unconditional ``import fcntl``
   deep in the lazy import chain behind ``verifiers.load_environment``
   (``verifiers.v1`` -> experimental harness utils -> ``file_locks``). ``fcntl``
   is Unix-only, so on Windows every ``prime eval`` / ``prime gepa`` / ``prime
   rl`` fails at attribute-access time with the misleading message
   ``To use verifiers.load_environment, install as verifiers[all]``. We register
   a no-op ``fcntl`` stub. Nothing on the eval/train code path actually calls
   ``fcntl.flock`` -- that file-locking helper is only pulled in transitively --
   so the stub is safe; the only downside is that the experimental git-checkout
   cache would not be crash-safe under concurrent processes, which this
   environment never exercises.

2. **``ipc://`` ZMQ transport** -- ``verifiers``' out-of-process env server
   binds its router<->worker sockets on ``ipc:///tmp/vf-...``. ZeroMQ ``ipc://``
   is not supported by the Windows ``pyzmq`` wheel (``zmq.error.ZMQError:
   Protocol not supported``), so the server never becomes healthy and the eval
   times out after ten minutes. We rewrite ``make_ipc_address`` to hand out
   ``tcp://127.0.0.1:<port>`` addresses instead, and make the router's
   shutdown-time ``os.unlink`` of those "paths" non-fatal.  (The env server on
   Windows additionally needs ``tornado>=6.1`` for ``zmq.asyncio`` on the
   Proactor event loop; that is declared as a Windows-only dependency of this
   package.)

3. **Console ``UnicodeEncodeError``** -- ``prime`` and ``verifiers`` print
   non-cp1252 glyphs (``check`` marks, box drawing). When stdout is a pipe or a
   redirected file the default Windows encoding is cp1252 and the process dies
   with ``UnicodeEncodeError``. We switch stdio to UTF-8.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":  # pragma: no cover - exercised only on Windows
    # ------------------------------------------------------------------ #
    # 3. UTF-8 stdio                                                     #
    # ------------------------------------------------------------------ #
    for _name in ("stdout", "stderr"):
        _stream = getattr(sys, _name, None)
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 1. no-op ``fcntl``                                                 #
    # ------------------------------------------------------------------ #
    if "fcntl" not in sys.modules:
        try:
            import fcntl  # noqa: F401  (real module, e.g. under Cygwin/MSYS)
        except ModuleNotFoundError:
            import types as _types

            _fcntl = _types.ModuleType("fcntl")
            _fcntl.LOCK_SH, _fcntl.LOCK_EX, _fcntl.LOCK_NB, _fcntl.LOCK_UN = 1, 2, 4, 8
            _fcntl.F_GETFD, _fcntl.F_SETFD, _fcntl.F_GETFL, _fcntl.F_SETFL = 1, 2, 3, 4
            _fcntl.flock = lambda *a, **k: None
            _fcntl.lockf = lambda *a, **k: None
            _fcntl.fcntl = lambda *a, **k: 0
            _fcntl.ioctl = lambda *a, **k: 0
            sys.modules["fcntl"] = _fcntl

    # ------------------------------------------------------------------ #
    # 2. tcp:// instead of ipc:// for the verifiers env server          #
    # ------------------------------------------------------------------ #
    import importlib.abc
    import importlib.util

    _addr_cache: dict = {}

    def _tcp_ipc_address(session_id: str, name: str) -> str:
        """Drop-in replacement for ``verifiers.utils.serve_utils.make_ipc_address``.

        Returns a loopback TCP endpoint (which ZMQ supports everywhere) instead
        of an ``ipc://`` one, cached per ``(session_id, name)`` so repeated calls
        for the same socket are stable within a process.
        """
        key = (session_id, name)
        addr = _addr_cache.get(key)
        if addr is None:
            import socket

            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            finally:
                probe.close()
            addr = f"tcp://127.0.0.1:{port}"
            _addr_cache[key] = addr
        return addr

    class _OsUnlinkSafe:
        """Proxy for the ``os`` module that never raises from ``unlink``.

        The env router cleans up by calling ``os.unlink`` on each address with
        the ``ipc://`` prefix stripped; with TCP addresses that argument is not
        a filesystem path and Windows raises ``OSError`` (not the
        ``FileNotFoundError`` the router catches). Everything else forwards to
        the real module.
        """

        def __init__(self, real):
            self._real = real

        def __getattr__(self, item):
            return getattr(self._real, item)

        def unlink(self, path, *args, **kwargs):
            try:
                return self._real.unlink(path, *args, **kwargs)
            except OSError:
                return None

    def _patch_serve_utils(module) -> None:
        module.make_ipc_address = _tcp_ipc_address

    def _patch_env_router(module) -> None:
        module.make_ipc_address = _tcp_ipc_address
        import os as _os

        if not isinstance(getattr(module, "os", None), _OsUnlinkSafe):
            module.os = _OsUnlinkSafe(_os)

    _POST_IMPORT_PATCHES = {
        "verifiers.utils.serve_utils": _patch_serve_utils,
        "verifiers.serve.server.env_router": _patch_env_router,
    }

    class _VerifiersWinPatchFinder(importlib.abc.MetaPathFinder):
        """Patches select ``verifiers`` modules immediately after they load."""

        def find_spec(self, fullname, path, target=None):
            patch = _POST_IMPORT_PATCHES.get(fullname)
            if patch is None:
                return None
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                found = getattr(finder, "find_spec", None)
                if found is None:
                    continue
                spec = found(fullname, path, target)
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return None

            _real_exec = spec.loader.exec_module

            def exec_module(module, _real_exec=_real_exec, _patch=patch):
                _real_exec(module)
                try:
                    _patch(module)
                except Exception:
                    pass

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    if not any(isinstance(f, _VerifiersWinPatchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _VerifiersWinPatchFinder())

    # Cover the case where a target module is already imported (e.g. the parent
    # process imported verifiers before this shim ran).
    for _mod_name, _patch_fn in _POST_IMPORT_PATCHES.items():
        _existing = sys.modules.get(_mod_name)
        if _existing is not None:
            try:
                _patch_fn(_existing)
            except Exception:
                pass
