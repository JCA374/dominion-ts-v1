"""ts-v1's C library: c/ts.c (bv-v1's teacher + value-net leaves + successive halving) built into ts.so, loaded through
bv-v1's Python side (bv.core, submodule bv-v1/) with BV_SO pointing at it, so seats, games and the advisor call are
bv-v1's own code."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BV = os.path.join(ROOT, "bv-v1")
ENGINE_C = os.path.join(ROOT, "engine", "c")
SO = os.path.join(ROOT, "ts.so")
SRC = [os.path.join(ENGINE_C, "engine.c"), os.path.join(ROOT, "c", "ts.c")]
DEPS = SRC + [os.path.join(ROOT, "c", "fastnet.h"), os.path.join(ROOT, "rl-v3", "c", "rl.c"),
              os.path.join(ROOT, "rl-v3", "az-v1", "c", "az.c"), os.path.join(ENGINE_C, "engine.h")]


def build(so=SO) -> str:
    if not os.path.exists(so) or max(os.path.getmtime(d) for d in DEPS) > os.path.getmtime(so):
        tmp = f"{so}.{os.getpid()}"
        subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-fno-finite-math-only", "-shared", "-fPIC",
                               "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-unused-function", "-I", ENGINE_C,
                               "-o", tmp, *SRC, "-lm"])
        os.replace(tmp, so)
    return so


if "bv.core" not in sys.modules:
    os.environ.setdefault("BV_SO", build())
if BV not in sys.path:
    sys.path.insert(0, BV)
from bv import core  # noqa: E402

lib = core.lib
lib.ts_set_leaf.argtypes = [ctypes.c_int]
lib.ts_set_alloc.argtypes = [ctypes.c_int]
lib.ts_leaf_cuts.restype = ctypes.c_long
lib.ts_sh_final.restype = ctypes.c_long


def configure(leaf=0, alloc=0, **kw):
    """bv-v1's teacher settings (core.configure) + ts-v1's: leaf = turns before a value-net leaf (0 = to the end),
    alloc = 0 uniform (bv-v1), 1 successive halving."""
    core.configure(**kw)
    lib.ts_set_leaf(int(leaf))
    lib.ts_set_alloc(int(alloc))
