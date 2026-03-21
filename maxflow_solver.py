"""
maxflow_solver.py
-----------------
Thin ctypes wrapper around graph_cut_solver.c (Dinic's max-flow).

Auto-compiles the shared library on first import if needed.
Build command used:
    gcc -O3 -shared -fPIC -o graph_cut_solver.so graph_cut_solver.c
"""

import ctypes
import os
import subprocess

import numpy as np

_LIB = None


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB

    _dir = os.path.dirname(os.path.abspath(__file__))
    so   = os.path.join(_dir, "graph_cut_solver.so")
    c    = os.path.join(_dir, "graph_cut_solver.c")

    # Re-compile if .so is missing or older than the source
    needs_build = not os.path.exists(so)
    if not needs_build and os.path.exists(c):
        needs_build = os.path.getmtime(c) > os.path.getmtime(so)

    if needs_build:
        ret = subprocess.run(
            ["gcc", "-O3", "-shared", "-fPIC", "-o", so, c],
            capture_output=True, text=True,
        )
        if ret.returncode != 0:
            raise RuntimeError(
                f"[maxflow_solver] gcc failed:\n{ret.stderr}"
            )
        print(f"[maxflow_solver] compiled {so}")

    lib = ctypes.CDLL(so)
    lib.solve_maxflow.restype  = ctypes.c_double
    lib.solve_maxflow.argtypes = [
        ctypes.c_int,                    # num_nodes
        ctypes.c_int,                    # source
        ctypes.c_int,                    # sink
        ctypes.POINTER(ctypes.c_int),    # from_arr
        ctypes.POINTER(ctypes.c_int),    # to_arr
        ctypes.POINTER(ctypes.c_double), # fwd_cap
        ctypes.POINTER(ctypes.c_double), # rev_cap
        ctypes.c_int,                    # num_edges
        ctypes.POINTER(ctypes.c_int),    # segs_out
    ]
    _LIB = lib
    return lib


def _ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


def solve_graph_cut(from_arr, to_arr, fwd_cap, rev_cap, num_nodes, source, sink):
    """
    Run Dinic's max-flow / min-cut via the compiled C solver.

    Parameters
    ----------
    from_arr, to_arr : int32 arrays  — edge tail/head indices
    fwd_cap          : float64 array — forward edge capacities
    rev_cap          : float64 array — reverse edge capacities
                       (same as fwd_cap for symmetric/undirected edges,
                        0 for directed terminal edges)
    num_nodes        : int — total node count (pixels + 2 for source/sink)
    source, sink     : int — terminal node indices

    Returns
    -------
    flow     : float   — max-flow value (== min-cut cost)
    segments : int32 ndarray [num_nodes]
               1 = node is on the source side (patch), 0 = sink side (query)
    """
    lib  = _load()
    segs = np.zeros(num_nodes, dtype=np.int32)

    flow = lib.solve_maxflow(
        num_nodes, source, sink,
        _ptr(from_arr, ctypes.c_int),
        _ptr(to_arr,   ctypes.c_int),
        _ptr(fwd_cap,  ctypes.c_double),
        _ptr(rev_cap,  ctypes.c_double),
        len(from_arr),
        _ptr(segs,     ctypes.c_int),
    )
    return float(flow), segs
