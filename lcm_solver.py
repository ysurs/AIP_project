"""
lcm_solver.py
-------------
Thin ctypes wrapper around lcm_solver.c.
Auto-compiles the shared library on first import if needed.
Build command used:
    gcc -O3 -march=native -shared -fPIC -o lcm_solver_lib.so lcm_solver.c -lm
"""

import ctypes
import os
import subprocess

import numpy as np  # used only in return statements

_LIB = None


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB

    _dir = os.path.dirname(os.path.abspath(__file__))
    so   = os.path.join(_dir, "lcm_solver_lib.so")
    c    = os.path.join(_dir, "lcm_solver.c")

    # Re-compile if .so is missing or older than the source
    needs_build = not os.path.exists(so)
    if not needs_build and os.path.exists(c):
        needs_build = os.path.getmtime(c) > os.path.getmtime(so)

    if needs_build:
        ret = subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC",
             "-o", so, c, "-lm"],
            capture_output=True, text=True,
        )
        if ret.returncode != 0:
            raise RuntimeError(f"[lcm_solver] gcc failed:\n{ret.stderr}")
        print(f"[lcm_solver] compiled {so}")

    lib = ctypes.CDLL(so)

    # --- lcm_bgr_to_gray ---
    lib.lcm_bgr_to_gray.restype  = None
    lib.lcm_bgr_to_gray.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),   # bgr
        ctypes.c_int,                     # h
        ctypes.c_int,                     # w
        ctypes.POINTER(ctypes.c_float),   # out
    ]

    # --- lcm_bgr_to_lab ---
    lib.lcm_bgr_to_lab.restype  = None
    lib.lcm_bgr_to_lab.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),   # bgr
        ctypes.c_int,                     # h
        ctypes.c_int,                     # w
        ctypes.POINTER(ctypes.c_float),   # out
    ]

    # --- lcm_dilate ---
    lib.lcm_dilate.restype  = None
    lib.lcm_dilate.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),   # mask_in
        ctypes.c_int,                     # h
        ctypes.c_int,                     # w
        ctypes.c_int,                     # radius
        ctypes.POINTER(ctypes.c_uint8),   # mask_out
    ]

    # --- lcm_resize ---
    lib.lcm_resize.restype  = None
    lib.lcm_resize.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # src
        ctypes.c_int,                     # sh
        ctypes.c_int,                     # sw
        ctypes.POINTER(ctypes.c_float),   # dst
        ctypes.c_int,                     # dh
        ctypes.c_int,                     # dw
        ctypes.c_int,                     # nc
    ]

    # --- lcm_texture_map ---
    lib.lcm_texture_map.restype  = None
    lib.lcm_texture_map.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # gray
        ctypes.c_int,                     # h
        ctypes.c_int,                     # w
        ctypes.POINTER(ctypes.c_float),   # out
    ]

    # --- lcm_match_ssd ---
    lib.lcm_match_ssd.restype  = None
    lib.lcm_match_ssd.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # search
        ctypes.c_int,                     # sh
        ctypes.c_int,                     # sw
        ctypes.POINTER(ctypes.c_float),   # templ
        ctypes.c_int,                     # th
        ctypes.c_int,                     # tw
        ctypes.POINTER(ctypes.c_float),   # mask
        ctypes.c_int,                     # nc
        ctypes.POINTER(ctypes.c_float),   # out_ssd
    ]

    _LIB = lib
    return lib


def _cbuf_f(n):
    """Allocate a zeroed ctypes float32 buffer of length n."""
    return (ctypes.c_float * n)()


def _cbuf_u(n):
    """Allocate a zeroed ctypes uint8 buffer of length n."""
    return (ctypes.c_uint8 * n)()


# ============================================================
# Public wrapper functions
# ============================================================

def bgr_to_gray(img_bgr):
    """uint8 BGR [H,W,3] -> float32 [H,W]"""
    img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
    h, w = img_bgr.shape[0], img_bgr.shape[1]
    bgr  = (ctypes.c_uint8  * (h * w * 3)).from_buffer_copy(img_bgr)
    out  = _cbuf_f(h * w)
    _load().lcm_bgr_to_gray(bgr, h, w, out)
    return np.array(out, dtype=np.float32).reshape(h, w)


def bgr_to_lab(img_bgr):
    """uint8 BGR [H,W,3] -> float32 Lab [H,W,3]"""
    img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
    h, w = img_bgr.shape[0], img_bgr.shape[1]
    bgr  = (ctypes.c_uint8  * (h * w * 3)).from_buffer_copy(img_bgr)
    out  = _cbuf_f(h * w * 3)
    _load().lcm_bgr_to_lab(bgr, h, w, out)
    return np.array(out, dtype=np.float32).reshape(h, w, 3)


def dilate(mask_u8, radius):
    """uint8 mask [H,W] -> uint8 dilated [H,W]"""
    h, w = mask_u8.shape[0], mask_u8.shape[1]
    mask = (ctypes.c_uint8  * (h * w)).from_buffer_copy(mask_u8)
    out  = _cbuf_u(h * w)
    _load().lcm_dilate(mask, h, w, radius, out)
    return np.array(out, dtype=np.uint8).reshape(h, w)


def resize(img, new_h, new_w):
    """float32 [H,W] or [H,W,C] -> float32 [new_h,new_w] or [new_h,new_w,C]"""
    is_2d = len(img.shape) == 2
    if is_2d:
        h, w, nc = img.shape[0], img.shape[1], 1
    else:
        h, w, nc = img.shape[0], img.shape[1], img.shape[2]
    # 2D [H,W] and 3D [H,W,1] share the same flat byte layout, so from_buffer_copy works for both
    src = (ctypes.c_float * (h * w * nc)).from_buffer_copy(img)
    out = _cbuf_f(new_h * new_w * nc)
    _load().lcm_resize(src, h, w, out, new_h, new_w, nc)
    return np.array(out, dtype=np.float32).reshape(new_h, new_w) if is_2d \
        else np.array(out, dtype=np.float32).reshape(new_h, new_w, nc)


def texture_map(img_gray):
    """float32 [H,W] -> float32 [H,W] Sobel+median+normalised"""
    h, w = img_gray.shape[0], img_gray.shape[1]
    gray = (ctypes.c_float * (h * w)).from_buffer_copy(img_gray)
    out  = _cbuf_f(h * w)
    _load().lcm_texture_map(gray, h, w, out)
    return np.array(out, dtype=np.float32).reshape(h, w)


def match_ssd(search, template, mask_1ch):
    """
    Masked SSD template matching.
    search   : float32 [sh,sw] or [sh,sw,nc]
    template : float32 [th,tw] or [th,tw,nc]
    mask_1ch : float32 [th,tw]
    Returns  : float32 [sh-th+1, sw-tw+1]
    """
    sh, sw = search.shape[0],  search.shape[1]
    th, tw = template.shape[0], template.shape[1]
    nc     = search.shape[2] if len(search.shape) == 3 else 1
    # 2D [H,W] and 3D [H,W,1] share the same flat byte layout
    s      = (ctypes.c_float * (sh * sw * nc)).from_buffer_copy(np.ascontiguousarray(search))
    t      = (ctypes.c_float * (th * tw * nc)).from_buffer_copy(np.ascontiguousarray(template))
    m      = (ctypes.c_float * (th * tw)).from_buffer_copy(np.ascontiguousarray(mask_1ch))
    out_h  = sh - th + 1
    out_w  = sw - tw + 1
    out    = _cbuf_f(out_h * out_w)
    _load().lcm_match_ssd(
        s, sh, sw,
        t, th, tw,
        m, nc,
        out,
    )
    return np.array(out, dtype=np.float32).reshape(out_h, out_w)
