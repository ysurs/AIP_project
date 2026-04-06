import numpy as np          # used ONLY for ctypes data transfer to solve_graph_cut
from maxflow_solver import solve_graph_cut

def _dt_manhattan(hole, h, w):
    """
    4-pass Manhattan distance transform, pure Python.
    hole : 2-D Python list — values > 127 are the seed (distance = 0).
    Returns a flat Python list of length h*w.
    """
    INF  = float(h + w + 1)
    dist = [0.0 if hole[r][c] > 127 else INF for r in range(h) for c in range(w)]

    # Forward: left → right
    for r in range(h):
        base = r * w
        for c in range(1, w):
            v = dist[base + c - 1] + 1.0
            if v < dist[base + c]:
                dist[base + c] = v
    # Forward: top → bottom
    for r in range(1, h):
        base = r * w
        for c in range(w):
            v = dist[base - w + c] + 1.0
            if v < dist[base + c]:
                dist[base + c] = v
    # Backward: right → left
    for r in range(h):
        base = r * w
        for c in range(w - 2, -1, -1):
            v = dist[base + c + 1] + 1.0
            if v < dist[base + c]:
                dist[base + c] = v
    # Backward: bottom → top
    for r in range(h - 2, -1, -1):
        base = r * w
        for c in range(w):
            v = dist[base + w + c] + 1.0
            if v < dist[base + c]:
                dist[base + c] = v
    return dist


# ─────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────

def find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop,
                      first_component=False):
    h, w = q_crop.shape[:2]

    # Convert inputs to Python lists (numpy used only for data transfer)
    q    = q_crop.tolist()
    m    = m_crop.tolist()
    hole = hole_mask_crop.tolist()
    ctx  = context_mask_crop.tolist()

    # --- 1. Per-pixel SSD (the diff map used by the paper) ---
    # SSD(p) = ||I_query(p) - I_patch(p)||^2  (sum of squared channel diffs)
    ssd = [
        (q[r][c][0]-m[r][c][0])**2 +
        (q[r][c][1]-m[r][c][1])**2 +
        (q[r][c][2]-m[r][c][2])**2
        for r in range(h) for c in range(w)
    ]

    # Distance transform from hole pixels (raw pixel units, not normalised)
    # Used for the unary penalty on context pixels
    dist_raw = _dt_manhattan(hole, h, w)

    # --- 2. Build edge lists ---
    num_pixels = h * w
    SOURCE     = num_pixels
    SINK       = num_pixels + 1
    num_nodes  = num_pixels + 2
    INF        = 1e9
    K          = 0.002   # empirical constant from Eq. 2

    from_list, to_list, fwd_list, rev_list = [], [], [], []

    # Symmetric pairwise edges (4-connected)
    for r in range(h):
        base = r * w
        for c in range(w - 1):            # horizontal
            u  = base + c
            v  = base + c + 1
            wt = abs(ssd[v] - ssd[u]) + 1e-5
            from_list.append(u); to_list.append(v)
            fwd_list.append(wt); rev_list.append(wt)
    for r in range(h - 1):
        base = r * w
        for c in range(w):               # vertical
            u  = base + c
            v  = base + w + c
            wt = abs(ssd[v] - ssd[u]) + 1e-5
            from_list.append(u); to_list.append(v)
            fwd_list.append(wt); rev_list.append(wt)

    # Directed terminal (unary) edges
    for r in range(h):
        for c in range(w):
            i       = r * w + c
            is_hole = hole[r][c] > 127
            is_ctx  = ctx[r][c] > 127
            is_out  = not is_hole and not is_ctx

            if is_hole:                  # Cd(p, exist) = INF — must be from patch
                from_list.append(SOURCE); to_list.append(i)
                fwd_list.append(INF);     rev_list.append(0.0)
            elif is_out:                 # Cd(p, patch) = INF — must stay in query
                from_list.append(i);      to_list.append(SINK)
                fwd_list.append(INF);     rev_list.append(0.0)
            else:                        # context: Cd(p, patch) = (k*Dist)^3  — Eq. 2
                d    = dist_raw[i]
                cost = (K * d) ** 3
                from_list.append(i);      to_list.append(SINK)
                fwd_list.append(cost);    rev_list.append(0.0)

    # --- 3. Pack for C solver (numpy used only for data transfer) ---
    fa = np.array(from_list, dtype=np.int32)
    ta = np.array(to_list,   dtype=np.int32)
    fw = np.array(fwd_list,  dtype=np.float64)
    rv = np.array(rev_list,  dtype=np.float64)

    flow, segs = solve_graph_cut(fa, ta, fw, rv, num_nodes, SOURCE, SINK)

    # segs == 1 → source side (patch) → mask = 255  (blend src here)
    # segs == 0 → sink  side (query)  → mask = 0    (keep dst here)
    seam_mask = segs[:num_pixels].reshape(h, w).astype(np.uint8) * 255

    if first_component:
        return seam_mask, flow
    return seam_mask