import numpy as np          # used ONLY for ctypes data transfer to solve_graph_cut
from maxflow_solver import solve_graph_cut

# def find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop):
#     """
#     Finds the optimal seam between the query and the matched patch using Graph Cuts.
#     All inputs must be exactly the same shape (the cropped bounding box of the context).
    
#     q_crop: BGR query image crop.
#     m_crop: BGR matched candidate patch crop.
#     hole_mask_crop: Grayscale mask (255 = missing region).
#     context_mask_crop: Grayscale mask (255 = context donut).
#     """
#     h, w = q_crop.shape[:2]
    
#     # 1. Calculate Per-Pixel Differences (Smoothness Base)
#     # L2 norm of the difference in color space (can also be done in Lab)
#     q_float = q_crop.astype(np.float32)
#     m_float = m_crop.astype(np.float32)
#     diff = np.sum((q_float - m_float)**2, axis=2)
#     diff = np.sqrt(diff)
    
#     # Optional but recommended paper tweak: add small epsilon so cost is never literally 0
#     diff += 1e-5 

#     # 2. Compute Pairwise Edge Costs (4-connected neighborhood)
#     # Cost(p, q) = diff(p) + diff(q)
#     # Right edges (horizontal connections)
#     cost_right = diff[:, :-1] + diff[:, 1:]
#     # Down edges (vertical connections)
#     cost_down = diff[:-1, :] + diff[1:, :]

#     # 3. Initialize Graph
#     g = maxflow.Graph[float]()
#     nodes = g.add_grid_nodes((h, w))

#     # 4. Add Pairwise (Smoothness) Edges
#     # Add horizontal edges
#     g.add_grid_edges(nodes[:, :-1], cost_right, 
#                      structure=np.array([[0, 0, 0],
#                                          [0, 0, 1],
#                                          [0, 0, 0]]), 
#                      symmetric=True)
    
#     # Add vertical edges
#     g.add_grid_edges(nodes[:-1, :], cost_down, 
#                      structure=np.array([[0, 0, 0],
#                                          [0, 0, 0],
#                                          [0, 1, 0]]), 
#                      symmetric=True)

#     # 5. Add Unary (Terminal) Edges
#     # Capacity to Source (Patch) and Sink (Query Image)
#     INF = 1e10
    
#     # Initialize terminal capacities to 0
#     cap_source = np.zeros((h, w), dtype=np.float32) # Weight to belong to the Patch
#     cap_sink = np.zeros((h, w), dtype=np.float32)   # Weight to belong to the Query
    
#     # Hard constraints based on masks
#     is_hole = hole_mask_crop > 127
#     is_context = context_mask_crop > 127
#     is_outside = ~(is_hole | is_context)
    
#     # Pixels in the hole MUST be from the patch (Source)
#     cap_source[is_hole] = INF
#     cap_sink[is_hole] = 0
    
#     # Pixels outside the context MUST be from the query (Sink)
#     cap_source[is_outside] = 0
#     cap_sink[is_outside] = INF
    
#     # Overlapping context region has 0 terminal cost, allowing pairwise costs to guide the cut
    
#     g.add_grid_tedges(nodes, cap_source, cap_sink)

#     # 6. Run Maxflow / Min-cut
#     g.maxflow()

#     # 7. Extract the Optimal Mask
#     # g.get_grid_segments returns True for nodes connected to the Sink (Query)
#     # We want a mask where 255 = Patch, 0 = Query
#     seam_mask = np.logical_not(g.get_grid_segments(nodes)).astype(np.uint8) * 255
    
#     return seam_mask



# def find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop):

#     h, w = q_crop.shape[:2]



#     # --------------------------------------------------

#     # 1. Compute color difference

#     # --------------------------------------------------

#     qf = q_crop.astype(np.float32)

#     mf = m_crop.astype(np.float32)



#     color_diff = np.sqrt(np.sum((qf - mf) ** 2, axis=2))



#     # --------------------------------------------------

#     # 2. Compute gradient (structure) difference

#     # --------------------------------------------------

#     q_gray = cv2.cvtColor(q_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

#     m_gray = cv2.cvtColor(m_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)



#     qgx = cv2.Sobel(q_gray, cv2.CV_32F, 1, 0, ksize=3)

#     qgy = cv2.Sobel(q_gray, cv2.CV_32F, 0, 1, ksize=3)

#     mgx = cv2.Sobel(m_gray, cv2.CV_32F, 1, 0, ksize=3)

#     mgy = cv2.Sobel(m_gray, cv2.CV_32F, 0, 1, ksize=3)



#     q_mag = np.sqrt(qgx**2 + qgy**2)

#     m_mag = np.sqrt(mgx**2 + mgy**2)



#     grad_diff = np.abs(q_mag - m_mag)



#     # --------------------------------------------------

#     # 3. Combine costs

#     # --------------------------------------------------

#     lambda_grad = 2.0

#     diff = color_diff + lambda_grad * grad_diff



#     diff += 1e-5  # avoid zero weights



#     # --------------------------------------------------

#     # 4. Build graph

#     # --------------------------------------------------

#     g = maxflow.Graph[float]()

#     nodes = g.add_grid_nodes((h, w))



#     # Proper grid edges (4-connected)

#     g.add_grid_edges(nodes, weights=diff, structure=np.array([[0,1,0],

#                                                               [1,0,1],

#                                                               [0,1,0]]),

#                      symmetric=True)



#     # --------------------------------------------------

#     # 5. Terminal costs (IMPORTANT FIX)

#     # --------------------------------------------------

#     INF = 1e9



#     cap_source = np.zeros((h, w), dtype=np.float32)  # patch

#     cap_sink = np.zeros((h, w), dtype=np.float32)    # query



#     is_hole = hole_mask_crop > 127

#     is_context = context_mask_crop > 127

#     is_outside = ~(is_hole | is_context)



#     # Force hole → patch

#     cap_source[is_hole] = INF

#     cap_sink[is_hole] = 0



#     # Force outside → query

#     cap_source[is_outside] = 0

#     cap_sink[is_outside] = INF



#     # Context = free (cut happens here)



#     g.add_grid_tedges(nodes, cap_source, cap_sink)



#     # --------------------------------------------------

#     # 6. Solve

#     # --------------------------------------------------

#     g.maxflow()



#     segments = g.get_grid_segments(nodes)



#     # True = sink (query), False = source (patch)

#     seam_mask = (~segments).astype(np.uint8) * 255



#     return seam_mask

# def find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop, first_component=False):
#     h, w = q_crop.shape[:2]

#     # 1. Compute color difference
#     qf = q_crop.astype(np.float32)
#     mf = m_crop.astype(np.float32)
#     color_diff = np.sqrt(np.sum((qf - mf) ** 2, axis=2))

#     # 2. Compute gradient (structure) difference
#     q_gray = cv2.cvtColor(q_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
#     m_gray = cv2.cvtColor(m_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
#     qgx = cv2.Sobel(q_gray, cv2.CV_32F, 1, 0, ksize=3)
#     qgy = cv2.Sobel(q_gray, cv2.CV_32F, 0, 1, ksize=3)
#     mgx = cv2.Sobel(m_gray, cv2.CV_32F, 1, 0, ksize=3)
#     mgy = cv2.Sobel(m_gray, cv2.CV_32F, 0, 1, ksize=3)
    
#     q_mag = np.sqrt(qgx**2 + qgy**2)
#     m_mag = np.sqrt(mgx**2 + mgy**2)
#     grad_diff = np.abs(q_mag - m_mag)

#     # --------------------------------------------------
#     # 3. Combine costs
#     # --------------------------------------------------
#     weight_grad = 2.0
    
#     hole_inv = (~(hole_mask_crop > 127)).astype(np.uint8) * 255
#     dist = cv2.distanceTransform(hole_inv, cv2.DIST_L2, 5)
#     if np.max(dist) > 0:
#         dist = dist / np.max(dist)
        
#     # THE FIX: Massively increase the distance weight. 
#     # color_diff can easily be 100-200+, so this needs to be comparable.
#     weight_dist = 300.0 
    
#     # Optional but highly recommended: square the distance so the penalty
#     # is low near the hole, but curves upward aggressively near the edges.
#     diff = color_diff + (weight_grad * grad_diff) + (weight_dist * (dist ** 2))
#     diff += 1e-5

#     # 4. Build graph
#     g = maxflow.Graph[float]()
#     nodes = g.add_grid_nodes((h, w))
#     g.add_grid_edges(nodes, weights=diff, structure=np.array([[0,1,0],
#                                                               [1,0,1],
#                                                               [0,1,0]]), symmetric=True)

#     # --------------------------------------------------
#     # 5. Terminal costs (NEW: THE BOUNDARY WALL)
#     # --------------------------------------------------
#     INF = 1e9
#     cap_source = np.zeros((h, w), dtype=np.float32)  # patch
#     cap_sink = np.zeros((h, w), dtype=np.float32)    # query

#     is_hole = hole_mask_crop > 127
#     is_context = context_mask_crop > 127
#     is_outside = ~(is_hole | is_context)

#     # Force hole -> patch
#     cap_source[is_hole] = INF
#     cap_sink[is_hole] = 0

#     # Force outside -> query
#     cap_source[is_outside] = 0
#     cap_sink[is_outside] = INF

#     # THE WALL: Force the outermost 3 pixels of the array to be Query (Sink).
#     # This prevents the patch from EVER creating a straight edge along the bounding box.
#     cap_source[0:3, :] = 0; cap_sink[0:3, :] = INF  # Top edge
#     cap_source[-3:, :] = 0; cap_sink[-3:, :] = INF  # Bottom edge
#     cap_source[:, 0:3] = 0; cap_sink[:, 0:3] = INF  # Left edge
#     cap_source[:, -3:] = 0; cap_sink[:, -3:] = INF  # Right edge

#     g.add_grid_tedges(nodes, cap_source, cap_sink)

#     # 6. Solve
#     g.maxflow()
#     segments = g.get_grid_segments(nodes)
#     seam_mask = (~segments).astype(np.uint8) * 255
    
#     if first_component:
#         # 6. Solve
#         seam_energy = g.maxflow() # CAPTURE the max flow value here
#         segments = g.get_grid_segments(nodes)
#         seam_mask = (~segments).astype(np.uint8) * 255

#         return seam_mask, seam_energy
    

#     return seam_mask

# ─────────────────────────────────────────────────────────
# Helpers — pure Python, no numpy arithmetic
# ─────────────────────────────────────────────────────────

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
    # Used for the unary penalty on context pixels — Eq. 2 in the paper
    dist_raw = _dt_manhattan(hole, h, w)

    # --- 2. Build edge lists (pure Python) ---
    num_pixels = h * w
    SOURCE     = num_pixels
    SINK       = num_pixels + 1
    num_nodes  = num_pixels + 2
    INF        = 1e9
    K          = 0.002   # empirical constant from Eq. 2

    from_list, to_list, fwd_list, rev_list = [], [], [], []

    # Symmetric pairwise edges (4-connected)
    # Paper Eq. 1: Ci(p,q,...) = ∇diff(p,q) = |SSD(p) - SSD(q)|
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