import cv2
import numpy as np
import maxflow  # pip install PyMaxflow

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

def find_optimal_seam(q_crop, m_crop, hole_mask_crop, context_mask_crop):

    h, w = q_crop.shape[:2]



    # --------------------------------------------------

    # 1. Compute color difference

    # --------------------------------------------------

    qf = q_crop.astype(np.float32)

    mf = m_crop.astype(np.float32)



    color_diff = np.sqrt(np.sum((qf - mf) ** 2, axis=2))



    # --------------------------------------------------

    # 2. Compute gradient (structure) difference

    # --------------------------------------------------

    q_gray = cv2.cvtColor(q_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    m_gray = cv2.cvtColor(m_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)



    qgx = cv2.Sobel(q_gray, cv2.CV_32F, 1, 0, ksize=3)

    qgy = cv2.Sobel(q_gray, cv2.CV_32F, 0, 1, ksize=3)

    mgx = cv2.Sobel(m_gray, cv2.CV_32F, 1, 0, ksize=3)

    mgy = cv2.Sobel(m_gray, cv2.CV_32F, 0, 1, ksize=3)



    q_mag = np.sqrt(qgx**2 + qgy**2)

    m_mag = np.sqrt(mgx**2 + mgy**2)



    grad_diff = np.abs(q_mag - m_mag)



    # --------------------------------------------------

    # 3. Combine costs

    # --------------------------------------------------

    lambda_grad = 2.0

    diff = color_diff + lambda_grad * grad_diff



    diff += 1e-5  # avoid zero weights



    # --------------------------------------------------

    # 4. Build graph

    # --------------------------------------------------

    g = maxflow.Graph[float]()

    nodes = g.add_grid_nodes((h, w))



    # Proper grid edges (4-connected)

    g.add_grid_edges(nodes, weights=diff, structure=np.array([[0,1,0],

                                                              [1,0,1],

                                                              [0,1,0]]),

                     symmetric=True)



    # --------------------------------------------------

    # 5. Terminal costs (IMPORTANT FIX)

    # --------------------------------------------------

    INF = 1e9



    cap_source = np.zeros((h, w), dtype=np.float32)  # patch

    cap_sink = np.zeros((h, w), dtype=np.float32)    # query



    is_hole = hole_mask_crop > 127

    is_context = context_mask_crop > 127

    is_outside = ~(is_hole | is_context)



    # Force hole → patch

    cap_source[is_hole] = INF

    cap_sink[is_hole] = 0



    # Force outside → query

    cap_source[is_outside] = 0

    cap_sink[is_outside] = INF



    # Context = free (cut happens here)



    g.add_grid_tedges(nodes, cap_source, cap_sink)



    # --------------------------------------------------

    # 6. Solve

    # --------------------------------------------------

    g.maxflow()



    segments = g.get_grid_segments(nodes)



    # True = sink (query), False = source (patch)

    seam_mask = (~segments).astype(np.uint8) * 255



    return seam_mask