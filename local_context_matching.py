import math
import numpy as np
from lcm_solver import bgr_to_gray, bgr_to_lab, dilate, resize, texture_map, match_ssd


def get_texture_map(img_gray):
    """
    Sobel gradient magnitude, 5×5 median-blurred and normalised to [0, 255].
    img_gray: float32 or uint8 2-D numpy array.
    Returns float32 2-D numpy array.
    """
    return texture_map(img_gray)


def match_context_optimized(query_img, mask_img, match_img_list, weight_tex=10):
    """
    Local context matching (Hays & Efros, Section 4).

    Parameters
    ----------
    query_img      : BGR uint8 numpy array (H × W × 3).
    mask_img       : Grayscale uint8 numpy array (H × W); hole pixels > 127.
    match_img_list : List of BGR uint8 numpy arrays.
    weight_tex     : Weight for the texture SSD term.

    Returns
    -------
    List of dicts sorted by score (best first):
        {'match_idx': int, 'score': float, 'placement': (scale, tx, ty)}
    where tx = column offset, ty = row offset in the scaled candidate image.
    """
    q_h, q_w = query_img.shape[:2]

    # 1. Query features (C)
    q_lab  = bgr_to_lab(query_img)
    q_gray = bgr_to_gray(query_img)
    q_tex  = texture_map(q_gray)

    # 2. Context 'donut' mask — dilate hole, subtract hole (C for dilation)
    dilated = dilate(mask_img, radius=80)
    context_mask = [
        [(1.0 if (dilated[y, x] > 0 and mask_img[y, x] <= 127) else 0.0)
         for x in range(q_w)]
        for y in range(q_h)
    ]
    num_pixels = float(sum(
        context_mask[y][x]
        for y in range(q_h) for x in range(q_w)
    ))

    # 3. Bounding-box crop of templates
    y1, x1, y2, x2 = q_h, q_w, -1, -1
    for y in range(q_h):
        for x in range(q_w):
            if context_mask[y][x] > 0:
                if y < y1: y1 = y
                if y > y2: y2 = y
                if x < x1: x1 = x
                if x > x2: x2 = x
    y2 += 1; x2 += 1

    t_lab      = q_lab[y1:y2, x1:x2]
    t_tex      = q_tex[y1:y2, x1:x2]
    t_mask_1ch = np.array([context_mask[y][x1:x2] for y in range(y1, y2)], dtype=np.float32)
    th, tw = y2 - y1, x2 - x1

    rel_center_q_y = (y1 + th / 2.0) / q_h
    rel_center_q_x = (x1 + tw / 2.0) / q_w

    scales = [0.81, 0.90, 1.0]
    results = []

    for idx, match_img in enumerate(match_img_list):
        m_h, m_w = match_img.shape[:2]
        m_lab_full  = bgr_to_lab(match_img)
        m_gray_full = bgr_to_gray(match_img)
        m_tex_full  = texture_map(m_gray_full)

        best_score     = float('inf')
        best_placement = None

        for s in scales:
            sh = int(m_h * s)
            sw = int(m_w * s)
            if sh < th or sw < tw:
                continue

            s_lab = resize(m_lab_full, sh, sw)
            s_tex = resize(m_tex_full, sh, sw)

            ssd_lab = match_ssd(s_lab, t_lab, t_mask_1ch)   # numpy [res_h, res_w]
            ssd_tex = match_ssd(s_tex, t_tex, t_mask_1ch)

            res_h, res_w = ssd_lab.shape

            for row in range(res_h):
                for col in range(res_w):
                    combined = float(ssd_lab[row, col]) + weight_tex * float(ssd_tex[row, col])
                    rel_m_y = (row + th / 2.0) / s / m_h
                    rel_m_x = (col + tw / 2.0) / s / m_w
                    dist = math.sqrt((rel_center_q_y - rel_m_y) ** 2
                                     + (rel_center_q_x - rel_m_x) ** 2)
                    penalty = 1.0 + dist / 0.1
                    score = (combined / num_pixels) * penalty
                    if score < best_score:
                        best_score = score
                        best_placement = (s, col, row)  # (scale, tx, ty)

        results.append({
            'match_idx': idx,
            'score':     best_score,
            'placement': best_placement,
        })

    return sorted(results, key=lambda r: r['score'])
