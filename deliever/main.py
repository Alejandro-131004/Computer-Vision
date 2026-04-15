"""
Pool Ball Detection Pipeline
============================
Reads images from input/images/ (listed in input/input.json),
detects and identifies billiard balls, generates top-view warps,
and writes output/results.json + output/top_view/*.jpg

Parâmetros heurísticos optimizados por hill climbing v4.
Centroides de cor calculados offline sobre todo o dataset anotado.
KNN dinâmico substituído por distância a centroides fixos.
"""

import os
import csv
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────
# Directory structure
# ─────────────────────────────────────────────────────────────
INPUT_JSON       = Path("input/input.json")
IMAGE_DIR        = Path("input/images")
OUTPUT_DIR       = Path("output")
TOP_VIEW_DIR     = OUTPUT_DIR / "top_view"
RESULTS_JSON     = OUTPUT_DIR / "results.json"
GROUND_TRUTH_CSV = Path("input/ground_truth_counts.csv")   # opcional

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
BALL_LABEL_TO_NUMBER = {
    "cue": 0,
    "yellow_solid": 1,  "blue_solid": 2,   "red_solid": 3,
    "purple_solid": 4,  "orange_solid": 5,  "green_solid": 6,
    "maroon_solid": 7,  "black": 8,
    "yellow_stripe": 9, "blue_stripe": 10,  "red_stripe": 11,
    "purple_stripe": 12,"orange_stripe": 13,"green_stripe": 14,
    "maroon_stripe": 15,
}
VALID_LABELS           = set(BALL_LABEL_TO_NUMBER.keys())
REQUIRED_UNIQUE_LABELS = ["cue", "black"]

WHITE_LOW   = np.array([0,   0, 160], dtype=np.uint8)
WHITE_HIGH  = np.array([179,  70, 255], dtype=np.uint8)
BLACK_LOW   = np.array([0,   0,   0], dtype=np.uint8)
BLACK_HIGH  = np.array([179, 255,  55], dtype=np.uint8)

ROI_PAD             = 2.0
INNER_RADIUS_FACTOR = 0.7

# ─────────────────────────────────────────────────────────────
# Parâmetros optimizados (hill climbing v4)
# ─────────────────────────────────────────────────────────────
P = {
    "s1_cue_white_ratio_min":        0.4,
    "s1_cue_white_center_min":       0.6,
    "s1_cue_black_ratio_max":        0.01,
    "s1_cue_score_bonus":            1.0,
    "s1_dark_black_ratio_min":       0.4,
    "s1_dark_colour_ratio_min":      0.45,
    "s1_dark_v_std_min":             25.0,
    "s1_dark_score_bonus":           1.0,
    "s1_uniform_colour_ratio_min":   0.98,
    "s1_uniform_black_ratio_max":    0.05,
    "s1_uniform_white_center_max":   0.02,
    "s1_uniform_white_ratio_max":    0.02,
    "s1_uniform_v_std_min":          8.0,
    "s1_uniform_score_bonus":        2.4,
    "s1_artifact_black_ratio_min":   0.25,
    "s1_artifact_h_std_max":         25.0,
    "s1_artifact_white_center_max":  0.02,
    "s1_artifact_score_penalty":    -1.5,
    "s1_pos_white_center_min":       0.02,
    "s1_pos_white_center_bonus":     2.0,
    "s1_pos_white_ratio_min":        0.02,
    "s1_pos_white_ratio_bonus":      0.7,
    "s1_pos_h_std_good_min":         12.0,
    "s1_pos_h_std_good_bonus":       1.6,
    "s1_pos_h_std_vgood_min":        20.0,
    "s1_pos_h_std_vgood_bonus":      0.6,
    "s1_pos_v_std_good_min":         20.0,
    "s1_pos_v_std_good_bonus":       0.2,
    "s1_pos_v_std_vgood_min":        40.0,
    "s1_pos_v_std_vgood_bonus":      0.5,
    "s1_pos_colour_ratio_min":       0.75,
    "s1_pos_colour_ratio_bonus":     0.2,
    "s1_sus_h_std_low_thr":          15.0,
    "s1_sus_h_std_low_penalty":     -0.6,
    "s1_sus_h_std_vlow_thr":         5.0,
    "s1_sus_h_std_vlow_penalty":    -1.0,
    "s1_sus_sat_thr":               180.0,
    "s1_sus_sat_h_std_thr":          8.0,
    "s1_sus_sat_penalty":           -0.8,
    "s1_sus_white_center_vlow_thr":  0.01,
    "s1_sus_white_center_vlow_pen": -2.6,
    "s1_sus_white_ratio_vlow_thr":   0.01,
    "s1_sus_white_ratio_vlow_pen":  -0.2,
    "s1_sus_center_minus_ring_thr":  0.1,
    "s1_sus_center_minus_ring_pen": -0.3,
    "s1_sus_black_ratio_thr":        0.22,
    "s1_sus_black_ratio_pen":       -0.2,
    "s1_black_black_ratio_min":      0.28,
    "s1_black_colour_ratio_min":     0.55,
    "s1_black_v_std_min":            30.0,
    "s1_black_white_center_max":     0.03,
    "s1_black_score_bonus":          3.0,
    "s1_suspect_score_thr":         -0.5,
    "s2_black_black_ratio_min":      0.28,
    "s2_black_white_ratio_max":      0.22,
    "s2_black_v_median_max":        150.0,
    "s2_yellow_white_ratio_lo":      0.17,
    "s2_yellow_white_ratio_hi":      0.21,
    "s2_yellow_black_ratio_max":     0.02,
    "s2_yellow_colour_ratio_min":    0.75,
    "s2_yellow_h_median_max":        60.0,
    "s2_white_white_ratio_min":      0.21,
    "s2_white_v_p90_min":           190.0,
    "s2_white_s_median_max":        230.0,
    "s2_white_black_ratio_max":      0.1,
    "s2_white_center_min":           0.1,
    "s2_white_center_ratio":         0.65,
    "s2_chrom_s_min":               100.0,
    "s2_chrom_colour_ratio_min":     0.55,
    "s2_red_h_max":                  10.0,
    "s2_red_h_min_wrap":            173.0,
    "s2_maroon_h_lo":                4.0,
    "s2_maroon_h_hi":               28.0,
    "s2_maroon_v_max":              160.0,
    "s2_orange_h_lo":                8.0,
    "s2_orange_h_hi":               17.0,
    "s2_orange_v_min":              120.0,
    "s3_stripe_score_coef":          2.5,
    "s3_white_ratio_ring_thr":       0.01,
    "s3_center_ring_diff_coef":      0.5,
    "s3_white_ratio_offset":         0.08,
    "s3_white_ratio_coef":           1.4,
    "s3_sigmoid_scale":             13.0,
    "s3_purple_h_std_thr":          18.0,
    "s3_purple_colour_ratio_thr":    0.9,
    "s3_purple_solid_score":        -5.0,
    "s3_purple_stripe_coef":         3.5,
    "s4_w_final_conf":               2.0,
    "s4_w_base_conf":                0.5,
    "s4_w_type_conf":                0.3,
    "s4_w_validity_score":           0.8,
    "s4_suspect_penalty":           -4.0,
    "s4_weak_suspect_strength_thr": -1.0,
    "s4_strong_invalid_score_thr":  -3.5,
}

# ─────────────────────────────────────────────────────────────
# Centroides fixos (calculados offline com extract_centroids.py)
# Normalização: xz = (x - GLOBAL_MEAN) / GLOBAL_STD
# ─────────────────────────────────────────────────────────────
GLOBAL_MEAN = [-0.7877705373367201, -0.2752829286857764, 154.76539855072463,
               166.87409420289856, 0.085306419094573, 0.03025238070689081,
               0.8844412001985352, 28.541104124680547, 220.50797101449274,
               48.03183535989974, 190.07952898550727, -0.0984191093056313]
GLOBAL_STD  = [0.4325788883435084, 0.34133912028640145, 24.010428331908983,
               25.43220597252245, 0.09921620913286307, 0.06727568863076369,
               0.10393705542017162, 17.22926682797084, 16.86739226941064,
               8.35336093823409, 26.424894206715685, 0.16410393140780577]

# centroides já na escala normalizada (mean dos treinos normalizados)
_RAW_CENTROIDS = {
    "black":  [-0.9056799560168798, -0.4227003234995948, 165.18627450980392, 153.86274509803923, 0.015626100001772274, 0.22311838013279497, 0.7612555198654324, 9.884810465992155, 214.0254901960784, 55.37450357523072, 208.70980392156864, -0.0171297200238404],
    "blue":   [-0.8899953854459092, -0.45258956075809476, 171.03846153846155, 157.03846153846155, 0.06122250332515758, 0.004259385075639972, 0.9345181115992025, 9.330295700138016, 214.44615384615383, 43.92850415957547, 211.6076923076923, -0.0880571211424841],
    "green":  [-0.9578594060773775, -0.243198035979285, 157.99295774647888, 147.27464788732394, 0.07213934746825254, 0.012523673715811891, 0.9153369788159356, 15.696441463010029, 213.05633802816902, 50.60714451903573, 181.31549295774647, -0.10538448555916931],
    "maroon": [-0.7353791837655905, -0.22418705264276587, 145.17910447761193, 158.15671641791045, 0.0575824457713959, 0.023260701361317426, 0.9191568528672868, 41.05985431838963, 215.7955223880597, 48.459382860533196, 181.50597014925373, -0.058389048762309756],
    "orange": [-0.5608191950739649, -0.14601214209492486, 155.84375, 188.9921875, 0.07090530657784065, 0.015095014057448779, 0.9139996793647105, 47.676385986385036, 233.09375, 48.888990330754716, 194.765625, -0.0435643829055777],
    "purple": [-0.8777972362386933, -0.47132834018384734, 150.10833333333332, 151.51666666666668, 0.08570170291882372, 0.0016690608762964627, 0.9126292362048799, 15.954384572954151, 213.66, 45.50846119067524, 184.76500000000001, -0.12529724993323402],
    "red":    [-0.8398023021856365, -0.4113999908508977, 151.29365079365078, 175.76190476190476, 0.055067342042510174, 0.013157536157002181, 0.9317751218004875, 52.719001911206426, 223.41111111111113, 48.09003823234268, 186.9793650793651, -0.032014001222198635],
    "white":  [-0.8988359196905334, -0.32243111843416583, 144.08, 184.76, 0.30618409379049727, 0.0037145545065434805, 0.6901013517029593, 22.375420980480033, 221.14399999999998, 41.38835208949159, 174.98399999999998, -0.43769047883832735],
    "yellow": [-0.4446218356386533, 0.17986372533151804, 151.27868852459017, 188.68032786885246, 0.07990726152024494, 0.008440850049358399, 0.9116518884303968, 38.19700643290785, 236.24754098360657, 49.7662571758562, 187.0672131147541, -0.034890057026059275],
}

_GMEAN = np.array(GLOBAL_MEAN, dtype=float)
_GSTD  = np.array(GLOBAL_STD,  dtype=float)
_GSTD[_GSTD == 0] = 1.0

# pré-normaliza os centroides uma única vez no arranque
CENTROIDS_Z = {
    colour: (np.array(vec, dtype=float) - _GMEAN) / _GSTD
    for colour, vec in _RAW_CENTROIDS.items()
}

# ─────────────────────────────────────────────────────────────
# CSV ground-truth label map
# ─────────────────────────────────────────────────────────────
CSV_LABEL_MAP = {
    "0_white_cue":      "cue",
    "1_yellow_solid":   "yellow_solid",
    "2_blue_solid":     "blue_solid",
    "3_red_solid":      "red_solid",
    "4_purple_solid":   "purple_solid",
    "5_orange_solid":   "orange_solid",
    "6_green_solid":    "green_solid",
    "7_maroon_solid":   "maroon_solid",
    "8_black_8ball":    "black",
    "9_yellow_stripe":  "yellow_stripe",
    "10_blue_stripe":   "blue_stripe",
    "11_red_stripe":    "red_stripe",
    "12_purple_stripe": "purple_stripe",
    "13_orange_stripe": "orange_stripe",
    "14_green_stripe":  "green_stripe",
    "15_maroon_stripe": "maroon_stripe",
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def safe_float(v, default=0.0):
    if v is None:
        return float(default)
    try:
        f = float(v)
        return default if (f != f) else f   # NaN check sem numpy
    except Exception:
        return float(default)

def label_to_base_colour(label):
    if label == "cue":               return "white"
    if label == "black":             return "black"
    if label.endswith("_solid"):     return label.replace("_solid", "")
    if label.endswith("_stripe"):    return label.replace("_stripe", "")
    return None

def label_to_ball_type(label):
    if label == "cue":               return "cue"
    if label == "black":             return "black"
    if label.endswith("_solid"):     return "solid"
    if label.endswith("_stripe"):    return "stripe"
    return None

def final_label_from_parts(base_colour, ball_type):
    if base_colour == "white":       return "cue"
    if base_colour == "black":       return "black"
    if base_colour is None or ball_type is None: return "not_ball"
    return f"{base_colour}_{ball_type}"

def order_corners(pts):
    sorted_y = pts[np.argsort(pts[:, 1])]
    top    = sorted_y[:2][np.argsort(sorted_y[:2, 0])]
    bottom = sorted_y[2:][np.argsort(sorted_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# Stage 0 — Table segmentation & top-view warp
# ─────────────────────────────────────────────────────────────
def segment_table(img):
    hsv        = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask  = cv2.inRange(hsv, np.array([90, 80, 50]),  np.array([130, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 80, 50]),  np.array([85,  255, 255]))
    mask       = cv2.bitwise_or(blue_mask, green_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_cnt   = max(contours, key=cv2.contourArea)
    approx      = cv2.approxPolyDP(table_cnt, 0.02 * cv2.arcLength(table_cnt, True), True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        corners = cv2.boxPoints(cv2.minAreaRect(table_cnt)).astype(np.float32)

    corners    = order_corners(corners)
    clean_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(clean_mask, [corners.astype(np.int32)], 255)
    return clean_mask, corners

def get_top_view(img, corners, expand_px=20, table_ratio=2.0):
    center   = corners.mean(axis=0)
    expanded = np.array([
        pt + (pt - center) / (np.linalg.norm(pt - center) + 1e-9) * expand_px
        for pt in corners
    ], dtype=np.float32)
    tl, tr, br, bl = expanded
    W   = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H   = int(W * table_ratio)
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(expanded, dst)
    warped = cv2.warpPerspective(img, M, (W, H))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

# ─────────────────────────────────────────────────────────────
# Stage 0 — Ball detection
# ─────────────────────────────────────────────────────────────
def detect_balls(img, mask):
    masked   = cv2.bitwise_and(img, img, mask=mask)
    hsv      = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = cv2.split(hsv)
    s_c = cv2.convertScaleAbs(s_c, alpha=1.35, beta=10)
    v_c = cv2.convertScaleAbs(v_c, alpha=1.15, beta=8)
    boosted  = cv2.cvtColor(cv2.merge([h_c, s_c, v_c]), cv2.COLOR_HSV2BGR)

    cloth_mean = np.array(cv2.mean(boosted, mask=mask)[:3])
    diff = np.clip(np.sqrt(np.sum((boosted.astype(np.float32) - cloth_mean) ** 2, axis=2)), 0, 255).astype(np.uint8)

    gray_raw  = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    _, s2, _  = cv2.split(cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV))
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    s_enh     = clahe.apply(s2)
    gray_cl   = clahe.apply(gray_raw)

    gray      = cv2.GaussianBlur(cv2.addWeighted(diff, 0.9, s_enh, 0.1, 0), (3, 3), 2)
    gray_enh  = cv2.GaussianBlur(cv2.addWeighted(
                    cv2.addWeighted(diff, 0.8, s_enh, 0.2, 0), 0.85, gray_cl, 0.15, 0), (3, 3), 2)

    h, _     = img.shape[:2]
    min_r    = max(8, int(h * 0.01))
    max_r    = int(h * 0.025)
    edge     = cv2.Canny(gray, 40, 140)

    candidates = []
    for g, p2 in [(gray, 29.3), (gray, 26.5), (gray_enh, 26.5)]:
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 1.7,
                             param1=38, param2=p2, minRadius=min_r, maxRadius=max_r)
        if c is not None:
            candidates.extend([(int(x), int(y), int(r), "hough")
                                for x, y, r in np.round(c[0]).astype(int).tolist()])

    hsv2 = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)
    pmask = cv2.morphologyEx(
        cv2.morphologyEx(
            cv2.bitwise_and(cv2.inRange(hsv2, np.array([110, 40, 30]), np.array([175, 255, 255])), mask),
            cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8)),
        cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    for cnt in cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_r**2 * 0.35 or area > np.pi * max_r**2 * 1.5:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0 or 4 * np.pi * area / perim**2 < 0.45:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        candidates.append((int(x), int(y), int(r), "purple"))

    balls = []
    for (x, y, r, src) in candidates:
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            continue
        if src == "purple":
            im = np.zeros_like(mask)
            cv2.circle(im, (x, y), max(2, int(r * 0.65)), 255, -1)
            if cv2.countNonZero(cv2.bitwise_and(pmask, im)) / max(cv2.countNonZero(im), 1) < 0.45:
                continue
        cm = np.zeros_like(mask); cv2.circle(cm, (x, y), r, 255, -1)
        if cv2.mean(gray, mask=cm)[0] < 25:
            continue
        overlap = cv2.countNonZero(cv2.bitwise_and(mask, cm))
        if cv2.countNonZero(cm) == 0 or overlap / cv2.countNonZero(cm) <= 0.35:
            continue
        rm = np.zeros_like(mask); cv2.circle(rm, (x, y), r, 255, 2)
        if src != "purple" and cv2.countNonZero(cv2.bitwise_and(edge, rm)) / max(2*np.pi*r, 1) < 0.20:
            continue
        balls.append((x, y, r))

    balls.sort(key=lambda b: b[2], reverse=True)
    deduped = []
    for x, y, r in balls:
        if not any(np.hypot(x-x2, y-y2) < max(r, r2) for x2, y2, r2 in deduped):
            deduped.append((x, y, r))

    if len(deduped) >= 4:
        corrected = []
        for i, (x, y, r) in enumerate(deduped):
            local_r = sorted([(np.hypot(x-x2, y-y2), r2)
                               for j, (x2, y2, r2) in enumerate(deduped) if i != j])
            local_r = [r2 for _, r2 in local_r[:3]]
            med = np.median(local_r)
            corrected.append((x, y, int(med) if (r < 0.7*med or r > 1.2*med) else r))
        deduped = corrected
    return deduped

# ─────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────
def compute_stripe_score(roi, r_inner=0.45, r_outer=0.75):
    h, w = roi.shape[:2]; cx, cy = w//2, h//2
    ri, ro = int(min(cx, cy)*r_inner), int(min(cx, cy)*r_outer)
    mi = np.zeros((h, w), np.uint8); cv2.circle(mi, (cx, cy), ri, 255, -1)
    mo = np.zeros((h, w), np.uint8); cv2.circle(mo, (cx, cy), ro, 255, -1)
    mr = cv2.bitwise_and(mo, cv2.bitwise_not(mi))
    wm = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), WHITE_LOW, WHITE_HIGH)
    wc = cv2.countNonZero(cv2.bitwise_and(wm, mi))
    wr = cv2.countNonZero(cv2.bitwise_and(wm, mr))
    tc = max(cv2.countNonZero(mi), 1); tr = max(cv2.countNonZero(mr), 1)
    return float(wr/tr - wc/tc), float(wc/tc), float(wr/tr)

def compute_patch_statistics(roi):
    if roi is None or roi.size == 0:
        return None
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h2, w2 = roi.shape[:2]
    cm   = np.zeros((h2, w2), np.uint8)
    cv2.circle(cm, (w2//2, h2//2), int(min(w2//2, h2//2)*INNER_RADIUS_FACTOR), 255, -1)

    wm  = cv2.bitwise_and(cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH), cm)
    bm  = cv2.bitwise_and(cv2.inRange(hsv, BLACK_LOW, BLACK_HIGH), cm)
    clm = cv2.bitwise_and(cm, cv2.bitwise_not(cv2.bitwise_or(wm, bm)))

    tot = max(cv2.countNonZero(cm), 1)
    stats = {
        "white_ratio":  cv2.countNonZero(wm)  / tot,
        "black_ratio":  cv2.countNonZero(bm)  / tot,
        "colour_ratio": cv2.countNonZero(clm) / tot,
    }
    cpx = cv2.countNonZero(clm)
    if cpx > 0:
        hv = hsv[:,:,0][clm > 0].astype(float)
        sv = hsv[:,:,1][clm > 0].astype(float)
        vv = hsv[:,:,2][clm > 0].astype(float)
        stats.update({
            "h_median": float(np.median(hv)), "s_median": float(np.median(sv)),
            "v_median": float(np.median(vv)), "h_std":    float(np.std(hv)),
            "s_p90":    float(np.percentile(sv, 90)),
            "v_p90":    float(np.percentile(vv, 90)), "v_std": float(np.std(vv)),
        })
    else:
        for k in ["h_median","s_median","v_median","h_std","s_p90","v_p90","v_std"]:
            stats[k] = None
    ss, rc, rr = compute_stripe_score(roi)
    stats["stripe_score"] = ss
    stats["white_ratio_center"] = rc
    stats["white_ratio_ring"]   = rr
    return stats

def crop_ball_roi(img, x, y, r):
    h, w = img.shape[:2]; rr = int(round(r * ROI_PAD))
    x1, y1 = max(0, x-rr), max(0, y-rr)
    x2, y2 = min(w, x+rr), min(h, y+rr)
    return img[y1:y2, x1:x2].copy()

def build_feature_vector(row):
    h = row.get("h_median")
    hcos = hsin = 0.0
    if h is not None:
        a = 2.0 * np.pi * float(h) / 180.0
        hcos, hsin = float(np.cos(a)), float(np.sin(a))
    return np.array([
        hcos, hsin,
        safe_float(row.get("s_median")),  safe_float(row.get("v_median")),
        safe_float(row.get("white_ratio")), safe_float(row.get("black_ratio")),
        safe_float(row.get("colour_ratio")), safe_float(row.get("h_std")),
        safe_float(row.get("v_p90")),     safe_float(row.get("v_std")),
        safe_float(row.get("s_p90")),     safe_float(row.get("stripe_score")),
    ], dtype=float)

# ─────────────────────────────────────────────────────────────
# Stage 1 — Validity scoring (parâmetros optimizados)
# ─────────────────────────────────────────────────────────────
def predict_validity(row):
    wr  = safe_float(row.get("white_ratio"))
    br  = safe_float(row.get("black_ratio"))
    cr  = safe_float(row.get("colour_ratio"))
    hs  = safe_float(row.get("h_std"))
    vs  = safe_float(row.get("v_std"))
    sm  = safe_float(row.get("s_median"))
    wrc = safe_float(row.get("white_ratio_center"))
    wrr = safe_float(row.get("white_ratio_ring"))
    cmr = wrc - wrr
    score = 0.0

    likely_cue     = wr >= P["s1_cue_white_ratio_min"] and wrc >= P["s1_cue_white_center_min"] and br <= P["s1_cue_black_ratio_max"]
    likely_dark    = br >= P["s1_dark_black_ratio_min"] and cr >= P["s1_dark_colour_ratio_min"] and vs >= P["s1_dark_v_std_min"]
    likely_uniform = (cr >= P["s1_uniform_colour_ratio_min"] and br <= P["s1_uniform_black_ratio_max"]
                      and wrc <= P["s1_uniform_white_center_max"] and wr <= P["s1_uniform_white_ratio_max"]
                      and vs >= P["s1_uniform_v_std_min"])
    likely_black   = (br >= P["s1_black_black_ratio_min"] and cr >= P["s1_black_colour_ratio_min"]
                      and vs >= P["s1_black_v_std_min"] and wrc <= P["s1_black_white_center_max"])

    if likely_cue:     score += P["s1_cue_score_bonus"]
    if likely_dark:    score += P["s1_dark_score_bonus"]
    if likely_uniform: score += P["s1_uniform_score_bonus"]
    if likely_black:   score += P["s1_black_score_bonus"]

    likely_artifact = br >= P["s1_artifact_black_ratio_min"] and hs <= P["s1_artifact_h_std_max"] and wrc <= P["s1_artifact_white_center_max"]
    if likely_artifact: score += P["s1_artifact_score_penalty"]

    if wrc >= P["s1_pos_white_center_min"]: score += P["s1_pos_white_center_bonus"]
    if wr  >= P["s1_pos_white_ratio_min"]:  score += P["s1_pos_white_ratio_bonus"]
    if hs  >= P["s1_pos_h_std_good_min"]:   score += P["s1_pos_h_std_good_bonus"]
    if hs  >= P["s1_pos_h_std_vgood_min"]:  score += P["s1_pos_h_std_vgood_bonus"]
    if vs  >= P["s1_pos_v_std_good_min"]:   score += P["s1_pos_v_std_good_bonus"]
    if vs  >= P["s1_pos_v_std_vgood_min"]:  score += P["s1_pos_v_std_vgood_bonus"]
    if cr  >= P["s1_pos_colour_ratio_min"]: score += P["s1_pos_colour_ratio_bonus"]

    if hs < P["s1_sus_h_std_low_thr"]  and not likely_uniform and not likely_dark: score += P["s1_sus_h_std_low_penalty"]
    if hs < P["s1_sus_h_std_vlow_thr"] and not likely_uniform and not likely_dark: score += P["s1_sus_h_std_vlow_penalty"]
    if sm >= P["s1_sus_sat_thr"] and hs < P["s1_sus_sat_h_std_thr"]: score += P["s1_sus_sat_penalty"]
    if wrc < P["s1_sus_white_center_vlow_thr"] and not likely_uniform: score += P["s1_sus_white_center_vlow_pen"]
    if wr  < P["s1_sus_white_ratio_vlow_thr"]  and not likely_uniform: score += P["s1_sus_white_ratio_vlow_pen"]
    if cmr > P["s1_sus_center_minus_ring_thr"] and not likely_cue:     score += P["s1_sus_center_minus_ring_pen"]
    if br  > P["s1_sus_black_ratio_thr"] and not likely_dark:          score += P["s1_sus_black_ratio_pen"]

    vc = "suspect" if score <= P["s1_suspect_score_thr"] else "valid_ball"
    return {"validity_class": vc, "validity_score": float(score)}

# ─────────────────────────────────────────────────────────────
# Stage 2 — Base colour (regras explícitas + centroides fixos)
# ─────────────────────────────────────────────────────────────
def apply_colour_rules(row):
    br  = safe_float(row.get("black_ratio"))
    wr  = safe_float(row.get("white_ratio"))
    cr  = safe_float(row.get("colour_ratio"))
    hm  = safe_float(row.get("h_median"))
    vm  = safe_float(row.get("v_median"))
    sm  = safe_float(row.get("s_median"))
    vp  = safe_float(row.get("v_p90"), vm)
    wrc = safe_float(row.get("white_ratio_center"))
    bt  = row.get("ball_type_hint")

    if br >= P["s2_black_black_ratio_min"] and wr <= P["s2_black_white_ratio_max"] and vm <= P["s2_black_v_median_max"]:
        return "black"
    if (bt == "stripe" and P["s2_yellow_white_ratio_lo"] <= wr <= P["s2_yellow_white_ratio_hi"]
            and br <= P["s2_yellow_black_ratio_max"] and cr >= P["s2_yellow_colour_ratio_min"]
            and hm <= P["s2_yellow_h_median_max"]):
        return "yellow"
    if (wr >= P["s2_white_white_ratio_min"] and vp >= P["s2_white_v_p90_min"]
            and sm <= P["s2_white_s_median_max"] and br <= P["s2_white_black_ratio_max"]
            and wrc >= P["s2_white_center_min"] and wr >= wrc * P["s2_white_center_ratio"]):
        return "white"
    if sm >= P["s2_chrom_s_min"] and cr >= P["s2_chrom_colour_ratio_min"]:
        if hm <= P["s2_red_h_max"] or hm >= P["s2_red_h_min_wrap"]:                            return "red"
        if P["s2_maroon_h_lo"] < hm <= P["s2_maroon_h_hi"] and vm <= P["s2_maroon_v_max"]:     return "maroon"
        if P["s2_orange_h_lo"] < hm <= P["s2_orange_h_hi"] and vm >  P["s2_orange_v_min"]:     return "orange"
    return None

def predict_base_colour(row):
    rule = apply_colour_rules(row)
    if rule is not None:
        return {"top1": rule, "ranked": [(rule, 1.0)], "confidence": 1.0}

    # Fallback: distância a centroides fixos (substitui KNN dinâmico)
    xz = (build_feature_vector(row) - _GMEAN) / _GSTD
    dists = {c: float(np.linalg.norm(xz - cz)) for c, cz in CENTROIDS_Z.items()}
    # exclui white/black do fallback cromático (tratados pelas regras)
    chrom = {c: d for c, d in dists.items() if c not in ("white", "black")}
    ranked_raw = sorted(chrom.items(), key=lambda x: x[1])

    # converte distância em "confiança" por softmin
    min_d  = ranked_raw[0][1] + 1e-9
    scores = [(c, 1.0 / (d / min_d + 1e-9)) for c, d in ranked_raw]
    total  = sum(s for _, s in scores)
    ranked = [(c, s / total) for c, s in scores]

    return {"top1": ranked[0][0], "ranked": ranked, "confidence": float(ranked[0][1])}

# ─────────────────────────────────────────────────────────────
# Stage 3 — Ball type
# ─────────────────────────────────────────────────────────────
def predict_ball_type(row, colour):
    if colour == "white": return {"top1": "cue",   "ranked": [("cue",   1.0)], "confidence": 1.0}
    if colour == "black": return {"top1": "black", "ranked": [("black", 1.0)], "confidence": 1.0}

    ss  = safe_float(row.get("stripe_score"))
    wr  = safe_float(row.get("white_ratio"))
    wrc = safe_float(row.get("white_ratio_center"))
    wrr = safe_float(row.get("white_ratio_ring"))
    hs  = safe_float(row.get("h_std"))
    cr  = safe_float(row.get("colour_ratio"))

    if colour == "purple":
        score = P["s3_purple_solid_score"] if (hs < P["s3_purple_h_std_thr"] and cr > P["s3_purple_colour_ratio_thr"]) \
                else -P["s3_purple_stripe_coef"] * ss
    else:
        score = -P["s3_stripe_score_coef"] * ss
        if ss < -0.03 and wrr > P["s3_white_ratio_ring_thr"]:
            score += P["s3_center_ring_diff_coef"] * (wrc - wrr)
        score += P["s3_white_ratio_coef"] * (wr - P["s3_white_ratio_offset"])

    prob_stripe = 1.0 / (1.0 + np.exp(-P["s3_sigmoid_scale"] * score))
    prob_solid  = 1.0 - prob_stripe
    ranked = sorted([("stripe", prob_stripe), ("solid", prob_solid)], key=lambda x: x[1], reverse=True)
    return {"top1": ranked[0][0], "ranked": ranked, "confidence": float(ranked[0][1])}

def build_ranked_final(base_pred, type_pred):
    options = []
    for colour, cp in base_pred["ranked"]:
        if colour in ("white", "black"):
            rtypes = [("cue", 1.0)] if colour == "white" else [("black", 1.0)]
        elif colour == base_pred["top1"]:
            rtypes = type_pred["ranked"]
        else:
            rtypes = [("solid", 0.5), ("stripe", 0.5)]
        for bt, tp in rtypes:
            options.append({"final_label": final_label_from_parts(colour, bt),
                            "base_colour": colour, "ball_type": bt,
                            "joint_prob": float(cp) * float(tp)})
    dedup = {}
    for item in sorted(options, key=lambda d: d["joint_prob"], reverse=True):
        if item["final_label"] not in dedup:
            dedup[item["final_label"]] = item
    return list(dedup.values())

def predict_single(row):
    validity   = predict_validity(row)
    base_pred  = predict_base_colour(row)
    type_pred  = predict_ball_type(row, base_pred["top1"])
    ranked     = build_ranked_final(base_pred, type_pred)

    out = dict(row)
    out.update(validity)
    out["pred_base_colour"]      = base_pred["top1"]
    out["pred_base_colour_conf"] = float(base_pred["confidence"])
    out["pred_ball_type"]        = type_pred["top1"]
    out["pred_ball_type_conf"]   = float(type_pred["confidence"])
    out["pred_final_label"]      = ranked[0]["final_label"] if ranked else "not_ball"
    out["pred_final_conf"]       = ranked[0]["joint_prob"]  if ranked else 0.0
    out["ranked_final_predictions"] = ranked
    return out

# ─────────────────────────────────────────────────────────────
# Stage 4 — Global consistency
# ─────────────────────────────────────────────────────────────
def combined_strength(row):
    penalty = P["s4_suspect_penalty"] if row["validity_class"] == "suspect" else 0.0
    return (
        P["s4_w_final_conf"]    * float(row["pred_final_conf"]) +
        P["s4_w_base_conf"]     * float(row["pred_base_colour_conf"]) +
        P["s4_w_type_conf"]     * float(row["pred_ball_type_conf"]) +
        P["s4_w_validity_score"]* float(row["validity_score"]) +
        penalty
    )

def get_label_prob(row, label):
    for opt in row["ranked_final_predictions"]:
        if opt["final_label"] == label:
            return float(opt["joint_prob"])
    return 0.0

def choose_required(rows, req):
    cands = []
    for idx, row in enumerate(rows):
        prob = get_label_prob(row, req)
        if prob <= 0.0: continue
        tier = 0 if row["validity_class"] == "valid_ball" else 1
        cands.append((tier, row["strength"] + 2.0*prob, idx, prob))
    if not cands: return None
    cands.sort(key=lambda x: (x[0], -x[1]))
    return cands[0][2], cands[0][3]

def try_reassign(row, assigned, forbidden=None):
    counts = Counter(assigned)
    for opt in row["ranked_final_predictions"]:
        cand = opt["final_label"]
        if cand == forbidden or cand == "not_ball": continue
        if counts[cand] < 1:
            return cand, float(opt["joint_prob"]), "reassigned"
    return row["final_global_label"], row["final_global_conf"], "kept_duplicate"

def resolve_global(pred_rows):
    rows = [dict(r) for r in pred_rows]
    for row in rows:
        row["strength"]           = combined_strength(row)
        row["final_global_label"] = row["pred_final_label"]
        row["final_global_conf"]  = float(row["pred_final_conf"])
        row["global_resolution"]  = "kept_top1"

    for _ in range(2):
        for req in REQUIRED_UNIQUE_LABELS:
            w = choose_required(rows, req)
            if w is None: continue
            widx, wprob = w
            rows[widx]["final_global_label"] = req
            rows[widx]["final_global_conf"]  = wprob
            rows[widx]["global_resolution"]  = f"forced_{req}"

        changed = True
        while changed:
            changed = False
            grouped = defaultdict(list)
            for idx, row in enumerate(rows):
                if row["final_global_label"] != "not_ball":
                    grouped[row["final_global_label"]].append((idx, row))
            for label, items in grouped.items():
                if len(items) <= 1: continue
                items.sort(key=lambda x: x[1]["strength"], reverse=True)
                overflow = {idx for idx, _ in items[1:]}
                kept = [rows[i]["final_global_label"] for i in range(len(rows))
                        if rows[i]["final_global_label"] != "not_ball" and i not in overflow]
                for idx, row in items[1:]:
                    nl, np_, mode = try_reassign(row, kept, forbidden=label)
                    if nl != row["final_global_label"]:
                        rows[idx]["final_global_label"] = nl
                        rows[idx]["final_global_conf"]  = np_
                        rows[idx]["global_resolution"]  = mode
                        changed = True

    for row in rows:
        if row["final_global_label"] in ("cue", "black"): continue
        if row["validity_class"] == "suspect" and row["strength"] < P["s4_weak_suspect_strength_thr"]:
            row["final_global_label"] = "not_ball"; row["final_global_conf"] = 0.0
            row["global_resolution"]  = "weak_suspect_demoted"
        if row["validity_score"] <= P["s4_strong_invalid_score_thr"]:
            row["final_global_label"] = "not_ball"; row["final_global_conf"] = 0.0
            row["global_resolution"]  = "final_strong_invalid"

    return sorted(rows, key=lambda d: (d["x"], d["y"]))

# ─────────────────────────────────────────────────────────────
# Avaliação contra ground-truth CSV
# ─────────────────────────────────────────────────────────────
def load_ground_truth_csv(csv_path):
    skip = {"filename", "ball_count", ""}
    gt = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = row["filename"].strip()
            gt[fname] = {}
            for k, v in row.items():
                if k is None: continue
                k = k.strip()
                if k in skip or k.startswith("Unnamed"): continue
                v = (v or "").strip()
                if v:
                    try: gt[fname][k] = int(v)
                    except ValueError: pass
    return gt

def evaluate_against_csv(results, csv_path):
    gt_data = load_ground_truth_csv(csv_path)
    preds_by_image = {r["image"]: Counter(b["label"] for b in r["balls"]) for r in results}

    total_tp = total_fp = total_fn = 0
    per_image = []

    for filename, gt_row in gt_data.items():
        pred_counts = preds_by_image.get(filename, Counter())
        img_tp = img_fp = img_fn = 0
        for csv_col, label in CSV_LABEL_MAP.items():
            gt_c   = gt_row.get(csv_col, 0)
            pred_c = pred_counts.get(label, 0)
            img_tp += min(gt_c, pred_c)
            img_fp += max(0, pred_c - gt_c)
            img_fn += max(0, gt_c - pred_c)
        total_tp += img_tp; total_fp += img_fp; total_fn += img_fn
        p = img_tp / max(img_tp + img_fp, 1)
        r = img_tp / max(img_tp + img_fn, 1)
        per_image.append({"image": filename, "tp": img_tp, "fp": img_fp, "fn": img_fn,
                           "precision": round(p, 4), "recall": round(r, 4),
                           "f1": round(2*p*r/max(p+r, 1e-9), 4)})

    prec = total_tp / max(total_tp + total_fp, 1)
    rec  = total_tp / max(total_tp + total_fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "per_image": per_image}

# ─────────────────────────────────────────────────────────────
# Per-image pipeline
# ─────────────────────────────────────────────────────────────
def predict_image(img_bgr, image_name):
    mask, corners = segment_table(img_bgr)
    balls         = detect_balls(img_bgr, mask)

    pred_rows = []
    for (x, y, r) in balls:
        roi   = crop_ball_roi(img_bgr, x, y, r)
        stats = compute_patch_statistics(roi)
        if stats is None: continue
        row = {"filename": image_name, "x": int(x), "y": int(y), "r": int(r),
               **stats,
               "white_center_minus_ring": stats["white_ratio_center"] - stats["white_ratio_ring"],
               "ball_type_hint": None}
        pred_rows.append(predict_single(row))

    resolved  = resolve_global(pred_rows)
    top_view  = get_top_view(img_bgr, order_corners(corners))
    return resolved, top_view

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOP_VIEW_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_JSON) as f:
        image_names = json.load(f)["images"]
    image_paths = [IMAGE_DIR / n for n in image_names]
    print(f"Found {len(image_paths)} images.")

    results = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  Could not read {path.name}, skipping.")
            continue
        try:
            pred_rows, top_view = predict_image(img, path.name)
        except Exception as e:
            print(f"  Error [{path.name}]: {e}")
            pred_rows, top_view = [], img

        cv2.imwrite(str(TOP_VIEW_DIR / path.name), top_view)

        balls_out = [
            {"label": r["final_global_label"],
             "ball_number": BALL_LABEL_TO_NUMBER.get(r["final_global_label"], -1),
             "x": int(r["x"]), "y": int(r["y"]), "r": int(r["r"]),
             "bbox": [int(r["x"])-int(r["r"]), int(r["y"])-int(r["r"]),
                      int(r["x"])+int(r["r"]), int(r["y"])+int(r["r"])]}
            for r in pred_rows if r["final_global_label"] != "not_ball"
        ]
        results.append({"image": path.name, "ball_count": len(balls_out), "balls": balls_out})
        print(f"  {path.name}: {len(balls_out)} balls detected.")

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nDone. Results saved to {RESULTS_JSON}")


    if GROUND_TRUTH_CSV.exists():
        print(f"\n{'='*55}")
        print("AVALIAÇÃO CONTRA GROUND TRUTH CSV")
        print(f"{'='*55}")
        eval_res = evaluate_against_csv(results, GROUND_TRUTH_CSV)
        print(f"  Precision : {eval_res['precision']:.4f}")
        print(f"  Recall    : {eval_res['recall']:.4f}")
        print(f"  F1        : {eval_res['f1']:.4f}")
        print(f"\n  Por imagem:")
        for img_r in eval_res["per_image"]:
            print(f"    {img_r['image']:55s}  P={img_r['precision']:.3f}  R={img_r['recall']:.3f}  F1={img_r['f1']:.3f}  (TP={img_r['tp']} FP={img_r['fp']} FN={img_r['fn']})")

        eval_out = OUTPUT_DIR / "evaluation.json"
        with open(eval_out, "w", encoding="utf-8") as f:
            json.dump(eval_res, f, indent=2)
        print(f"\n  Detalhes guardados em {eval_out}")
    else:
        print(f"\n  (CSV não encontrado em {GROUND_TRUTH_CSV}, avaliação ignorada)")

if __name__ == "__main__":
    main()