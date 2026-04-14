"""
Pool Ball Detection Pipeline
============================
Reads images from input/images/ (listed in input/input.json),
detects and identifies billiard balls, generates top-view warps,
and writes output/results.json + output/top_view/*.jpg
"""

import os
import json
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────
# Directory structure
# ─────────────────────────────────────────────────────────────
INPUT_JSON  = Path("input/input.json")
IMAGE_DIR   = Path("input/images")
OUTPUT_DIR  = Path("output")
TOP_VIEW_DIR = OUTPUT_DIR / "top_view"
RESULTS_JSON = OUTPUT_DIR / "results.json"

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
BALL_LABEL_TO_NUMBER = {
    "cue": 0,
    "yellow_solid": 1,
    "blue_solid": 2,
    "red_solid": 3,
    "purple_solid": 4,
    "orange_solid": 5,
    "green_solid": 6,
    "maroon_solid": 7,
    "black": 8,
    "yellow_stripe": 9,
    "blue_stripe": 10,
    "red_stripe": 11,
    "purple_stripe": 12,
    "orange_stripe": 13,
    "green_stripe": 14,
    "maroon_stripe": 15,
}

VALID_LABELS = set(BALL_LABEL_TO_NUMBER.keys())
CHROMATIC_CLASSES = ["yellow", "blue", "red", "purple", "orange", "green", "maroon"]
REQUIRED_UNIQUE_LABELS = ["cue", "black"]

FEATURE_NAMES = [
    "hue_cos", "hue_sin", "s_median", "v_median",
    "white_ratio", "black_ratio", "colour_ratio",
    "h_std", "v_p90", "v_std", "s_p90", "stripe_score",
]
KNN_K = 5

WHITE_LOW  = np.array([0,   0, 160], dtype=np.uint8)
WHITE_HIGH = np.array([179, 70, 255], dtype=np.uint8)
BLACK_LOW  = np.array([0,   0,   0], dtype=np.uint8)
BLACK_HIGH = np.array([179, 255, 55], dtype=np.uint8)

ROI_PAD = 2.0
INNER_RADIUS_FACTOR = 0.7

GLOBAL_WEIGHTS = {
    "final_conf": 2.0,
    "base_conf": 0.5,
    "type_conf": 0.3,
    "validity_score": 0.8,
}

# ─────────────────────────────────────────────────────────────
# Utils (previously in utils.py)
# ─────────────────────────────────────────────────────────────

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, clear_output
import os
import random
import pandas as pd
import glob

# Helper to show images side by side
def show(imgs, titles=None, figsize=(16, 5)):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def order_corners(pts):
    """
    Order 4 corner points as: TopLeft, TopRight, BottomRight, BottomLeft:
        1. Sort by y to get top vs bottom 
        2. Sort by x inside each pair
    """
    sorted_by_y = pts[np.argsort(pts[:, 1])]

    top = sorted_by_y[:2]
    top = top[np.argsort(top[:, 0])]  # left first
    bottom = sorted_by_y[2:]
    bottom = bottom[np.argsort(bottom[:, 0])]  # left first

    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

def annotate_image_dataset(image_folder):
    """
    Iterates through images in a folder, displays them, 
    and prompts the user for the number of balls.
    Returns a pandas DataFrame with the filenames and counts.
    """
    annotations = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    # Check if directory exists
    if not os.path.exists(image_folder):
        print(f"Error: Directory '{image_folder}' not found.")
        return pd.DataFrame()

    # Filter and sort only image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        print(f"No images found in {image_folder}.")
        return pd.DataFrame()

    for filename in image_files:
        img_path = os.path.join(image_folder, filename)
        
        # Read and show the image
        img = mpimg.imread(img_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"File: {filename}")
        plt.show()
        
        # Keep asking until a valid integer or 'q' is provided
        while True:
            try:
                user_input = input(f"Enter the number of balls for '{filename}' (or type 'q' to quit): ")
                
                if user_input.lower() == 'q':
                    print("Manual annotation interrupted.")
                    clear_output(wait=True)
                    return pd.DataFrame(annotations, columns=['filename', 'ball_count'])
                
                num_balls = int(user_input)
                
                if num_balls < 0:
                    print("Error: The number of balls cannot be negative.")
                    continue
                
                break # Valid input, exit while loop
                
            except ValueError:
                print("Error: Invalid input. Please enter an integer.")
        
        # Store the result
        annotations.append({'filename': filename, 'ball_count': num_balls})
        
        # Clear the cell output to simulate the "game" transition
        clear_output(wait=True)
        
    print("Annotation process finished successfully!")
    
    # Create and return the DataFrame
    df_results = pd.DataFrame(annotations, columns=['filename', 'ball_count'])
    return df_results


# ─────────────────────────────────────────────────────────────
# Stage 0 — Table segmentation & top-view warp
# ─────────────────────────────────────────────────────────────

def segment_table(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask  = cv2.inRange(hsv, np.array([90, 80, 50]),  np.array([130, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 80, 50]),  np.array([85,  255, 255]))
    mask = cv2.bitwise_or(blue_mask, green_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(table_contour, True)
    approx  = cv2.approxPolyDP(table_contour, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        rect    = cv2.minAreaRect(table_contour)
        corners = cv2.boxPoints(rect).astype(np.float32)

    corners    = order_corners(corners)
    clean_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(clean_mask, [corners.astype(np.int32)], 255)
    return clean_mask, corners


def expand_corners(corners, expand_px=20):
    corners = np.array(corners, dtype=np.float32)
    center  = corners.mean(axis=0)
    expanded = []
    for pt in corners:
        direction = pt - center
        norm      = np.linalg.norm(direction) + 1e-9
        expanded.append(pt + (direction / norm) * expand_px)
    return np.array(expanded, dtype=np.float32)


def get_top_view(img, corners, expand_px=20, table_ratio=2.0):
    corners_exp = expand_corners(corners, expand_px=expand_px)
    tl, tr, br, bl = corners_exp

    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(W * table_ratio) if table_ratio else int(
        max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    )

    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(corners_exp, dst)
    warped = cv2.warpPerspective(img, M, (W, H))

    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped, M, corners_exp


# ─────────────────────────────────────────────────────────────
# Stage 0 — Ball detection (Hough + purple blob)
# ─────────────────────────────────────────────────────────────

def detect_balls(img, mask):
    masked = cv2.bitwise_and(img, img, mask=mask)
    hsv    = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    s_chan = cv2.convertScaleAbs(s_chan, alpha=1.35, beta=10)
    v_chan = cv2.convertScaleAbs(v_chan, alpha=1.15, beta=8)
    hsv_boosted   = cv2.merge([h_chan, s_chan, v_chan])
    masked_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    cloth_mean = np.array(cv2.mean(masked_boosted, mask=mask)[:3])
    diff = np.sqrt(np.sum((masked_boosted.astype(np.float32) - cloth_mean) ** 2, axis=2))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    gray_raw = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2GRAY)
    hsv2     = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2HSV)
    _, s2, _ = cv2.split(hsv2)

    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe  = clahe.apply(gray_raw)
    s_enhanced  = clahe.apply(s2)

    gray = cv2.addWeighted(diff, 0.9, s_enhanced, 0.1, 0)
    gray = cv2.addWeighted(gray, 0.9, gray_raw,   0.1, 0)
    gray = cv2.GaussianBlur(gray, (3, 3), 2)

    gray_enhanced = cv2.addWeighted(diff, 0.8, s_enhanced, 0.2, 0)
    gray_enhanced = cv2.addWeighted(gray_enhanced, 0.85, gray_clahe, 0.15, 0)
    gray_enhanced = cv2.GaussianBlur(gray_enhanced, (3, 3), 2)

    h, w   = img.shape[:2]
    min_r  = max(8, int(h * 0.01))
    max_r  = int(h * 0.025)

    edge = cv2.Canny(gray, 40, 140)

    candidates = []
    for g, p2 in [(gray, 29.3), (gray, 26.5), (gray_enhanced, 26.5)]:
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 1.7,
                             param1=38, param2=p2, minRadius=min_r, maxRadius=max_r)
        if c is not None:
            candidates.extend([(x, y, r, "hough")
                                for x, y, r in np.round(c[0]).astype(int).tolist()])

    # Purple blob detection
    lower_purple = np.array([110, 40, 30],  dtype=np.uint8)
    upper_purple = np.array([175, 255, 255], dtype=np.uint8)
    purple_mask  = cv2.inRange(hsv2, lower_purple, upper_purple)
    purple_mask  = cv2.bitwise_and(purple_mask, mask)
    kernel       = np.ones((3, 3), np.uint8)
    purple_mask  = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN,  kernel)
    purple_mask  = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)

    for cnt in cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area < np.pi * (min_r ** 2) * 0.35 or area > np.pi * (max_r ** 2) * 1.5:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        if 4 * np.pi * area / (perimeter * perimeter) < 0.45:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        candidates.append((int(x), int(y), int(r), "purple"))

    if not candidates:
        return []

    balls = []
    for (x, y, r, source) in candidates:
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            continue

        if source == "purple":
            inner_mask  = np.zeros_like(mask)
            cv2.circle(inner_mask, (x, y), max(2, int(r * 0.65)), 255, -1)
            purple_px   = cv2.countNonZero(cv2.bitwise_and(purple_mask, inner_mask))
            total_px    = cv2.countNonZero(inner_mask)
            if purple_px / max(total_px, 1) < 0.45:
                continue

        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, (x, y), r, 255, -1)

        if cv2.mean(gray, mask=circle_mask)[0] < 25:
            continue

        overlap = cv2.countNonZero(cv2.bitwise_and(mask, circle_mask))
        area    = cv2.countNonZero(circle_mask)
        if area == 0 or overlap / area <= 0.35:
            continue

        ring_mask = np.zeros_like(mask)
        cv2.circle(ring_mask, (x, y), r, 255, 2)
        edge_overlap  = cv2.countNonZero(cv2.bitwise_and(edge, ring_mask))
        circumference = max(2 * np.pi * r, 1)
        if source != "purple" and edge_overlap / circumference < 0.20:
            continue

        balls.append((x, y, r))

    balls.sort(key=lambda b: b[2], reverse=True)

    deduped = []
    for x, y, r in balls:
        if not any(np.hypot(x - x2, y - y2) < max(r, r2) for x2, y2, r2 in deduped):
            deduped.append((x, y, r))

    # Radius correction via local neighbours
    if len(deduped) >= 4:
        corrected = []
        for i, (x, y, r) in enumerate(deduped):
            dists_r = sorted(
                [(np.hypot(x - x2, y - y2), r2) for j, (x2, y2, r2) in enumerate(deduped) if i != j]
            )
            local_radii = [r2 for _, r2 in dists_r[:3]]
            if local_radii:
                med = np.median(local_radii)
                if r < 0.7 * med or r > 1.2 * med:
                    r = int(med)
            corrected.append((x, y, r))
        deduped = corrected

    return deduped


# ─────────────────────────────────────────────────────────────
# Feature extraction (for KNN)
# ─────────────────────────────────────────────────────────────

def build_inner_circle_mask(roi_shape, radius_factor=INNER_RADIUS_FACTOR):
    h, w = roi_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(w // 2, h // 2) * radius_factor), 255, -1)
    return mask


def compute_stripe_score(roi_bgr, r_inner=0.45, r_outer=0.75):
    h, w = roi_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    ri = int(min(cx, cy) * r_inner)
    ro = int(min(cx, cy) * r_outer)

    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_inner, (cx, cy), ri, 255, -1)
    mask_outer = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_outer, (cx, cy), ro, 255, -1)
    mask_ring = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    wm  = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)

    wc = cv2.countNonZero(cv2.bitwise_and(wm, mask_inner))
    wr = cv2.countNonZero(cv2.bitwise_and(wm, mask_ring))
    tc = max(cv2.countNonZero(mask_inner), 1)
    tr = max(cv2.countNonZero(mask_ring),  1)

    ratio_c = wc / tc
    ratio_r = wr / tr
    return float(ratio_r - ratio_c), float(ratio_c), float(ratio_r)


def compute_patch_statistics(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    hsv         = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    circle_mask = build_inner_circle_mask(roi_bgr.shape)

    white_mask  = cv2.bitwise_and(cv2.inRange(hsv, WHITE_LOW,  WHITE_HIGH),  circle_mask)
    black_mask  = cv2.bitwise_and(cv2.inRange(hsv, BLACK_LOW,  BLACK_HIGH),  circle_mask)
    colour_mask = cv2.bitwise_and(circle_mask, cv2.bitwise_not(cv2.bitwise_or(white_mask, black_mask)))

    total_px   = max(cv2.countNonZero(circle_mask), 1)
    white_px   = cv2.countNonZero(white_mask)
    black_px   = cv2.countNonZero(black_mask)
    colour_px  = cv2.countNonZero(colour_mask)

    stats = {
        "white_ratio":  white_px  / total_px,
        "black_ratio":  black_px  / total_px,
        "colour_ratio": colour_px / total_px,
        "total_pixels": total_px,
        "colour_pixels": colour_px,
    }

    if colour_px > 0:
        h_v = hsv[:, :, 0][colour_mask > 0].astype(float)
        s_v = hsv[:, :, 1][colour_mask > 0].astype(float)
        v_v = hsv[:, :, 2][colour_mask > 0].astype(float)

        stats.update({
            "h_median": float(np.median(h_v)),
            "s_median": float(np.median(s_v)),
            "v_median": float(np.median(v_v)),
            "h_mean":   float(np.mean(h_v)),
            "s_mean":   float(np.mean(s_v)),
            "v_mean":   float(np.mean(v_v)),
            "h_p10":    float(np.percentile(h_v, 10)),
            "h_p90":    float(np.percentile(h_v, 90)),
            "h_std":    float(np.std(h_v)),
            "s_p10":    float(np.percentile(s_v, 10)),
            "s_p90":    float(np.percentile(s_v, 90)),
            "v_p10":    float(np.percentile(v_v, 10)),
            "v_p90":    float(np.percentile(v_v, 90)),
            "v_std":    float(np.std(v_v)),
        })
    else:
        for k in ["h_median","s_median","v_median","h_mean","s_mean","v_mean",
                  "h_p10","h_p90","h_std","s_p10","s_p90","v_p10","v_p90","v_std"]:
            stats[k] = None

    stripe_score, ratio_center, ratio_ring = compute_stripe_score(roi_bgr)
    stats["stripe_score"]       = stripe_score
    stats["white_ratio_center"] = ratio_center
    stats["white_ratio_ring"]   = ratio_ring
    return stats


def crop_ball_roi(img_bgr, x, y, r, pad=ROI_PAD):
    h, w = img_bgr.shape[:2]
    rr   = int(round(r * pad))
    x1   = max(0, int(round(x - rr)))
    y1   = max(0, int(round(y - rr)))
    x2   = min(w, int(round(x + rr)))
    y2   = min(h, int(round(y + rr)))
    return img_bgr[y1:y2, x1:x2].copy()


# ─────────────────────────────────────────────────────────────
# Stage 1 — Validity scoring
# ─────────────────────────────────────────────────────────────

def safe_float(v, default=0.0):
    if v is None:
        return float(default)
    try:
        if np.isnan(v):
            return float(default)
    except Exception:
        pass
    return float(v)


def predict_validity(row):
    wr   = safe_float(row.get("white_ratio"))
    br   = safe_float(row.get("black_ratio"))
    cr   = safe_float(row.get("colour_ratio"))
    hs   = safe_float(row.get("h_std"))
    vs   = safe_float(row.get("v_std"))
    sm   = safe_float(row.get("s_median"))
    wrc  = safe_float(row.get("white_ratio_center"))
    wrr  = safe_float(row.get("white_ratio_ring"))
    wcmr = wrc - wrr

    score = 0.0

    likely_cue        = wr >= 0.30 and wrc >= 0.70 and br <= 0.03
    likely_dark_ball  = br >= 0.20 and cr >= 0.60 and vs >= 30.0
    likely_uniform    = cr >= 0.97 and br <= 0.05 and wrc <= 0.02 and wr <= 0.02 and vs >= 18.0
    likely_dark_art   = br >= 0.30 and hs <= 10.0 and wrc <= 0.02
    likely_black_ball = br >= 0.14 and cr >= 0.60 and vs >= 30.0 and wrc <= 0.03

    if likely_cue:       score += 3.0
    if likely_dark_ball: score += 1.5
    if likely_uniform:   score += 1.2
    if likely_dark_art:  score -= 2.0
    if likely_black_ball:score += 1.8

    if wrc >= 0.02:  score += 1.0
    if wr  >= 0.02:  score += 0.5
    if hs  >= 12.0:  score += 0.6
    if hs  >= 20.0:  score += 0.6
    if vs  >= 20.0:  score += 0.8
    if vs  >= 40.0:  score += 0.5
    if cr  >= 0.75:  score += 0.2

    if hs < 10.0 and not likely_uniform and not likely_dark_ball: score -= 1.0
    if hs < 5.0  and not likely_uniform and not likely_dark_ball: score -= 1.2
    if sm >= 180.0 and hs < 8.0:     score -= 0.8
    if wrc < 0.01 and not likely_uniform: score -= 1.0
    if wr  < 0.01 and not likely_uniform: score -= 0.6
    if wcmr > 0.10 and not likely_cue:   score -= 0.4
    if br  > 0.22 and not likely_dark_ball: score -= 0.5

    validity_class = "suspect" if score <= -2.0 else "valid_ball"
    return {"validity_class": validity_class, "validity_score": float(score)}


# ─────────────────────────────────────────────────────────────
# Stage 2 — Base colour (rules + KNN)
# ─────────────────────────────────────────────────────────────

def label_to_base_colour(label):
    if label == "cue":    return "white"
    if label == "black":  return "black"
    if label.endswith("_solid"):  return label.replace("_solid", "")
    if label.endswith("_stripe"): return label.replace("_stripe", "")
    return None

def label_to_ball_type(label):
    if label == "cue":    return "cue"
    if label == "black":  return "black"
    if label.endswith("_solid"):  return "solid"
    if label.endswith("_stripe"): return "stripe"
    return None

def final_label_from_parts(base_colour, ball_type):
    if base_colour == "white": return "cue"
    if base_colour == "black": return "black"
    if base_colour is None or ball_type is None: return "not_ball"
    return f"{base_colour}_{ball_type}"


def apply_explicit_base_colour_rules(row):
    br   = safe_float(row.get("black_ratio"))
    wr   = safe_float(row.get("white_ratio"))
    cr   = safe_float(row.get("colour_ratio"))
    hm   = safe_float(row.get("h_median"))
    vm   = safe_float(row.get("v_median"))
    sm   = safe_float(row.get("s_median"))
    vp90 = safe_float(row.get("v_p90"), vm)
    wrc  = safe_float(row.get("white_ratio_center"))
    wrr  = safe_float(row.get("white_ratio_ring"))
    bt   = row.get("ball_type_hint")

    if br >= 0.12 and wr <= 0.12 and vm <= 190:
        return "black"
    if bt == "stripe" and 0.20 <= wr <= 0.27 and br <= 0.02 and cr >= 0.72 and hm <= 95:
        return "yellow"
    if wr >= 0.15 and vp90 >= 190 and sm <= 190 and br <= 0.10 and wrc >= 0.10 and wr >= wrc * 0.6:
        return "white"
    if sm >= 100 and cr >= 0.55:
        if hm <= 7 or hm >= 173:            return "red"
        if 7  < hm <= 16 and vm <= 160:     return "maroon"
        if 8  < hm <= 22 and vm > 130:      return "orange"
    return None


def build_feature_vector_from_row(row):
    h    = row.get("h_median")
    hcos, hsin = (0.0, 0.0) if h is None else (
        float(np.cos(2 * np.pi * h / 180)),
        float(np.sin(2 * np.pi * h / 180)),
    )
    return np.array([
        hcos, hsin,
        safe_float(row.get("s_median")),
        safe_float(row.get("v_median")),
        safe_float(row.get("white_ratio")),
        safe_float(row.get("black_ratio")),
        safe_float(row.get("colour_ratio")),
        safe_float(row.get("h_std")),
        safe_float(row.get("v_p90")),
        safe_float(row.get("v_std")),
        safe_float(row.get("s_p90")),
        safe_float(row.get("stripe_score")),
    ], dtype=float)


def predict_base_colour_with_probabilities(row, Xz_train, y_train, feature_mean, feature_std):
    rule = apply_explicit_base_colour_rules(row)
    if rule is not None:
        return {"top1": rule, "ranked": [(rule, 1.0)], "confidence": 1.0, "source": "rule"}

    x  = build_feature_vector_from_row(row)
    xz = (x - feature_mean) / feature_std

    dists  = np.linalg.norm(Xz_train - xz, axis=1)
    nn_idx = np.argsort(dists)[:KNN_K]

    votes = defaultdict(float)
    for idx in nn_idx:
        cls = y_train[idx]
        if cls in ("white", "black"):
            continue
        votes[cls] += 1.0 / max(dists[idx], 1e-6)

    if not votes:
        top_cls = y_train[nn_idx[0]]
        return {"top1": top_cls, "ranked": [(top_cls, 1.0)], "confidence": 1.0, "source": "knn_fallback"}

    total  = sum(votes.values())
    ranked = sorted([(c, s / total) for c, s in votes.items()], key=lambda x: x[1], reverse=True)
    return {"top1": ranked[0][0], "ranked": ranked, "confidence": float(ranked[0][1]), "source": "knn"}


# ─────────────────────────────────────────────────────────────
# Stage 3 — Ball type (solid vs stripe)
# ─────────────────────────────────────────────────────────────

def predict_ball_type_with_probabilities(row, predicted_base_colour):
    if predicted_base_colour == "white":
        return {"top1": "cue",   "ranked": [("cue",   1.0)], "confidence": 1.0, "source": "forced"}
    if predicted_base_colour == "black":
        return {"top1": "black", "ranked": [("black", 1.0)], "confidence": 1.0, "source": "forced"}

    ss   = safe_float(row.get("stripe_score"))
    wr   = safe_float(row.get("white_ratio"))
    wrc  = safe_float(row.get("white_ratio_center"))
    wrr  = safe_float(row.get("white_ratio_ring"))
    hs   = safe_float(row.get("h_std"))
    cr   = safe_float(row.get("colour_ratio"))

    if predicted_base_colour == "purple":
        score = -5.0 if (hs < 16.0 and cr > 0.93) else -3.5 * ss
    else:
        score = -3.5 * ss
        if ss < -0.03 and wrr > 0.01:
            score += 1.5 * (wrc - wrr)
        score += 0.8 * (wr - 0.10)

    prob_stripe = 1.0 / (1.0 + np.exp(-6.0 * score))
    prob_solid  = 1.0 - prob_stripe
    ranked = sorted([("stripe", prob_stripe), ("solid", prob_solid)], key=lambda x: x[1], reverse=True)
    return {"top1": ranked[0][0], "ranked": ranked, "confidence": float(ranked[0][1]), "source": "rule_score"}


def build_ranked_final_predictions(base_colour_pred, ball_type_pred):
    options = []
    for colour_cls, colour_prob in base_colour_pred["ranked"]:
        if colour_cls in ("white", "black"):
            ranked_types = [("cue", 1.0)] if colour_cls == "white" else [("black", 1.0)]
        else:
            ranked_types = ball_type_pred["ranked"] if colour_cls == base_colour_pred["top1"] \
                           else [("solid", 0.5), ("stripe", 0.5)]
        for type_cls, type_prob in ranked_types:
            options.append({
                "final_label": final_label_from_parts(colour_cls, type_cls),
                "base_colour": colour_cls,
                "ball_type":   type_cls,
                "joint_prob":  float(colour_prob) * float(type_prob),
            })

    dedup = {}
    for item in sorted(options, key=lambda d: d["joint_prob"], reverse=True):
        if item["final_label"] not in dedup:
            dedup[item["final_label"]] = item
    return list(dedup.values())


def predict_single_detection(row, Xz_train, y_train, feature_mean, feature_std):
    validity        = predict_validity(row)
    base_colour_pred = predict_base_colour_with_probabilities(row, Xz_train, y_train, feature_mean, feature_std)
    ball_type_pred  = predict_ball_type_with_probabilities(row, base_colour_pred["top1"])
    ranked_final    = build_ranked_final_predictions(base_colour_pred, ball_type_pred)

    pred_final_label = ranked_final[0]["final_label"] if ranked_final else "not_ball"
    pred_final_conf  = ranked_final[0]["joint_prob"]  if ranked_final else 0.0

    out = dict(row)
    out.update(validity)
    out["pred_base_colour"]       = base_colour_pred["top1"]
    out["pred_base_colour_conf"]  = float(base_colour_pred["confidence"])
    out["pred_ball_type"]         = ball_type_pred["top1"]
    out["pred_ball_type_conf"]    = float(ball_type_pred["confidence"])
    out["pred_final_label"]       = pred_final_label
    out["pred_final_conf"]        = float(pred_final_conf)
    out["ranked_final_predictions"] = ranked_final
    return out


# ─────────────────────────────────────────────────────────────
# Stage 4 — Global consistency
# ─────────────────────────────────────────────────────────────

def combined_strength(row):
    penalty = -1.5 if row["validity_class"] == "suspect" else 0.0
    return (
        GLOBAL_WEIGHTS["final_conf"]    * float(row["pred_final_conf"]) +
        GLOBAL_WEIGHTS["base_conf"]     * float(row["pred_base_colour_conf"]) +
        GLOBAL_WEIGHTS["type_conf"]     * float(row["pred_ball_type_conf"]) +
        GLOBAL_WEIGHTS["validity_score"]* float(row["validity_score"]) +
        penalty
    )

def get_allowed_count_for_label(label):
    if label == "not_ball": return 999999
    return 1  # at most 1 of each ball

def get_label_prob_from_ranked(row, wanted_label):
    for opt in row["ranked_final_predictions"]:
        if opt["final_label"] == wanted_label:
            return float(opt["joint_prob"])
    return 0.0

def choose_required_candidate(rows, required_label):
    candidates = []
    for idx, row in enumerate(rows):
        prob = get_label_prob_from_ranked(row, required_label)
        if prob <= 0.0:
            continue
        tier  = 0 if row["validity_class"] == "valid_ball" else 1
        score = row["strength"] + 2.0 * prob
        candidates.append((tier, score, idx, prob))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], -x[1]))
    _, _, idx, prob = candidates[0]
    return idx, prob

def try_reassign_row(row, assigned_labels, forbidden=None):
    counts = Counter(assigned_labels)
    for option in row["ranked_final_predictions"]:
        cand = option["final_label"]
        prob = float(option["joint_prob"])
        if cand == forbidden or cand == "not_ball":
            continue
        if counts[cand] < get_allowed_count_for_label(cand):
            return cand, prob, "reassigned"
    return row["final_global_label"], row["final_global_conf"], "kept_duplicate"

def force_required_labels(rows):
    for required in REQUIRED_UNIQUE_LABELS:
        winner = choose_required_candidate(rows, required)
        if winner is None:
            continue
        widx, wprob = winner
        rows[widx]["final_global_label"] = required
        rows[widx]["final_global_conf"]  = wprob
        rows[widx]["global_resolution"]  = f"forced_{required}"
    return rows

def resolve_overflow_conflicts(rows):
    changed = True
    while changed:
        changed = False
        grouped = defaultdict(list)
        for idx, row in enumerate(rows):
            if row["final_global_label"] != "not_ball":
                grouped[row["final_global_label"]].append((idx, row))
        for label, items in grouped.items():
            allowed = get_allowed_count_for_label(label)
            if len(items) <= allowed:
                continue
            items.sort(key=lambda x: x[1]["strength"], reverse=True)
            overflow_ids = {idx for idx, _ in items[allowed:]}
            kept = [rows[i]["final_global_label"] for i in range(len(rows))
                    if rows[i]["final_global_label"] != "not_ball" and i not in overflow_ids]
            for idx, row in items[allowed:]:
                nl, np_, mode = try_reassign_row(row, kept, forbidden=label)
                if nl != row["final_global_label"]:
                    rows[idx]["final_global_label"] = nl
                    rows[idx]["final_global_conf"]  = np_
                    rows[idx]["global_resolution"]  = mode
                    changed = True
    return rows

def resolve_global_consistency(pred_rows):
    rows = [dict(r) for r in pred_rows]
    for row in rows:
        row["strength"]           = combined_strength(row)
        row["final_global_label"] = row["pred_final_label"]
        row["final_global_conf"]  = float(row["pred_final_conf"])
        row["global_resolution"]  = "kept_top1"

    rows = force_required_labels(rows)
    rows = resolve_overflow_conflicts(rows)
    rows = force_required_labels(rows)
    rows = resolve_overflow_conflicts(rows)

    for row in rows:
        if row["final_global_label"] in ("cue", "black"):
            continue
        if row["validity_class"] == "suspect" and row["strength"] < -1:
            row["final_global_label"] = "not_ball"
            row["final_global_conf"]  = 0.0
            row["global_resolution"]  = "weak_suspect_demoted"
        if row["validity_score"] <= -3.5:
            row["final_global_label"] = "not_ball"
            row["final_global_conf"]  = 0.0
            row["global_resolution"]  = "final_strong_invalid"

    return sorted(rows, key=lambda d: (d["x"], d["y"]))


# ─────────────────────────────────────────────────────────────
# KNN training from detected balls (self-supervised bootstrap)
# ─────────────────────────────────────────────────────────────

def build_knn_from_detections(all_rows_with_rule_pred):
    """
    Build KNN training set using only detections where the
    explicit colour rules fired (source == 'rule'), which are
    the most reliable labels available without ground truth.
    """
    training_rows = []
    for row in all_rows_with_rule_pred:
        if row.get("pred_base_colour_source") == "rule":
            training_rows.append(row)

    if not training_rows:
        # Fallback: use all rows
        training_rows = all_rows_with_rule_pred

    X = np.vstack([build_feature_vector_from_row(r) for r in training_rows])
    y = np.array([r["pred_base_colour"] for r in training_rows], dtype=object)

    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1.0
    Xz = (X - mean) / std
    return Xz, y, mean, std


# ─────────────────────────────────────────────────────────────
# Per-image prediction
# ─────────────────────────────────────────────────────────────

def predict_image(img_bgr, image_name, Xz_train, y_train, feature_mean, feature_std):
    mask, corners = segment_table(img_bgr)
    balls = detect_balls(img_bgr, mask)

    pred_rows = []
    for (x, y, r) in balls:
        roi  = crop_ball_roi(img_bgr, x, y, r)
        stats = compute_patch_statistics(roi)
        if stats is None:
            continue

        row = {
            "filename": image_name,
            "x": int(x), "y": int(y), "r": int(r),
            **stats,
            "white_center_minus_ring": stats["white_ratio_center"] - stats["white_ratio_ring"],
            "ball_type_hint": None,
        }
        pred = predict_single_detection(row, Xz_train, y_train, feature_mean, feature_std)
        pred["pred_base_colour_source"] = pred.get("pred_base_colour_conf") == 1.0 and \
            apply_explicit_base_colour_rules(row) is not None
        pred_rows.append(pred)

    pred_rows = resolve_global_consistency(pred_rows)

    # Top view
    corners   = order_corners(corners)
    top_view, _, _ = get_top_view(img_bgr, corners)

    return pred_rows, top_view


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOP_VIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Read image list
    with open(INPUT_JSON, "r") as f:
        input_data = json.load(f)
    image_names = input_data["images"]

    image_paths = [IMAGE_DIR / name for name in image_names]

    print(f"Found {len(image_paths)} images.")

    # ── Bootstrap KNN with a rule-only first pass ─────────────
    print("First pass: extracting rule-based labels for KNN bootstrap...")
    bootstrap_rows = []
    dummy_Xz    = np.zeros((1, len(FEATURE_NAMES)))
    dummy_y     = np.array(["yellow"])
    dummy_mean  = np.zeros(len(FEATURE_NAMES))
    dummy_std   = np.ones(len(FEATURE_NAMES))

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        try:
            mask, corners = segment_table(img)
            balls = detect_balls(img, mask)
            for (x, y, r) in balls:
                roi   = crop_ball_roi(img, x, y, r)
                stats = compute_patch_statistics(roi)
                if stats is None:
                    continue
                row = {"filename": path.name, "x": int(x), "y": int(y), "r": int(r),
                       **stats,
                       "white_center_minus_ring": stats["white_ratio_center"] - stats["white_ratio_ring"],
                       "ball_type_hint": None}
                rule = apply_explicit_base_colour_rules(row)
                if rule:
                    row["pred_base_colour"] = rule
                    row["pred_base_colour_source"] = "rule"
                    bootstrap_rows.append(row)
        except Exception as e:
            print(f"  Warning [{path.name}]: {e}")

    if bootstrap_rows:
        Xz_train, y_train, feat_mean, feat_std = build_knn_from_detections(bootstrap_rows)
        print(f"  KNN trained on {len(bootstrap_rows)} rule-labeled detections.")
    else:
        Xz_train, y_train, feat_mean, feat_std = dummy_Xz, dummy_y, dummy_mean, dummy_std
        print("  Warning: no rule-labeled detections found; KNN will use fallback.")

    # ── Main prediction pass ──────────────────────────────────
    print("Second pass: full pipeline...")
    results = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  Could not read {path.name}, skipping.")
            continue

        try:
            pred_rows, top_view = predict_image(
                img, path.name, Xz_train, y_train, feat_mean, feat_std
            )
        except Exception as e:
            print(f"  Error [{path.name}]: {e}")
            pred_rows = []
            top_view  = img

        # Save top view
        top_view_path = TOP_VIEW_DIR / path.name
        cv2.imwrite(str(top_view_path), top_view)

        # Build result entry
        balls_out = []
        for row in pred_rows:
            label = row["final_global_label"]
            if label == "not_ball":
                continue
            x, y, r = int(row["x"]), int(row["y"]), int(row["r"])
            balls_out.append({
                "label":       label,
                "ball_number": BALL_LABEL_TO_NUMBER.get(label, -1),
                "x": x, "y": y, "r": r,
                "bbox": [x - r, y - r, x + r, y + r],
            })

        results.append({
            "image":      path.name,
            "ball_count": len(balls_out),
            "balls":      balls_out,
        })
        print(f"  {path.name}: {len(balls_out)} balls detected.")

    # Write results.json
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_JSON}")
    print(f"Top-view images saved to {TOP_VIEW_DIR}/")


if __name__ == "__main__":
    main()