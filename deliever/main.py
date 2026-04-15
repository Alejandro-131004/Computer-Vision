
import os
import sys
import json
import math
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict, Counter


# =============================================================
# Ball label mappings
# =============================================================

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

REQUIRED_UNIQUE_LABELS = ["cue", "black"]
UNIQUE_PER_IMAGE = {"cue", "black"}

# =============================================================
# Optimized parameters (loaded from JSON, fallback hardcoded)
# =============================================================

OPTIMIZED_PARAMS = {
    "s1_cue_white_ratio_min": 0.1,
    "s1_cue_white_center_min": 0.65,
    "s1_cue_black_ratio_max": 0.0,
    "s1_cue_score_bonus": 3.0,
    "s1_dark_black_ratio_min": 0.2,
    "s1_dark_colour_ratio_min": 0.6,
    "s1_dark_v_std_min": 30.0,
    "s1_dark_score_bonus": 4.5,
    "s1_uniform_colour_ratio_min": 0.96,
    "s1_uniform_black_ratio_max": 0.11,
    "s1_uniform_white_center_max": 0.03,
    "s1_uniform_white_ratio_max": 0.07,
    "s1_uniform_v_std_min": 4.0,
    "s1_uniform_score_bonus": 1.0,
    "s1_artifact_black_ratio_min": 0.05,
    "s1_artifact_h_std_max": 5.0,
    "s1_artifact_white_center_max": 0.02,
    "s1_artifact_score_penalty": -3.2,
    "s1_pos_white_center_min": 0.1,
    "s1_pos_white_center_bonus": 0.7,
    "s1_pos_white_ratio_min": 0.04,
    "s1_pos_white_ratio_bonus": 0.0,
    "s1_pos_h_std_good_min": 2.0,
    "s1_pos_h_std_good_bonus": 0.6,
    "s1_pos_h_std_vgood_min": 18.0,
    "s1_pos_h_std_vgood_bonus": 0.0,
    "s1_pos_v_std_good_min": 24.0,
    "s1_pos_v_std_good_bonus": 0.3,
    "s1_pos_v_std_vgood_min": 52.0,
    "s1_pos_v_std_vgood_bonus": 0.5,
    "s1_pos_colour_ratio_min": 0.68,
    "s1_pos_colour_ratio_bonus": 0.1,
    "s1_sus_h_std_low_thr": 10.0,
    "s1_sus_h_std_low_penalty": -0.4,
    "s1_sus_h_std_vlow_thr": 0.0,
    "s1_sus_h_std_vlow_penalty": -1.2,
    "s1_sus_sat_thr": 205.0,
    "s1_sus_sat_h_std_thr": 12.0,
    "s1_sus_sat_penalty": -0.8,
    "s1_sus_white_center_vlow_thr": 0.01,
    "s1_sus_white_center_vlow_pen": -1.2,
    "s1_sus_white_ratio_vlow_thr": 0.01,
    "s1_sus_white_ratio_vlow_pen": -0.6,
    "s1_sus_center_minus_ring_thr": 0.14,
    "s1_sus_center_minus_ring_pen": -0.4,
    "s1_sus_black_ratio_thr": 0.22,
    "s1_sus_black_ratio_pen": -0.5,
    "s1_black_black_ratio_min": 0.06,
    "s1_black_colour_ratio_min": 0.85,
    "s1_black_v_std_min": 5.0,
    "s1_black_white_center_max": 0.04,
    "s1_black_score_bonus": 2.0,
    "s1_suspect_score_thr": -0.8,
    "s2_black_black_ratio_min": 0.14,
    "s2_black_white_ratio_max": 0.16,
    "s2_black_v_median_max": 190.0,
    "s2_yellow_white_ratio_lo": 0.35,
    "s2_yellow_white_ratio_hi": 0.26,
    "s2_yellow_black_ratio_max": 0.0,
    "s2_yellow_colour_ratio_min": 0.8,
    "s2_yellow_h_median_max": 100.0,
    "s2_white_white_ratio_min": 0.05,
    "s2_white_v_p90_min": 210.0,
    "s2_white_s_median_max": 80.0,
    "s2_white_black_ratio_max": 0.16,
    "s2_white_center_min": 0.1,
    "s2_white_center_ratio": 0.3,
    "s2_chrom_s_min": 100.0,
    "s2_chrom_colour_ratio_min": 0.55,
    "s2_blue_center": 105.0,
    "s2_blue_tol": 12.0,
    "s2_blue_s_min": 125.0,
    "s2_blue_v_min": 110.0,
    "s2_blue_score_bias": 0.2,
    "s2_yellow_center": 20.0,
    "s2_yellow_tol": 14.0,
    "s2_yellow_s_min": 100.0,
    "s2_yellow_v_min": 120.0,
    "s2_yellow_score_bias": -0.2,
    "s2_purple_center": 113.0,
    "s2_purple_tol": 21.0,
    "s2_purple_s_min": 55.0,
    "s2_purple_v_min": 85.0,
    "s2_purple_score_bias": 0.0,
    "s2_red_center1": 8.0,
    "s2_red_center2": 175.0,
    "s2_red_tol": 12.0,
    "s2_red_s_min": 110.0,
    "s2_red_v_min": 135.0,
    "s2_red_score_bias": -0.2,
    "s2_orange_center": 12.0,
    "s2_orange_tol": 8.0,
    "s2_orange_s_min": 85.0,
    "s2_orange_v_min": 180.0,
    "s2_orange_score_bias": 0.4,
    "s2_green_center": 76.0,
    "s2_green_tol": 28.0,
    "s2_green_s_min": 96.6,
    "s2_green_v_min": 70.7,
    "s2_green_score_bias": -1.0,
    "s2_maroon_center": 17.0,
    "s2_maroon_tol": 16.0,
    "s2_maroon_s_min": 70.0,
    "s2_maroon_v_max": 150.0,
    "s2_maroon_score_bias": 0.2,
    "s2_manual_h_std_min": 1.0,
    "s2_manual_score_gap_min": 0.02,
    "s2_manual_unknown_penalty": 0.05,
    "s3_stripe_score_coef": 0.0,
    "s3_white_ratio_ring_thr": 0.01,
    "s3_center_ring_diff_coef": 1.6,
    "s3_white_ratio_offset": 0.13,
    "s3_white_ratio_coef": 1.1,
    "s3_sigmoid_scale": 6.0,
    "s3_purple_h_std_thr": 18.0,
    "s3_purple_colour_ratio_thr": 0.76,
    "s3_purple_solid_score": -3.5,
    "s3_purple_stripe_coef": 4.8,
    "s4_w_final_conf": 3.0,
    "s4_w_base_conf": 0.6,
    "s4_w_type_conf": 0.3,
    "s4_w_validity_score": 0.8,
    "s4_suspect_penalty": -1.5,
    "s4_weak_suspect_strength_thr": -1.0,
    "s4_strong_invalid_score_thr": -3.5,
}

# =============================================================
# HSV colour range constants
# =============================================================

WHITE_LOW  = np.array([0,   0, 160], dtype=np.uint8)
WHITE_HIGH = np.array([179, 70, 255], dtype=np.uint8)
BLACK_LOW  = np.array([0,   0,   0], dtype=np.uint8)
BLACK_HIGH = np.array([179, 255, 55], dtype=np.uint8)

ROI_PAD = 2.0


# =============================================================
# Geometry helpers
# =============================================================

def order_corners(corners):
    """Order 4 corners as top-left, top-right, bottom-right, bottom-left."""
    corners = np.array(corners, dtype=np.float32)
    center  = corners.mean(axis=0)
    angles  = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    order   = np.argsort(angles)
    corners = corners[order]
    # Rotate so top-left is first
    top_idx = np.argmin(corners[:, 0] + corners[:, 1])
    corners = np.roll(corners, -top_idx, axis=0)
    return corners


# =============================================================
# Stage 0a: table segmentation
# =============================================================

def segment_table(img):
    """Detect the billiard table cloth and return (mask, corners)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blue_mask  = cv2.inRange(hsv, np.array([90,  80,  50]), np.array([130, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35,  80,  50]), np.array([ 85, 255, 255]))
    mask = cv2.bitwise_or(blue_mask, green_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: full image mask
        full = np.ones(img.shape[:2], dtype=np.uint8) * 255
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        return full, corners

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


# =============================================================
# Stage 0b: top-view warp
# =============================================================

def get_top_view(img, corners, expand_px=20, table_ratio=2.0):
    """Perspective-warp the table region into a flat top-down view."""
    center   = corners.mean(axis=0)
    expanded = np.array([
        pt + (pt - center) / (np.linalg.norm(pt - center) + 1e-9) * expand_px
        for pt in corners
    ], dtype=np.float32)

    tl, tr, br, bl = expanded
    W   = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H   = int(W * table_ratio)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(expanded, dst)
    warped = cv2.warpPerspective(img, M, (W, H))

    # Rotate to landscape if height > width
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


# =============================================================
# Stage 0b: ball detection
# =============================================================

def detect_balls(img, mask):
    """
    Detect billiard balls in the masked image.
    Returns a list of (x, y, r) tuples.
    """
    masked = cv2.bitwise_and(img, img, mask=mask)
    hsv    = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    # Boost saturation and brightness
    s_chan = cv2.convertScaleAbs(s_chan, alpha=1.35, beta=10)
    v_chan = cv2.convertScaleAbs(v_chan, alpha=1.15, beta=8)

    hsv_boosted    = cv2.merge([h_chan, s_chan, v_chan])
    masked_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    cloth_mean = np.array(cv2.mean(masked_boosted, mask=mask)[:3])
    diff = np.sqrt(np.sum((masked_boosted.astype(np.float32) - cloth_mean) ** 2, axis=2))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    gray_raw = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2GRAY)
    hsv2     = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2HSV)
    _, s_chan2, _ = cv2.split(hsv2)

    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe  = clahe.apply(gray_raw)
    s_enhanced  = clahe.apply(s_chan2)

    gray = cv2.addWeighted(diff, 0.9, s_enhanced, 0.1, 0)
    gray = cv2.addWeighted(gray, 0.9, gray_raw,   0.1, 0)
    gray = cv2.GaussianBlur(gray, (3, 3), 2)

    gray_enhanced = cv2.addWeighted(diff, 0.8, s_enhanced, 0.2, 0)
    gray_enhanced = cv2.addWeighted(gray_enhanced, 0.85, gray_clahe, 0.15, 0)
    gray_enhanced = cv2.GaussianBlur(gray_enhanced, (3, 3), 2)

    h_img, _ = img.shape[:2]
    min_r = max(8, int(h_img * 0.01))
    max_r = int(h_img * 0.025)

    edge = cv2.Canny(gray, 40, 140)

    # Three Hough passes with different sensitivity
    circles1 = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 1.7,
        param1=38, param2=29.3, minRadius=min_r, maxRadius=max_r,
    )
    circles2 = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 1.7,
        param1=38, param2=26.5, minRadius=min_r, maxRadius=max_r,
    )
    circles3 = cv2.HoughCircles(
        gray_enhanced, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 1.7,
        param1=38, param2=26.5, minRadius=min_r, maxRadius=max_r,
    )

    candidates = []
    for circles in (circles1, circles2, circles3):
        if circles is not None:
            for x, y, r in np.round(circles[0]).astype(int).tolist():
                candidates.append((x, y, r, "hough"))

    # Purple blob detection (Hough misses dark purple balls)
    lower_purple = np.array([110, 40, 30], dtype=np.uint8)
    upper_purple = np.array([175, 255, 255], dtype=np.uint8)
    purple_mask  = cv2.inRange(hsv2, lower_purple, upper_purple)
    purple_mask  = cv2.bitwise_and(purple_mask, mask)

    kernel      = np.ones((3, 3), np.uint8)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN,  kernel)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * (min_r ** 2) * 0.35 or area > np.pi * (max_r ** 2) * 1.5:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        candidates.append((int(x), int(y), int(r), "purple"))

    if not candidates:
        return []

    # Filter candidates
    balls = []
    for (x, y, r, source) in candidates:
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            continue

        circle_mask = np.zeros_like(mask)

        if source == "purple":
            inner_mask = np.zeros_like(mask)
            cv2.circle(inner_mask, (x, y), max(2, int(r * 0.65)), 255, -1)
            purple_pixels = cv2.countNonZero(cv2.bitwise_and(purple_mask, inner_mask))
            total_pixels  = cv2.countNonZero(inner_mask)
            if total_pixels == 0 or purple_pixels / total_pixels < 0.45:
                continue

        cv2.circle(circle_mask, (x, y), r, 255, -1)

        mean_val = cv2.mean(gray, mask=circle_mask)[0]
        if mean_val < 25:
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

    # Sort largest first, then deduplicate
    balls.sort(key=lambda b: b[2], reverse=True)
    deduped = []
    for x, y, r in balls:
        duplicate = any(
            np.hypot(x - x2, y - y2) < 1.0 * max(r, r2)
            for x2, y2, r2 in deduped
        )
        if not duplicate:
            deduped.append((x, y, r))

    # Correct outlier radii using local neighbourhood median
    if len(deduped) >= 4:
        corrected = []
        for i, (x, y, r) in enumerate(deduped):
            neighbors = sorted(
                [(np.hypot(x - x2, y - y2), r2) for j, (x2, y2, r2) in enumerate(deduped) if i != j],
                key=lambda t: t[0],
            )
            local_radii = [r2 for _, r2 in neighbors[:3]]
            if local_radii:
                local_med = float(np.median(local_radii))
                if r < 0.7 * local_med or r > 1.2 * local_med:
                    r = int(local_med)
            corrected.append((x, y, r))
        deduped = corrected

    return deduped


# =============================================================
# Feature extraction
# =============================================================

def _build_inner_circle_mask(roi_shape, radius_factor=0.45):
    """Tight circular mask to avoid cloth pixel contamination."""
    h, w   = roi_shape[:2]
    mask   = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    radius = int(min(cx, cy) * radius_factor)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def _compute_stripe_score(roi_bgr, r_factor_inner=0.45, r_factor_outer=0.75):
    """
    Return (stripe_score, ratio_center, ratio_ring).
    High stripe_score means more white in the ring than the center (stripe ball).
    """
    h, w   = roi_bgr.shape[:2]
    cx, cy = w // 2, h // 2

    r_inner = int(min(cx, cy) * r_factor_inner)
    r_outer = int(min(cx, cy) * r_factor_outer)

    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_inner, (cx, cy), r_inner, 255, -1)

    mask_outer = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_outer, (cx, cy), r_outer, 255, -1)
    mask_ring = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

    hsv        = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)

    wc = cv2.countNonZero(cv2.bitwise_and(white_mask, mask_inner))
    wr = cv2.countNonZero(cv2.bitwise_and(white_mask, mask_ring))
    tc = max(cv2.countNonZero(mask_inner), 1)
    tr = max(cv2.countNonZero(mask_ring),  1)

    ratio_center = wc / tc
    ratio_ring   = wr / tr
    return float(ratio_ring - ratio_center), float(ratio_center), float(ratio_ring)


def compute_patch_statistics(roi_bgr):
    """
    Compute colour statistics from a cropped ball ROI.
    Uses a tight inner circle mask (radius_factor=0.45) matching the annotation pipeline.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return {}

    hsv          = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    circle_mask  = _build_inner_circle_mask(roi_bgr.shape, radius_factor=0.45)

    white_mask = cv2.bitwise_and(cv2.inRange(hsv, WHITE_LOW,  WHITE_HIGH), circle_mask)
    black_mask = cv2.bitwise_and(cv2.inRange(hsv, BLACK_LOW,  BLACK_HIGH), circle_mask)
    colour_mask = cv2.bitwise_and(
        circle_mask,
        cv2.bitwise_not(cv2.bitwise_or(white_mask, black_mask)),
    )

    total_px   = cv2.countNonZero(circle_mask)
    colour_px  = cv2.countNonZero(colour_mask)

    stats = {
        "white_ratio":  float(cv2.countNonZero(white_mask)  / max(total_px, 1)),
        "black_ratio":  float(cv2.countNonZero(black_mask)  / max(total_px, 1)),
        "colour_ratio": float(colour_px / max(total_px, 1)),
    }

    if colour_px > 0:
        h_vals = hsv[:, :, 0][colour_mask > 0]
        s_vals = hsv[:, :, 1][colour_mask > 0]
        v_vals = hsv[:, :, 2][colour_mask > 0]

        stats["h_median"] = float(np.median(h_vals))
        stats["s_median"] = float(np.median(s_vals))
        stats["v_median"] = float(np.median(v_vals))
        stats["h_std"]    = float(np.std(h_vals))
        stats["v_p90"]    = float(np.percentile(v_vals, 90))
        stats["v_std"]    = float(np.std(v_vals))
        stats["s_p90"]    = float(np.percentile(s_vals, 90))
    else:
        for key in ("h_median", "s_median", "v_median", "h_std", "v_p90", "v_std", "s_p90"):
            stats[key] = None

    stripe_score, ratio_center, ratio_ring = _compute_stripe_score(roi_bgr)
    stats["stripe_score"]       = stripe_score
    stats["white_ratio_center"] = ratio_center
    stats["white_ratio_ring"]   = ratio_ring

    return stats


def extract_ball_features(img_bgr, x, y, r):
    """
    Crop ROI around (x, y, r) and compute all features needed by the classifier.
    Returns a flat feature dict compatible with the greedy search pipeline.
    """
    h_img, w_img = img_bgr.shape[:2]
    rr = int(round(r * ROI_PAD))
    x1 = max(0, int(round(x - rr)))
    y1 = max(0, int(round(y - rr)))
    x2 = min(w_img, int(round(x + rr)))
    y2 = min(h_img, int(round(y + rr)))

    roi_bgr = img_bgr[y1:y2, x1:x2].copy()
    stats   = compute_patch_statistics(roi_bgr)

    wrc = stats.get("white_ratio_center") or 0.0
    wrr = stats.get("white_ratio_ring")   or 0.0

    return {
        "white_ratio":         stats.get("white_ratio"),
        "black_ratio":         stats.get("black_ratio"),
        "colour_ratio":        stats.get("colour_ratio"),
        "h_median":            stats.get("h_median"),
        "s_median":            stats.get("s_median"),
        "v_median":            stats.get("v_median"),
        "h_std":               stats.get("h_std"),
        "s_p90":               stats.get("s_p90"),
        "v_p90":               stats.get("v_p90"),
        "v_std":               stats.get("v_std"),
        "stripe_score":        stats.get("stripe_score"),
        "white_ratio_center":  wrc,
        "white_ratio_ring":    wrr,
        "white_center_minus_ring": wrc - wrr,
        "ball_type_hint":      None,  # unknown at inference time
    }


# =============================================================
# Pipeline helpers
# =============================================================

def safe_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        v = float(value)
        return float(default) if math.isnan(v) else v
    except Exception:
        return float(default)


def circular_hue_diff(h1, h2):
    diff = abs(float(h1) - float(h2))
    return min(diff, 180.0 - diff)


def final_label_from_parts(base_colour, ball_type):
    if base_colour == "white":
        return "cue"
    if base_colour == "black":
        return "black"
    if base_colour is None or ball_type is None:
        return "not_ball"
    return f"{base_colour}_{ball_type}"


# =============================================================
# Stage 1: validity scoring
# =============================================================

def predict_validity(row, p):
    wr  = safe_float(row.get("white_ratio"))
    br  = safe_float(row.get("black_ratio"))
    cr  = safe_float(row.get("colour_ratio"))
    hs  = safe_float(row.get("h_std"))
    vs  = safe_float(row.get("v_std"))
    sm  = safe_float(row.get("s_median"))
    wrc = safe_float(row.get("white_ratio_center"))
    wrr = safe_float(row.get("white_ratio_ring"))
    cmr = safe_float(row.get("white_center_minus_ring"), wrc - wrr)

    score = 0.0

    likely_cue = (
        wr  >= p["s1_cue_white_ratio_min"]
        and wrc >= p["s1_cue_white_center_min"]
        and br  <= p["s1_cue_black_ratio_max"]
    )
    if likely_cue:
        score += p["s1_cue_score_bonus"]

    likely_dark = (
        br >= p["s1_dark_black_ratio_min"]
        and cr >= p["s1_dark_colour_ratio_min"]
        and vs >= p["s1_dark_v_std_min"]
    )
    if likely_dark:
        score += p["s1_dark_score_bonus"]

    likely_uniform = (
        cr  >= p["s1_uniform_colour_ratio_min"]
        and br  <= p["s1_uniform_black_ratio_max"]
        and wrc <= p["s1_uniform_white_center_max"]
        and wr  <= p["s1_uniform_white_ratio_max"]
        and vs  >= p["s1_uniform_v_std_min"]
    )
    if likely_uniform:
        score += p["s1_uniform_score_bonus"]

    likely_artifact = (
    br >= p["s1_artifact_black_ratio_min"]
    and hs <= p["s1_artifact_h_std_max"]
    and wrc <= p["s1_artifact_white_center_max"]
)
    if likely_artifact:
        score += p["s1_artifact_score_penalty"]

    if wrc >= p["s1_pos_white_center_min"]:
        score += p["s1_pos_white_center_bonus"]
    if wr >= p["s1_pos_white_ratio_min"]:
        score += p["s1_pos_white_ratio_bonus"]
    if hs >= p["s1_pos_h_std_good_min"]:
        score += p["s1_pos_h_std_good_bonus"]
    if hs >= p["s1_pos_h_std_vgood_min"]:
        score += p["s1_pos_h_std_vgood_bonus"]
    if vs >= p["s1_pos_v_std_good_min"]:
        score += p["s1_pos_v_std_good_bonus"]
    if vs >= p["s1_pos_v_std_vgood_min"]:
        score += p["s1_pos_v_std_vgood_bonus"]
    if cr >= p["s1_pos_colour_ratio_min"]:
        score += p["s1_pos_colour_ratio_bonus"]

    if hs < p["s1_sus_h_std_low_thr"] and not likely_uniform and not likely_dark:
        score += p["s1_sus_h_std_low_penalty"]
    if hs < p["s1_sus_h_std_vlow_thr"] and not likely_uniform and not likely_dark:
        score += p["s1_sus_h_std_vlow_penalty"]
    if sm >= p["s1_sus_sat_thr"] and hs < p["s1_sus_sat_h_std_thr"]:
        score += p["s1_sus_sat_penalty"]
    if wrc < p["s1_sus_white_center_vlow_thr"] and not likely_uniform:
        score += p["s1_sus_white_center_vlow_pen"]
    if wr < p["s1_sus_white_ratio_vlow_thr"] and not likely_uniform:
        score += p["s1_sus_white_ratio_vlow_pen"]
    if cmr > p["s1_sus_center_minus_ring_thr"] and not likely_cue:
        score += p["s1_sus_center_minus_ring_pen"]
    if br > p["s1_sus_black_ratio_thr"] and not likely_dark:
        score += p["s1_sus_black_ratio_pen"]

    likely_black_ball = (
        br  >= p["s1_black_black_ratio_min"]
        and cr  >= p["s1_black_colour_ratio_min"]
        and vs  >= p["s1_black_v_std_min"]
        and wrc <= p["s1_black_white_center_max"]
    )
    if likely_black_ball:
        score += p["s1_black_score_bonus"]

    validity_class = "suspect" if score <= p["s1_suspect_score_thr"] else "valid_ball"
    return {"validity_class": validity_class, "validity_score": float(score)}


# =============================================================
# Stage 2: colour prediction
# =============================================================

def apply_colour_rules(row, p):
    """Hard-rule colour classification (black, yellow-stripe, white/cue)."""
    br  = safe_float(row.get("black_ratio"))
    wr  = safe_float(row.get("white_ratio"))
    cr  = safe_float(row.get("colour_ratio"))
    hm  = safe_float(row.get("h_median"))
    vm  = safe_float(row.get("v_median"))
    sm  = safe_float(row.get("s_median"))
    vp  = safe_float(row.get("v_p90"), vm)
    wrc = safe_float(row.get("white_ratio_center"))
    bt  = row.get("ball_type_hint")

    if (
        br >= p["s2_black_black_ratio_min"]
        and wr <= p["s2_black_white_ratio_max"]
        and vm <= p["s2_black_v_median_max"]
    ):
        return "black"

    y_lo = min(p["s2_yellow_white_ratio_lo"], p["s2_yellow_white_ratio_hi"])
    y_hi = max(p["s2_yellow_white_ratio_lo"], p["s2_yellow_white_ratio_hi"])

    if (
        bt == "stripe"
        and y_lo <= wr <= y_hi
        and br <= p["s2_yellow_black_ratio_max"]
        and cr >= p["s2_yellow_colour_ratio_min"]
        and hm <= p["s2_yellow_h_median_max"]
    ):
        return "yellow"

    if (
        wr  >= p["s2_white_white_ratio_min"]
        and vp  >= p["s2_white_v_p90_min"]
        and sm  <= p["s2_white_s_median_max"]
        and br  <= p["s2_white_black_ratio_max"]
        and wrc >= p["s2_white_center_min"]
        and wr  >= wrc * p["s2_white_center_ratio"]
    ):
        return "white"

    return None

def infer_ball_type_hint(features):
    ss = safe_float(features.get("stripe_score"))
    wr = safe_float(features.get("white_ratio"))
    wrc = safe_float(features.get("white_ratio_center"))
    wrr = safe_float(features.get("white_ratio_ring"))

    if wrr > wrc + 0.015 and wr >= 0.08:
        return "stripe"
    if wrc >= wrr:
        return "solid"
    return None


def score_colour_candidate(row, p, colour_name):
    hm = safe_float(row.get("h_median"))
    sm = safe_float(row.get("s_median"))
    vm = safe_float(row.get("v_median"))
    hs = safe_float(row.get("h_std"))
    cr = safe_float(row.get("colour_ratio"))

    score = 0.0

    # Soft gate penalties
    if cr < p["s2_chrom_colour_ratio_min"]:
        score -= 1.2 * (p["s2_chrom_colour_ratio_min"] - cr) / max(p["s2_chrom_colour_ratio_min"], 1e-9)
    if sm < p["s2_chrom_s_min"]:
        score -= 0.8 * (p["s2_chrom_s_min"] - sm) / max(p["s2_chrom_s_min"], 1e-9)
    if hs < p["s2_manual_h_std_min"]:
        score -= 0.5 * (p["s2_manual_h_std_min"] - hs) / max(p["s2_manual_h_std_min"] + 1e-9, 1.0)

    if colour_name == "red":
        d = min(circular_hue_diff(hm, p["s2_red_center1"]), circular_hue_diff(hm, p["s2_red_center2"]))
        score += 1.0 - d / max(p["s2_red_tol"], 1e-9)
        if d > p["s2_red_tol"]:
            score -= 1.0 + 0.08 * (d - p["s2_red_tol"])
        if sm < p["s2_red_s_min"]: score -= 0.4
        if vm < p["s2_red_v_min"]: score -= 0.3
        score += p["s2_red_score_bias"]
        return score

    if colour_name == "maroon":
        d = circular_hue_diff(hm, p["s2_maroon_center"])
        score += 1.0 - d / max(p["s2_maroon_tol"], 1e-9)
        if d > p["s2_maroon_tol"]:
            score -= 1.0 + 0.08 * (d - p["s2_maroon_tol"])
        if sm < p["s2_maroon_s_min"]: score -= 0.3
        if vm > p["s2_maroon_v_max"]: score -= 0.5
        score += p["s2_maroon_score_bias"]
        return score

    if colour_name == "orange":
        d = circular_hue_diff(hm, p["s2_orange_center"])
        score += 1.0 - d / max(p["s2_orange_tol"], 1e-9)
        if d > p["s2_orange_tol"]:
            score -= 1.0 + 0.08 * (d - p["s2_orange_tol"])
        if sm < p["s2_orange_s_min"]: score -= 0.3
        if vm < p["s2_orange_v_min"]: score -= 0.4
        score += p["s2_orange_score_bias"]
        return score

    if colour_name == "yellow":
        d = circular_hue_diff(hm, p["s2_yellow_center"])
        score += 1.0 - d / max(p["s2_yellow_tol"], 1e-9)
        if d > p["s2_yellow_tol"]:
            score -= 1.0 + 0.08 * (d - p["s2_yellow_tol"])
        if sm < p["s2_yellow_s_min"]: score -= 0.3
        if vm < p["s2_yellow_v_min"]: score -= 0.4
        score += p["s2_yellow_score_bias"]
        return score

    if colour_name == "green":
        d = circular_hue_diff(hm, p["s2_green_center"])
        score += 1.0 - d / max(p["s2_green_tol"], 1e-9)
        if d > p["s2_green_tol"]:
            score -= 1.0 + 0.08 * (d - p["s2_green_tol"])
        if sm < p["s2_green_s_min"]: score -= 0.3
        if vm < p["s2_green_v_min"]: score -= 0.3
        score += p["s2_green_score_bias"]
        return score

    if colour_name == "blue":
        d = circular_hue_diff(hm, p["s2_blue_center"])
        score += 1.0 - d / max(p["s2_blue_tol"], 1e-9)
        if d > p["s2_blue_tol"]:
            score -= 1.0 + 0.08 * (d - p["s2_blue_tol"])
        if sm < p["s2_blue_s_min"]: score -= 0.3
        if vm < p["s2_blue_v_min"]: score -= 0.3
        score += p["s2_blue_score_bias"]
        return score

    if colour_name == "purple":
        d = circular_hue_diff(hm, p["s2_purple_center"])
        score += 1.0 - d / max(p["s2_purple_tol"], 1e-9)
        if d > p["s2_purple_tol"]:
            score -= 1.0 + 0.08 * (d - p["s2_purple_tol"])
        if sm < p["s2_purple_s_min"]: score -= 0.3
        if vm < p["s2_purple_v_min"]: score -= 0.3
        score += p["s2_purple_score_bias"]
        return score

    return -999.0


def predict_base_colour(row, p):
    rule = apply_colour_rules(row, p)
    if rule is not None:
        return {"top1": rule, "ranked": [(rule, 1.0)], "confidence": 1.0}

    chromatic = ["red", "maroon", "orange", "yellow", "green", "blue", "purple"]
    scores = sorted(
        [(c, score_colour_candidate(row, p, c)) for c in chromatic],
        key=lambda x: x[1], reverse=True,
    )

    best_colour, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else best_score - 1.0

    adjusted = list(scores)
    if best_score - second_score < p["s2_manual_score_gap_min"]:
        adjusted[0] = (best_colour, best_score + p["s2_manual_unknown_penalty"])

    raw  = np.array([s for _, s in adjusted], dtype=float)
    raw -= np.max(raw)
    exps = np.exp(np.clip(raw, -20, 20))
    probs = exps / np.sum(exps)

    ranked     = [(c, float(pr)) for (c, _), pr in zip(adjusted, probs)]
    confidence = float(ranked[0][1])

    return {"top1": ranked[0][0], "ranked": ranked, "confidence": confidence}


# =============================================================
# Stage 3: ball type (solid vs stripe)
# =============================================================

def predict_ball_type(row, colour, p):
    if colour == "white":
        return {"top1": "cue",   "ranked": [("cue",   1.0)], "confidence": 1.0}
    if colour == "black":
        return {"top1": "black", "ranked": [("black", 1.0)], "confidence": 1.0}
    if colour is None:
        return {"top1": None,    "ranked": [],               "confidence": 0.0}

    ss  = safe_float(row.get("stripe_score"))
    wr  = safe_float(row.get("white_ratio"))
    wrc = safe_float(row.get("white_ratio_center"))
    wrr = safe_float(row.get("white_ratio_ring"))
    hs  = safe_float(row.get("h_std"))
    cr  = safe_float(row.get("colour_ratio"))

    if colour == "purple":
        score = (
            p["s3_purple_solid_score"]
            if hs < p["s3_purple_h_std_thr"] and cr > p["s3_purple_colour_ratio_thr"]
            else -p["s3_purple_stripe_coef"] * ss
        )
    else:
        score = -p["s3_stripe_score_coef"] * ss
        if ss < -0.03 and wrr > p["s3_white_ratio_ring_thr"]:
            score += p["s3_center_ring_diff_coef"] * (wrc - wrr)
        score += p["s3_white_ratio_coef"] * (wr - p["s3_white_ratio_offset"])

    prob_stripe = 1.0 / (1.0 + np.exp(-p["s3_sigmoid_scale"] * score))
    prob_solid  = 1.0 - prob_stripe

    ranked = sorted(
        [("stripe", prob_stripe), ("solid", prob_solid)],
        key=lambda x: x[1], reverse=True,
    )
    return {"top1": ranked[0][0], "ranked": ranked, "confidence": float(ranked[0][1])}


# =============================================================
# Stage 4: global consistency
# =============================================================

def build_ranked_final(base_pred, type_pred):
    options = []
    for colour_cls, colour_prob in base_pred["ranked"]:
        if colour_cls == "white":
            rtypes = [("cue",   1.0)]
        elif colour_cls == "black":
            rtypes = [("black", 1.0)]
        elif colour_cls == base_pred["top1"]:
            rtypes = type_pred["ranked"]
        else:
            rtypes = [("solid", 0.5), ("stripe", 0.5)]

        for type_cls, type_prob in rtypes:
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


def combined_strength(row, p):
    penalty = p["s4_suspect_penalty"] if row["validity_class"] == "suspect" else 0.0
    return (
        p["s4_w_final_conf"]    * float(row["pred_final_conf"])
        + p["s4_w_base_conf"]   * float(row["pred_base_colour_conf"])
        + p["s4_w_type_conf"]   * float(row["pred_ball_type_conf"])
        + p["s4_w_validity_score"] * float(row["validity_score"])
        + penalty
    )


def get_label_prob(row, wanted):
    for opt in row["ranked_final_predictions"]:
        if opt["final_label"] == wanted:
            return float(opt["joint_prob"])
    return 0.0


def choose_required(rows, req_label):
    candidates = []
    for idx, row in enumerate(rows):
        prob = get_label_prob(row, req_label)
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


def try_reassign(row, assigned_labels, forbidden=None):
    counts = Counter(assigned_labels)
    for option in row["ranked_final_predictions"]:
        cand = option["final_label"]
        prob = float(option["joint_prob"])
        if cand == forbidden or cand == "not_ball":
            continue
        if counts[cand] < 1:
            return cand, prob, "reassigned"
    return row["final_global_label"], row["final_global_conf"], "kept_duplicate"


def resolve_global(pred_rows, p):
    rows = [dict(r) for r in pred_rows]

    for row in rows:
        row["strength"]           = combined_strength(row, p)
        row["final_global_label"] = row["pred_final_label"]
        row["final_global_conf"]  = float(row["pred_final_conf"])
        row["global_resolution"]  = "kept_top1"

    for _ in range(2):
        for req in REQUIRED_UNIQUE_LABELS:
            winner = choose_required(rows, req)
            if winner is None:
                continue
            widx, wprob = winner
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
                if len(items) <= 1:
                    continue
                items        = sorted(items, key=lambda x: x[1]["strength"], reverse=True)
                overflow_idx = {idx for idx, _ in items[1:]}
                kept = [
                    rows[i]["final_global_label"]
                    for i in range(len(rows))
                    if rows[i]["final_global_label"] != "not_ball" and i not in overflow_idx
                ]
                for idx, row in items[1:]:
                    nl, np_, mode = try_reassign(row, kept, forbidden=label)
                    if nl != row["final_global_label"]:
                        rows[idx]["final_global_label"] = nl
                        rows[idx]["final_global_conf"]  = np_
                        rows[idx]["global_resolution"]  = mode
                        changed = True

    for row in rows:
        if row["final_global_label"] in ("cue", "black"):
            continue
        if row["validity_class"] == "suspect" and row["strength"] < p["s4_weak_suspect_strength_thr"]:
            row["final_global_label"] = "not_ball"
            row["final_global_conf"]  = 0.0
            row["global_resolution"]  = "weak_suspect_demoted"
        elif row["validity_score"] <= p["s4_strong_invalid_score_thr"]:
            row["final_global_label"] = "not_ball"
            row["final_global_conf"]  = 0.0
            row["global_resolution"]  = "final_strong_invalid"

    return rows



def predict_single(features, p):
    """Run stages 1-3 on a single feature dict."""
    validity   = predict_validity(features, p)
    base_pred  = predict_base_colour(features, p)
    type_pred  = predict_ball_type(features, base_pred["top1"], p)
    ranked     = build_ranked_final(base_pred, type_pred) if base_pred["ranked"] else []

    out = dict(features)
    out.update(validity)
    out["pred_base_colour"]      = base_pred["top1"]
    out["pred_base_colour_conf"] = float(base_pred["confidence"])
    out["pred_ball_type"]        = type_pred["top1"]
    out["pred_ball_type_conf"]   = float(type_pred["confidence"])
    out["pred_final_label"]      = ranked[0]["final_label"] if ranked else "not_ball"
    out["pred_final_conf"]       = ranked[0]["joint_prob"]  if ranked else 0.0
    out["ranked_final_predictions"] = ranked
    return out


# =============================================================
# Per-image processing
# =============================================================

def process_image(image_path, params):
    """
    Full pipeline for a single image.
    Returns a result dict with image_path, num_balls, and balls list.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Warning: could not read {image_path}", file=sys.stderr)
        return {"image_path": image_path, "num_balls": 0, "balls": []}

    h_img, w_img = img_bgr.shape[:2]

    # Stage 0: detect table + balls
    table_mask, corners = segment_table(img_bgr)

    # Save top-view warp before early return so all images get one
    top_view_dir = Path("output/top_view")
    top_view_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(top_view_dir / Path(image_path).name), get_top_view(img_bgr, corners))

    detections = detect_balls(img_bgr, table_mask)
    if not detections:
        return {"image_path": image_path, "num_balls": 0, "balls": []}

    # Extract features per detection
    feature_rows = []
    for det_idx, (x, y, r) in enumerate(detections):
        feats = extract_ball_features(img_bgr, x, y, r)
        feats["ball_type_hint"] = infer_ball_type_hint(feats)
        feats["_x"] = x
        feats["_y"] = y
        feats["_r"] = r
        feats["_det_idx"] = det_idx
        feature_rows.append(feats)

    # Stages 1-3 per ball
    pred_rows = [predict_single(f, params) for f in feature_rows]

    # Stage 4: global consistency across the image
    resolved = resolve_global(pred_rows, params)

    # Build output ball list (skip not_ball predictions)
    balls = []
    for row in resolved:
        label = row["final_global_label"]
        if label == "not_ball" or label not in BALL_LABEL_TO_NUMBER:
            continue

        x = row["_x"]
        y = row["_y"]
        r = row["_r"]

        # Normalized bounding box
        xmin = max(0.0, (x - r) / w_img)
        xmax = min(1.0, (x + r) / w_img)
        ymin = max(0.0, (y - r) / h_img)
        ymax = min(1.0, (y + r) / h_img)

        balls.append({
            "number": BALL_LABEL_TO_NUMBER[label],
            "xmin":   xmin,
            "xmax":   xmax,
            "ymin":   ymin,
            "ymax":   ymax,
        })

    return {
        "image_path": image_path,
        "num_balls":  len(balls),
        "balls":      balls,
    }

def evaluate_output(results, csv_path):
    import csv as csv_module
    from collections import Counter

    CSV_LABEL_MAP = {
        "0_white_cue": "cue", "1_yellow_solid": "yellow_solid", "2_blue_solid": "blue_solid",
        "3_red_solid": "red_solid", "4_purple_solid": "purple_solid", "5_orange_solid": "orange_solid",
        "6_green_solid": "green_solid", "7_maroon_solid": "maroon_solid", "8_black_8ball": "black",
        "9_yellow_stripe": "yellow_stripe", "10_blue_stripe": "blue_stripe", "11_red_stripe": "red_stripe",
        "12_purple_stripe": "purple_stripe", "13_orange_stripe": "orange_stripe",
        "14_green_stripe": "green_stripe", "15_maroon_stripe": "maroon_stripe",
    }
    NUMBER_TO_LABEL = {v: k for k, v in BALL_LABEL_TO_NUMBER.items()}

    gt = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            gt[fname] = {}
            for col, label in CSV_LABEL_MAP.items():
                val = row.get(col, "0").strip()
                gt[fname][label] = int(val) if val else 0

    total_tp, total_fp, total_fn = 0, 0, 0
    for res in results:
        fname = Path(res["image_path"]).name
        if fname not in gt:
            continue
        pred_counts = Counter(NUMBER_TO_LABEL[b["number"]] for b in res["balls"])
        for label, gt_count in gt[fname].items():
            pred_count = pred_counts.get(label, 0)
            total_tp += min(gt_count, pred_count)
            total_fp += max(0, pred_count - gt_count)
            total_fn += max(0, gt_count - pred_count)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"\nEvaluation vs ground truth:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")

    print(f"\n  Per-class breakdown:")
    print(f"  {'Label':<20} {'GT':>5} {'Pred':>5} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*45}")

    all_labels = sorted(CSV_LABEL_MAP.values())
    for label in all_labels:
        gt_total   = sum(gt[f].get(label, 0) for f in gt)
        pred_total = sum(Counter(NUMBER_TO_LABEL[b["number"]] for b in res["balls"]).get(label, 0)
                         for res in results if Path(res["image_path"]).name in gt)
        tp = sum(min(gt[f].get(label, 0),
                     Counter(NUMBER_TO_LABEL[b["number"]] for b in res["balls"]).get(label, 0))
                 for res in results for f in [Path(res["image_path"]).name] if f in gt)
        fp = max(0, pred_total - tp)
        fn = max(0, gt_total   - tp)
        print(f"  {label:<20} {gt_total:>5} {pred_total:>5} {tp:>5} {fp:>5} {fn:>5}")

    total_slots = 0
    total_correct_slots = 0

    for res in results:
        fname = Path(res["image_path"]).name
        if fname not in gt:
            continue

        pred_counts = Counter(NUMBER_TO_LABEL[b["number"]] for b in res["balls"])

        for label in CSV_LABEL_MAP.values():
            gt_count = gt[fname].get(label, 0)
            pred_count = pred_counts.get(label, 0)
            total_correct_slots += min(gt_count, pred_count)
            total_slots += max(gt_count, pred_count)
    overall_accuracy = total_correct_slots / max(total_slots, 1)
    print(f"  Overall accuracy: {overall_accuracy:.4f}")

# =============================================================
# Entry point
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="Billiard ball detection and classification")
    parser.add_argument("--input",  default="input/input.json",    help="Path to input JSON")
    parser.add_argument("--output", default="output/output.json",  help="Path to output JSON")
    args = parser.parse_args()

    params = OPTIMIZED_PARAMS

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    image_paths = input_data.get("image_path", [])
    base_dir = input_path.parent
    image_paths = [str(base_dir / p) for p in image_paths]

    if not image_paths:
        print("No image paths found in input JSON.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(image_paths)} image(s)...")

    results = []
    for i, image_path in enumerate(image_paths):
        print(f"  [{i + 1}/{len(image_paths)}] {image_path}")
        result = process_image(image_path, params)
        results.append(result)
        print(f"    -> {result['num_balls']} ball(s) detected")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")

    gt_csv = Path(args.input).parent / "ground_truth_counts.csv"
    if gt_csv.exists():
        evaluate_output(results, gt_csv)
    else:
        print(f"Ground truth CSV not found at {gt_csv}, skipping evaluation.")

if __name__ == "__main__":
    main()