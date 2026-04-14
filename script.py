# %% [markdown]
# # Assignment: Detection of Pool Balls & POV Warp for Table Top View
# 
# **Work assembled by:**
# - Alejandro Gonçalves (202205564)
# - Francisca Mihalache (202206022)
# - João Sousa ()
# - Rafael Pacheco ()
# 
# **Year:** 2025/2026
# 

# %% [markdown]
# ## Table of Contents
# 
# 1. [Introduction](#1-introduction)
# 2. [Libraries & Frameworks](#2-libraries--frameworks)
#     - 2.1. [Global Variables](#21-global-variables)
# 3. [Loading Images](#3-loading-images)
# 4. [Creating Cloth Mask](#4-creating-cloth-mask)
# 5. [Circle Detection](#5-circle-detection)
#     - 5.1. [Test ball detection on the sample image](#51-test-ball-detection-on-the-sample-image)
#     - 5.2. [Debug Maps and Image Enhancement](#52-debug-maps-and-image-enhancement)
# 6. [Ball Identification](#6-ball-identification)
#     - 6.1. [Test ball identification on the sample image](#61-test-ball-identification-on-the-sample-image)
# 7. [Ground Truth Generation](#7-ground-truth-generation)
# 8. [Bounding Box Generation](#8-bounding-box-generation)
# 9. [Evaluation Metrics](#9-evaluation-metrics)
# 

# %% [markdown]
# ### 1. Introduction
# [[go back to the top]](#table-of-contents)
# 
# This notebook presents a computer vision pipeline for detecting and identifying pool (billiard) balls on a table, producing a top-down (bird's eye) perspective warp.
# 
# The main challenge consists of reliably locating the billiard table's playing surface, detecting all balls present on it, and correctly identifying each ball by its number, colour, and type (solid or striped). The approach follows three main stages:
# 
# 1. **Cloth Mask** — We use colour segmentation in HSV space to isolate the green or blue playing surface, filtering out the surrounding environment.
# 2. **Ball Detection** — Using the Hough Circle Transform on the masked region, we detect circular objects that correspond to the billiard balls.
# 3. **Ball Identification** — Each detected circle is analysed using colour histograms and white-pixel ratios to assign a ball number.
# 
# All image processing is performed using **OpenCV** and **NumPy**, with **Matplotlib** used for visualisation. No additional libraries are required.
# 

# %% [markdown]
# ### 2. Libraries & Frameworks
# 
# [[go back to the top]](#table-of-contents)
# 

# %% [markdown]
# Utils file includes every necessary function and framework for this entire notebook.

# %%
from utils import *
%load_ext autoreload
%autoreload 2

# %% [markdown]
# #### 2.1. Global Variables
# [[go back to the topic]](#2-libraries--frameworks)

# %%
DATA_DIR = "data"

# %% [markdown]
# ### 3. Loading Images
# 
# [[go back to the top]](#table-of-contents)
# 

# %%
image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
print(f"Detected {len(image_paths)} imgs")

# %% [markdown]
# ### 4. Creating Cloth Mask
# [[go back to the top]](#table-of-contents)
# 
# In order to optimize object detection, we decided to first mask the image to only entail the region of interest, which is the table's cloth. It's usually bright blue, but the whole dataset also has some green cloth tables, and we also wanted to prepare for possible changes. To make this a robust method, we took advantage of HSV color space to detect regions with color ranges in either blue or green, in combination with edge/corner detection filters.
# 

# %%
def segment_table(img):
    # Enable hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect blue cloth (Hue ~90-130) OR green cloth (Hue ~35-85)
    blue_mask = cv2.inRange(hsv, np.array([90, 80, 50]), np.array([130, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 80, 50]), np.array([85, 255, 255]))
    mask = cv2.bitwise_or(blue_mask, green_mask)

    # Find the biggest contour -> most likely to be table
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_contour = max(contours, key=cv2.contourArea)

    # Approximates to 4 corners
    epsilon = 0.02 * cv2.arcLength(table_contour, True)
    approx = cv2.approxPolyDP(table_contour, epsilon, True)

    # If we don't get exactly 4 points, assumes shape 
    # closest to a 4-sided polygon
    if len(approx) == 4: corners = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(table_contour)
        corners = cv2.boxPoints(rect).astype(np.float32)

    # Order corners & create mask
    corners = order_corners(corners)
    clean_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(clean_mask, [corners.astype(np.int32)], 255)

    return clean_mask, corners

# %% [markdown]
# Let's visualize the mask and detected corners to make sure it works.

# %%
sample_img = cv2.imread(random.choice(image_paths))
mask, corners = segment_table(sample_img)

# Draws corners
vis = sample_img.copy()
if corners is not None:
    for i, c in enumerate(corners):
        cv2.circle(vis, tuple(c.astype(int)), 10, (0, 255, 0), -1)
        cv2.putText(vis, str(i), tuple(c.astype(int) + [10, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show corners, mask and cropped image
masked_img = cv2.bitwise_and(sample_img, sample_img, mask=mask)
show([vis, mask, masked_img], ["Corners", "Table mask", "Masked image"])

# %% [markdown]
# ### 5. Circle Detection
# [[go back to the top]](#table-of-contents)
# 
# Spheres have a very convenient property: no matter what angle they are photographed in, their 2D projection is always a circle.
# 
# We can take this into account and use a circle detection filter inside the mask to identify them.
# 
# First, we decided to use the simplest option, `cv2.HoughCircles`. It works by looking at edges, which are essentially sharp changes in brightness not necessarily by color. For this reason, we converted to grayscale.
# 
# For color optimization we boosted saturation and contrast and reduced shadows. The problem with this is that grayscale generalizes too much, and balls often blend with the background. In order to help with this, we also considered color distance between the balls and the cloth.
# 

# %% [markdown]
# | HoughCircles parameters     |  Meaning  | Value     |
# | ----------------------------|-----------|-------------------------------------------|
# | minDist              | threshold of overlap between circles  |          trial and error           |
# | param1              |  finds outlines high value will miss those that blend into the background  | trial and error (~40) |
# | param2              |  degree of perfection of the circle|       trial and error (~30)           |
# | minRadius/maxRadius   |  defines possible range of ball size    |         based on visual cues           |

# %%
def detect_balls(img, mask):
    masked = cv2.bitwise_and(img, img, mask=mask)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    # Boost saturation and brightness
    s_chan = cv2.convertScaleAbs(s_chan, alpha=1.35, beta=10)
    v_chan = cv2.convertScaleAbs(v_chan, alpha=1.15, beta=8)

    hsv_boosted = cv2.merge([h_chan, s_chan, v_chan])
    masked_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    cloth_mean = np.array(cv2.mean(masked_boosted, mask=mask)[:3])
    diff = np.sqrt(np.sum((masked_boosted.astype(np.float32) - cloth_mean) ** 2, axis=2))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    gray_raw = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_raw)
    s_enhanced = clahe.apply(s_chan)
    

    gray = cv2.addWeighted(diff, 0.9, s_enhanced, 0.1, 0)
    gray = cv2.addWeighted(gray, 0.9, gray_raw, 0.1, 0)
    gray = cv2.GaussianBlur(gray, (3, 3), 2)

    gray_enhanced = cv2.addWeighted(diff, 0.8, s_enhanced, 0.2, 0)
    gray_enhanced = cv2.addWeighted(gray_enhanced, 0.85, gray_clahe, 0.15, 0)
    gray_enhanced = cv2.GaussianBlur(gray_enhanced, (3, 3), 2)

    h, w = img.shape[:2]
    min_r = max(8, int(h * 0.01))
    max_r = int(h * 0.025)

    # Edge map for circle validation
    edge = cv2.Canny(gray, 40, 140)
    # Pass 1: Conservative on normal image (best for clear, well-lit balls)
    circles1 = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 1.7,
        param1=38, param2=29.3, minRadius=min_r, maxRadius=max_r
    )

    # Pass 2: Permissive on normal image
    circles2 = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 1.7,
        param1=38, param2=26.5, minRadius=min_r, maxRadius=max_r
    )

    # Pass 3: Permissive on CLAHE enhanced image (to catch shadow balls)
    circles3 = cv2.HoughCircles(
        gray_enhanced, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 1.7,
        param1=38, param2=26.5, minRadius=min_r, maxRadius=max_r
    )

    candidates = []
    if circles1 is not None:
        candidates.extend([(x, y, r, "hough") for x, y, r in np.round(circles1[0]).astype(int).tolist()])
    if circles2 is not None:
        candidates.extend([(x, y, r, "hough") for x, y, r in np.round(circles2[0]).astype(int).tolist()])
    if circles3 is not None:
        candidates.extend([(x, y, r, "hough") for x, y, r in np.round(circles3[0]).astype(int).tolist()])

        
    # detect purple blobs directly in HSV
    lower_purple = np.array([110, 40, 30], dtype=np.uint8)
    upper_purple = np.array([175, 255, 255], dtype=np.uint8)

    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    purple_mask = cv2.bitwise_and(purple_mask, mask)

    kernel = np.ones((3, 3), np.uint8)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
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

    balls = []

    for (x, y, r, source) in candidates:
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            continue

        circle_mask = np.zeros_like(mask)
        if source == "purple":
            inner_mask = np.zeros_like(mask)
            cv2.circle(inner_mask, (x, y), max(2, int(r * 0.65)), 255, -1)

            purple_pixels = cv2.countNonZero(cv2.bitwise_and(purple_mask, inner_mask))
            total_pixels = cv2.countNonZero(inner_mask)

            purple_ratio = purple_pixels / max(total_pixels, 1)

            if purple_ratio < 0.45:
                continue
        
        cv2.circle(circle_mask, (x, y), r, 255, -1)

        mean_val = cv2.mean(gray, mask=circle_mask)[0]
        if mean_val < 25:
            continue

        overlap = cv2.countNonZero(cv2.bitwise_and(mask, circle_mask))
        area = cv2.countNonZero(circle_mask)
        if area == 0 or overlap / area <= 0.35:
            continue

        ring_mask = np.zeros_like(mask)
        cv2.circle(ring_mask, (x, y), r, 255, 2)

        edge_overlap = cv2.countNonZero(cv2.bitwise_and(edge, ring_mask))
        circumference = max(2 * np.pi * r, 1)

        if source != "purple" and edge_overlap / circumference < 0.20:
            continue

        balls.append((x, y, r))

    # Sort by radius descending so we process larger, more reliable circles first
    balls.sort(key=lambda b: b[2], reverse=True)

    # Remove duplicates
    deduped = []
    for x, y, r in balls:
        duplicate = False
        for x2, y2, r2 in deduped:
            center_dist = np.hypot(x - x2, y - y2)
            # If the smaller circle's center is inside the larger circle, drop it
            if center_dist < 1.0* max(r, r2):
                duplicate = True
                break
        if not duplicate:
            deduped.append((x, y, r))
    
    if len(deduped) >= 4:
        corrected_balls = []

        for i, (x, y, r) in enumerate(deduped):
            neighbors = []

            # Calculate distance to all other detected balls
            for j, (x2, y2, r2) in enumerate(deduped):
                if i == j:
                    continue
                dist = np.hypot(x - x2, y - y2)
                neighbors.append((dist, r2))

            # Sort by distance to find the closest ones
            neighbors.sort(key=lambda t: t[0])

            # Pick the closest 3 neighbors to compute local scale
            k = min(3, len(neighbors))
            local_radii = [r2 for (_, r2) in neighbors[:k]]

            if len(local_radii) > 0:
                local_median_r = np.median(local_radii)

                # If the radius is an outlier compared to its immediate neighbors,
                # correct it to the local median instead of deleting the ball.
                if r < 0.7 * local_median_r or r > 1.2 * local_median_r:
                    r = int(local_median_r)

            corrected_balls.append((x, y, r))

        deduped = corrected_balls
    return deduped


# %% [markdown]
# #### 5.1. Test ball detection on the sample image
# [[go back to the topic]](#5-circle-detection)
# 
# Draw circles and bounding boxes around each detected ball.

# %%
ls = []

for path in image_paths: ls.append(path)
def build_debug_maps(img, mask):
    masked = cv2.bitwise_and(img, img, mask=mask)

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    s_boost = cv2.convertScaleAbs(s_chan, alpha=1.35, beta=10)
    v_boost = cv2.convertScaleAbs(v_chan, alpha=1.15, beta=8)

    hsv_boosted = cv2.merge([h_chan, s_boost, v_boost])
    masked_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    cloth_mean = np.array(cv2.mean(masked_boosted, mask=mask)[:3])
    diff = np.sqrt(np.sum((masked_boosted.astype(np.float32) - cloth_mean) ** 2, axis=2))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    gray_raw = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2GRAY)
    hsv2 = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2HSV)
    _, s_chan2, _ = cv2.split(hsv2)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_raw)
    s_enhanced = clahe.apply(s_chan2)

    gray = cv2.addWeighted(diff, 0.9, s_enhanced, 0.1, 0)
    gray = cv2.addWeighted(gray, 0.9, gray_raw, 0.1, 0)
    gray = cv2.GaussianBlur(gray, (3, 3), 2)

    gray_enhanced = cv2.addWeighted(diff, 0.8, s_enhanced, 0.2, 0)
    gray_enhanced = cv2.addWeighted(gray_enhanced, 0.85, gray_clahe, 0.15, 0)
    gray_enhanced = cv2.GaussianBlur(gray_enhanced, (3, 3), 2)

    lower_purple = np.array([120, 70, 40], dtype=np.uint8)
    upper_purple = np.array([165, 255, 255], dtype=np.uint8)
    purple_mask = cv2.inRange(hsv2, lower_purple, upper_purple)
    purple_mask = cv2.bitwise_and(purple_mask, mask)

    return masked, masked_boosted, diff, gray, gray_enhanced, purple_mask


for path in image_paths[:1]:
    img = cv2.imread(path)
    mask, corners = segment_table(img)
    balls = detect_balls(img, mask)

    vis = img.copy()
    for (x, y, r) in balls:
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)

    name = os.path.basename(path)
    show(
        [vis],
        [f"{name} — {len(balls)} balls"],
        figsize=(8, 6)
    )

# %% [markdown]
# #### 5.2. Debug Maps and Image Enhancement
# [[go back to the topic]](#5-circle-detection)

# %%
def build_debug_maps(img, mask):
    masked = cv2.bitwise_and(img, img, mask=mask)

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    # Boost saturation and brightness
    s_boost = cv2.convertScaleAbs(s_chan, alpha=1.35, beta=10)
    v_boost = cv2.convertScaleAbs(v_chan, alpha=1.15, beta=8)

    hsv_boosted = cv2.merge([h_chan, s_boost, v_boost])
    masked_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    cloth_mean = np.array(cv2.mean(masked_boosted, mask=mask)[:3])
    diff = np.sqrt(np.sum((masked_boosted.astype(np.float32) - cloth_mean) ** 2, axis=2))
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    gray_raw = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2GRAY)
    hsv2 = cv2.cvtColor(masked_boosted, cv2.COLOR_BGR2HSV)
    _, s_chan2, _ = cv2.split(hsv2)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_raw)
    s_enhanced = clahe.apply(s_chan2)

    gray = cv2.addWeighted(diff, 0.9, s_enhanced, 0.1, 0)
    gray = cv2.addWeighted(gray, 0.9, gray_raw, 0.1, 0)
    gray = cv2.GaussianBlur(gray, (3, 3), 2)

    gray_enhanced = cv2.addWeighted(diff, 0.8, s_enhanced, 0.2, 0)
    gray_enhanced = cv2.addWeighted(gray_enhanced, 0.85, gray_clahe, 0.15, 0)
    gray_enhanced = cv2.GaussianBlur(gray_enhanced, (3, 3), 2)

    lower_purple = np.array([110, 40, 30], dtype=np.uint8)
    upper_purple = np.array([175, 255, 255], dtype=np.uint8)

    purple_mask = cv2.inRange(hsv2, lower_purple, upper_purple)
    purple_mask = cv2.bitwise_and(purple_mask, mask)

    kernel = np.ones((3, 3), np.uint8)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)

    return masked_boosted, diff, gray_raw, gray_clahe, gray, gray_enhanced, purple_mask


for path in image_paths[:50]:
    img = cv2.imread(path)
    mask, corners = segment_table(img)
    balls = detect_balls(img, mask)

    vis = img.copy()
    for (x, y, r) in balls:
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)

    (
        masked_boosted,
        diff,
        gray_raw,
        gray_clahe,
        gray,
        gray_enhanced,
        purple_mask
    ) = build_debug_maps(img, mask)

    name = os.path.basename(path)

    show(
        [
          
            diff,
           
    
            
            gray_enhanced,
           
            vis
        ],
        [
           
            "diff",
           
    
            
            "gray enhanced",
            
            f"{name} - {len(balls)} balls"
        ],
        figsize=(22, 10)
    )

# %% [markdown]
# #### 5.3. Bectangle Bounding Boxes
# [[go back to the topic]](#5-circle-detection)

# %%
def circles_to_bboxes(circles, img_shape):
    """
    Convert detected circles (x, y, r) into rectangular bounding boxes
    [xmin, ymin, xmax, ymax], clipped to the image boundaries.
    """
    h, w = img_shape[:2]
    bboxes = []

    for (x, y, r) in circles:
        xmin = max(0, x - r)
        ymin = max(0, y - r)
        xmax = min(w, x + r)
        ymax = min(h, y + r)

        bboxes.append([xmin, ymin, xmax, ymax])

    return bboxes

sample_indices = [0, 10, 25, 40]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, ax in zip(sample_indices, axes):
    if idx >= len(image_paths):
        continue

    img = cv2.imread(image_paths[idx])
    mask, corners = segment_table(img)

    # Detect balls as circles
    balls = detect_balls(img, mask)

    # Convert circles to rectangular bounding boxes
    bboxes = circles_to_bboxes(balls, img.shape)

    vis = img.copy()

    # Draw rectangular bounding boxes
    for (xmin, ymin, xmax, ymax) in bboxes:
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{os.path.basename(image_paths[idx])} — {len(bboxes)} boxes")
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interface

# %%
import os
import json
import cv2
import numpy as np
import ipywidgets as widgets

from pathlib import Path
from IPython.display import display
from collections import defaultdict


# ============================================================
# Configuration
# ============================================================

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

LABELS = [
    "cue",
    "black",
    "yellow_solid",
    "yellow_stripe",
    "blue_solid",
    "blue_stripe",
    "red_solid",
    "red_stripe",
    "purple_solid",
    "purple_stripe",
    "orange_solid",
    "orange_stripe",
    "green_solid",
    "green_stripe",
    "maroon_solid",
    "maroon_stripe",
    "not_ball",
    "skip",
    "stop_and_save",
]

DISPLAY_SCALE = 4
INNER_RADIUS_FACTOR = 0.7
ROI_PAD = 2.0
AUTOSAVE_EVERY = 20

WHITE_LOW = np.array([0, 0, 160], dtype=np.uint8)
WHITE_HIGH = np.array([179, 70, 255], dtype=np.uint8)

BLACK_LOW = np.array([0, 0, 0], dtype=np.uint8)
BLACK_HIGH = np.array([179, 255, 55], dtype=np.uint8)


# ============================================================
# Optional white balance
# ============================================================

def gray_world_white_balance(img_bgr):
    """
    Simple gray-world white balance.
    """
    img = img_bgr.astype(np.float32)

    b_mean = img[:, :, 0].mean()
    g_mean = img[:, :, 1].mean()
    r_mean = img[:, :, 2].mean()

    gray_mean = (b_mean + g_mean + r_mean) / 3.0

    img[:, :, 0] *= gray_mean / max(b_mean, 1e-6)
    img[:, :, 1] *= gray_mean / max(g_mean, 1e-6)
    img[:, :, 2] *= gray_mean / max(r_mean, 1e-6)

    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# ROI extraction and feature computation
# ============================================================

def crop_ball_roi(img_bgr, x, y, r, pad=ROI_PAD):
    """
    Crop a square ROI around the detected ball.
    """
    h, w = img_bgr.shape[:2]

    rr = int(round(r * pad))
    x1 = max(0, int(round(x - rr)))
    y1 = max(0, int(round(y - rr)))
    x2 = min(w, int(round(x + rr)))
    y2 = min(h, int(round(y + rr)))

    roi = img_bgr[y1:y2, x1:x2].copy()
    return roi, x1, y1, x2, y2


def build_inner_circle_mask(roi_shape, radius_factor=INNER_RADIUS_FACTOR):
    """
    Inner circular mask to avoid cloth contamination.
    """
    h, w = roi_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx = w // 2
    cy = h // 2
    radius = int(min(cx, cy) * radius_factor)

    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask

def compute_relative_features(row, cloth_hsv_per_image):
    """
    cloth_hsv_per_image: dict {filename: (h_cloth, s_cloth, v_cloth)}
    """
    cloth_h, cloth_s, cloth_v = cloth_hsv_per_image.get(row["filename"], (0, 0, 0))

    h = row["h_median"] or 0.0
    s = row["s_median"] or 0.0
    v = row["v_median"] or 0.0

    # Delta de hue circular
    delta_h = float(h) - float(cloth_h)
    if delta_h > 90:  delta_h -= 180
    if delta_h < -90: delta_h += 180

    delta_v = float(v) - float(cloth_v)
    ratio_s  = float(s) / max(float(cloth_s), 1.0)

    return delta_h, delta_v, ratio_s

def compute_patch_statistics(roi_bgr):
    """
    Compute robust colour statistics from the ROI.
    HSV statistics are computed only on non-white and non-black pixels.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    circle_mask = build_inner_circle_mask(roi_bgr.shape)

    white_mask = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)
    white_mask = cv2.bitwise_and(white_mask, circle_mask)

    black_mask = cv2.inRange(hsv, BLACK_LOW, BLACK_HIGH)
    black_mask = cv2.bitwise_and(black_mask, circle_mask)

    colour_mask = cv2.bitwise_and(
        circle_mask,
        cv2.bitwise_not(cv2.bitwise_or(white_mask, black_mask))
    )

    total_pixels = cv2.countNonZero(circle_mask)
    white_pixels = cv2.countNonZero(white_mask)
    black_pixels = cv2.countNonZero(black_mask)
    colour_pixels = cv2.countNonZero(colour_mask)

    white_ratio = white_pixels / max(total_pixels, 1)
    black_ratio = black_pixels / max(total_pixels, 1)
    colour_ratio = colour_pixels / max(total_pixels, 1)

    stats = {
        "white_ratio": float(white_ratio),
        "black_ratio": float(black_ratio),
        "colour_ratio": float(colour_ratio),
        "total_pixels": int(total_pixels),
        "colour_pixels": int(colour_pixels),
    }

    if colour_pixels > 0:
        h_vals = hsv[:, :, 0][colour_mask > 0]
        s_vals = hsv[:, :, 1][colour_mask > 0]
        v_vals = hsv[:, :, 2][colour_mask > 0]

        stats["h_median"] = float(np.median(h_vals))
        stats["s_median"] = float(np.median(s_vals))
        stats["v_median"] = float(np.median(v_vals))

        stats["h_mean"] = float(np.mean(h_vals))
        stats["s_mean"] = float(np.mean(s_vals))
        stats["v_mean"] = float(np.mean(v_vals))

    

        stats["h_values_preview"] = [int(x) for x in np.percentile(h_vals, [10, 50, 90])]
        stats["s_values_preview"] = [int(x) for x in np.percentile(s_vals, [10, 50, 90])]
        stats["v_values_preview"] = [int(x) for x in np.percentile(v_vals, [10, 50, 90])]

        stats["h_p10"]  = float(np.percentile(h_vals, 10))
        stats["h_p90"]  = float(np.percentile(h_vals, 90))
        stats["h_std"]  = float(np.std(h_vals))

        stats["s_p10"]  = float(np.percentile(s_vals, 10))
        stats["s_p90"]  = float(np.percentile(s_vals, 90))

        stats["v_p10"]  = float(np.percentile(v_vals, 10))
        stats["v_p90"]  = float(np.percentile(v_vals, 90))
        stats["v_std"]  = float(np.std(v_vals))
    # Replace the end of compute_patch_statistics(roi_bgr) starting from the 'else' block
    else:
        stats["h_median"] = None
        stats["s_median"] = None
        stats["v_median"] = None
        stats["h_mean"] = None
        stats["s_mean"] = None
        stats["v_mean"] = None
        stats["h_values_preview"] = []
        stats["s_values_preview"] = []
        stats["v_values_preview"] = []

    # Calculate spatial white distribution to differentiate solids from stripes
    stripe_score, ratio_center, ratio_ring = compute_stripe_score(roi_bgr)
    
    stats["stripe_score"] = float(stripe_score)
    stats["white_ratio_center"] = float(ratio_center)
    stats["white_ratio_ring"] = float(ratio_ring)

    return stats

    
    
def compute_stripe_score(roi_bgr, r_factor_inner=0.45, r_factor_outer=0.75):
    """
    Retorna ratio de pixels brancos no anel vs centro.
    Stripe: anel tem muito mais branco que centro.
    Solid:  distribuição mais uniforme.
    """
    h, w = roi_bgr.shape[:2]
    cx, cy = w // 2, h // 2

    r_inner = int(min(cx, cy) * r_factor_inner)
    r_outer = int(min(cx, cy) * r_factor_outer)

    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_inner, (cx, cy), r_inner, 255, -1)

    mask_outer = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_outer, (cx, cy), r_outer, 255, -1)
    mask_ring = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    WHITE_LOW  = np.array([0,   0, 160], dtype=np.uint8)
    WHITE_HIGH = np.array([179, 70, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)

    white_in_center = cv2.countNonZero(cv2.bitwise_and(white_mask, mask_inner))
    white_in_ring   = cv2.countNonZero(cv2.bitwise_and(white_mask, mask_ring))

    total_center = max(cv2.countNonZero(mask_inner), 1)
    total_ring   = max(cv2.countNonZero(mask_ring), 1)

    ratio_center = white_in_center / total_center
    ratio_ring   = white_in_ring   / total_ring

    # Score alto = mais branco no anel = provável stripe
    stripe_score = ratio_ring - ratio_center
    return float(stripe_score), float(ratio_center), float(ratio_ring)

def draw_preview_patch(roi_bgr, detected_idx=None, total_detected=None):
    """
    Create a larger preview image for notebook display.
    """
    preview = roi_bgr.copy()

    mask = build_inner_circle_mask(preview.shape)

    overlay = preview.copy()
    overlay[mask == 0] = (40, 40, 40)
    preview = cv2.addWeighted(preview, 0.65, overlay, 0.35, 0)

    h, w = preview.shape[:2]
    preview = cv2.resize(
        preview,
        (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
        interpolation=cv2.INTER_LINEAR
    )

    if detected_idx is not None and total_detected is not None:
        text = f"Detection {detected_idx + 1}/{total_detected}"
        cv2.putText(
            preview,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return preview


# ============================================================
# Main annotator
# ============================================================

class BallCalibrationAnnotator:
    def __init__(
        self,
        image_paths,
        mask_fn,
        output_json="color_calibration.json",
        use_white_balance=True
    ):
        self.image_paths = [str(p) for p in image_paths]
        self.mask_fn = mask_fn
        self.output_json = output_json
        self.use_white_balance = use_white_balance

        self.samples = defaultdict(list)
        self.records = []

        self.current_image_idx = 0
        self.current_detection_idx = 0

        self.current_img_original = None
        self.current_img_balanced = None
        self.current_detections = []
        self.current_roi = None
        self.current_stats = None
        self.current_detection = None

        self.image_widget = widgets.Image(format="jpeg")
        self.full_image_widget = widgets.Image(format="jpeg")
        self.info_html = widgets.HTML()
        self.progress_html = widgets.HTML()

        self.buttons = []
        self._build_ui()
        self._load_next_valid_image()
        self._show_current_detection()

    def _make_button_handler(self, label):
        def handler(_):
            self._save_current_label(label)
        return handler

    def _build_ui(self):
        """
        Build button-based notebook UI.
        """
        label_colors = {
            "cue": "#f5f5f5",
            "black": "#222222",

            "yellow_solid": "#f4d03f",
            "yellow_stripe": "#f4d03f",

            "blue_solid": "#3498db",
            "blue_stripe": "#3498db",

            "red_solid": "#e74c3c",
            "red_stripe": "#e74c3c",

            "purple_solid": "#9b59b6",
            "purple_stripe": "#9b59b6",

            "orange_solid": "#e67e22",
            "orange_stripe": "#e67e22",

            "green_solid": "#27ae60",
            "green_stripe": "#27ae60",

            "maroon_solid": "#800000",
            "maroon_stripe": "#800000",

            "not_ball": "#7f8c8d",
            "skip": "#95a5a6",
            "stop_and_save": "#2c3e50",
        }

        dark_text_labels = {
            "cue",
            "yellow_solid",
            "yellow_stripe",
            "orange_solid",
            "orange_stripe",
        }

        self.buttons = []

        for label in LABELS:
            btn = widgets.Button(
                description=label,
                layout=widgets.Layout(width="170px", height="42px")
            )

            btn.style.button_color = label_colors.get(label, "#d3d3d3")

            if label in dark_text_labels:
                btn.style.text_color = "black"
            else:
                btn.style.text_color = "white"

            btn.on_click(self._make_button_handler(label))
            self.buttons.append(btn)

        buttons_grid = widgets.GridBox(
            self.buttons,
            layout=widgets.Layout(
                grid_template_columns="repeat(3, 180px)",
                grid_gap="8px 8px"
            )
        )

        images_box = widgets.HBox(
            [self.image_widget, self.full_image_widget],
            layout=widgets.Layout(align_items="flex-start", gap="20px")
        )

        display(self.progress_html)
        display(self.info_html)
        display(images_box)
        display(buttons_grid)

    def _build_counts_by_image(self):
        counts_by_image = {}

        valid_ball_labels = set(BALL_LABEL_TO_NUMBER.keys())

        for record in self.records:
            filename = Path(record["image_path"]).name
            label = record["label"]

            if filename not in counts_by_image:
                counts_by_image[filename] = {
                    "ball_count": 0,
                    **{ball_label: 0 for ball_label in BALL_LABEL_TO_NUMBER.keys()}
                }

            if label in valid_ball_labels:
                counts_by_image[filename][label] += 1
                counts_by_image[filename]["ball_count"] += 1

        return counts_by_image
        
    


    def _build_detections_by_image(self):
        detections_by_image = {}

        for record in self.records:
            filename = Path(record["image_path"]).name
            label = record["label"]
            x = int(record["x"])
            y = int(record["y"])
            r = int(record["r"])

            if filename not in detections_by_image:
                detections_by_image[filename] = {
                    "detections": []
                }

            if label in BALL_LABEL_TO_NUMBER:
                ball_number = BALL_LABEL_TO_NUMBER[label]
                is_real_ball = True
                is_false_detection = False
                was_skipped = False
            elif label in {"not_ball", "corner", "off_table"}:
                ball_number = None
                is_real_ball = False
                is_false_detection = True
                was_skipped = False
            elif label == "skip":
                ball_number = None
                is_real_ball = False
                is_false_detection = False
                was_skipped = True
            else:
                ball_number = None
                is_real_ball = False
                is_false_detection = False
                was_skipped = False

            if label.endswith("_solid"):
                ball_type = "solid"
                base_colour = label.replace("_solid", "")
            elif label.endswith("_stripe"):
                ball_type = "stripe"
                base_colour = label.replace("_stripe", "")
            elif label == "cue":
                ball_type = "cue"
                base_colour = "white"
            elif label == "black":
                ball_type = "black"
                base_colour = "black"
            else:
                ball_type = None
                base_colour = None

            bbox = [x - r, y - r, x + r, y + r]

            detections_by_image[filename]["detections"].append({
                "detection_index": int(record["detection_index"]),
                "x": x,
                "y": y,
                "r": r,
                "bbox": bbox,
                "label": label,
                "ball_number": ball_number,
                "ball_type": ball_type,
                "base_colour": base_colour,
                "is_real_ball": is_real_ball,
                "is_false_detection": is_false_detection,
                "was_skipped": was_skipped,
                "stats": record["stats"],
            })

        return detections_by_image

    

    def draw_full_image_preview(self, img_bgr, detections, current_idx):
        """
        Draw the full image with all detections.
        Current detection is red, others are green.
        """
        vis = img_bgr.copy()

        for i, (x, y, r) in enumerate(detections):
            if i == current_idx:
                color = (0, 0, 255)
                thickness = 3
            else:
                color = (0, 255, 0)
                thickness = 2

            cv2.circle(vis, (x, y), r, color, thickness)
            cv2.circle(vis, (x, y), 2, color, -1)
            cv2.putText(
                vis,
                str(i + 1),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        max_width = 500
        h, w = vis.shape[:2]
        if w > max_width:
            scale = max_width / w
            vis = cv2.resize(
                vis,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR
            )

        return vis

    def _load_image_and_detections(self, image_path):
        """
        Load image, compute table mask, optionally white-balance it,
        and detect balls using the original image.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None, None, []

        table_mask = self.mask_fn(img_bgr)
        if table_mask is None:
            return img_bgr, img_bgr.copy(), []

        balanced = gray_world_white_balance(img_bgr) if self.use_white_balance else img_bgr.copy()

        detections = detect_balls(img_bgr, table_mask)

        if detections is None:
            detections = []

        detections = np.array(detections)

        if detections.ndim == 3 and detections.shape[0] == 1:
            detections = detections[0]

        if detections.size == 0:
            detections = []
        else:
            detections = [tuple(map(int, d[:3])) for d in detections]

        return img_bgr, balanced, detections

    def _load_next_valid_image(self):
        """
        Move until finding an image with at least one detection.
        """
        while self.current_image_idx < len(self.image_paths):
            image_path = self.image_paths[self.current_image_idx]
            img_original, img_balanced, detections = self._load_image_and_detections(image_path)

            if img_original is None:
                self.current_image_idx += 1
                continue

            if len(detections) == 0:
                self.current_image_idx += 1
                continue

            self.current_img_original = img_original
            self.current_img_balanced = img_balanced
            self.current_detections = detections
            self.current_detection_idx = 0
            return

        self.current_img_original = None
        self.current_img_balanced = None
        self.current_detections = []
        self.current_detection_idx = 0

    def _advance(self):
        """
        Advance to the next detection, then next image if needed.
        """
        self.current_detection_idx += 1

        if self.current_detection_idx >= len(self.current_detections):
            self.current_image_idx += 1
            self._load_next_valid_image()

        self._show_current_detection()

    def _stop_and_save(self):
        """
        Save progress and stop annotation.
        """
        self._save_json()

        self.progress_html.value = (
            f"<b>Stopped by user.</b><br>"
            f"Images seen up to: {self.current_image_idx + 1}/{len(self.image_paths)}<br>"
            f"Saved labels: {len(self.records)}<br>"
            f"Output file: <code>{self.output_json}</code>"
        )

        self.info_html.value = "<b>Annotation stopped. Progress was saved.</b>"
        self.image_widget.value = b""
        self.full_image_widget.value = b""

        for btn in self.buttons:
            btn.disabled = True

        self.current_img_original = None
        self.current_img_balanced = None
        self.current_detections = []
        self.current_detection = None
        self.current_stats = None

    def _save_current_label(self, label):
        if label == "stop_and_save":
            self._stop_and_save()
            return

        if self.current_detection is None or self.current_stats is None:
            return

        image_path = self.image_paths[self.current_image_idx]
        x, y, r = self.current_detection

        record = {
            "image_path": image_path,
            "detection_index": int(self.current_detection_idx),
            "x": int(x),
            "y": int(y),
            "r": int(r),
            "label": label,
            "stats": self.current_stats,
        }

        self.records.append(record)
        INVALID_LABELS = {"skip", "not_ball", "corner", "off_table"}
        if label not in INVALID_LABELS:
            self.samples[label].append({
                "image_path": image_path,
                "x": int(x),
                "y": int(y),
                "r": int(r),
                **self.current_stats
            })

        if len(self.records) % AUTOSAVE_EVERY == 0:
            self._save_json()

        self._advance()

    def _show_current_detection(self):
        """
        Render current detection in notebook.
        """
        if self.current_img_balanced is None or len(self.current_detections) == 0:
            self.progress_html.value = (
                f"<b>Finished.</b><br>"
                f"Images processed: {len(self.image_paths)}<br>"
                f"Saved labels: {len(self.records)}<br>"
                f"Output file: <code>{self.output_json}</code>"
            )
            self.info_html.value = ""
            self.image_widget.value = b""
            self.full_image_widget.value = b""
            self._save_json()
            return

        image_path = self.image_paths[self.current_image_idx]
        x, y, r = self.current_detections[self.current_detection_idx]
        self.current_detection = (x, y, r)

        roi_bgr, _, _, _, _ = crop_ball_roi(self.current_img_balanced, x, y, r, pad=ROI_PAD)
        self.current_roi = roi_bgr
        self.current_stats = compute_patch_statistics(roi_bgr)

        preview = draw_preview_patch(
            roi_bgr,
            detected_idx=self.current_detection_idx,
            total_detected=len(self.current_detections)
        )

        ok_roi, encoded_roi = cv2.imencode(".jpg", preview)
        if ok_roi:
            self.image_widget.value = encoded_roi.tobytes()

        full_preview = self.draw_full_image_preview(
            self.current_img_original,
            self.current_detections,
            self.current_detection_idx
        )

        ok_full, encoded_full = cv2.imencode(".jpg", full_preview)
        if ok_full:
            self.full_image_widget.value = encoded_full.tobytes()

        self.progress_html.value = (
            f"<b>Image</b>: {self.current_image_idx + 1}/{len(self.image_paths)}"
            f" &nbsp; | &nbsp; "
            f"<b>Detection</b>: {self.current_detection_idx + 1}/{len(self.current_detections)}"
            f" &nbsp; | &nbsp; "
            f"<b>Total saved</b>: {len(self.records)}"
        )

        stats = self.current_stats
        self.info_html.value = (
            f"<b>File:</b> <code>{Path(image_path).name}</code><br>"
            f"<b>Circle:</b> x={x}, y={y}, r={r}<br>"
            f"<b>white_ratio:</b> {stats['white_ratio']:.3f} &nbsp; "
            f"<b>black_ratio:</b> {stats['black_ratio']:.3f} &nbsp; "
            f"<b>colour_ratio:</b> {stats['colour_ratio']:.3f}<br>"
            f"<b>HSV medians (colour only):</b> "
            f"H={stats['h_median']}, S={stats['s_median']}, V={stats['v_median']}"
        )

    def _save_json(self):
        """
        Save:
        1. calibration json
        2. counts by image json
        3. detailed detections by image json
        """
        compact = {}

        for label, items in self.samples.items():
            compact[label] = {
                "h": [item["h_median"] for item in items if item["h_median"] is not None],
                "s": [item["s_median"] for item in items if item["s_median"] is not None],
                "v": [item["v_median"] for item in items if item["v_median"] is not None],
                "white_ratio": [item["white_ratio"] for item in items],
                "black_ratio": [item["black_ratio"] for item in items],
                "colour_ratio": [item["colour_ratio"] for item in items],
                "samples": len(items),
            }

        calibration_payload = {
            "metadata": {
                "total_images": len(self.image_paths),
                "total_annotations": len(self.records),
                "labels": LABELS,
                "white_balance": self.use_white_balance,
                "inner_radius_factor": INNER_RADIUS_FACTOR,
                "roi_pad": ROI_PAD,
            },
            "compact_calibration": compact,
            "raw_annotations": self.records,
        }

        counts_by_image = self._build_counts_by_image()
        detections_by_image = self._build_detections_by_image()

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(calibration_payload, f, indent=4)

        counts_json_path = str(Path(self.output_json).with_name("ball_counts_by_image.json"))
        with open(counts_json_path, "w", encoding="utf-8") as f:
            json.dump(counts_by_image, f, indent=4)

        detections_json_path = str(Path(self.output_json).with_name("ball_annotations_by_image.json"))
        with open(detections_json_path, "w", encoding="utf-8") as f:
            json.dump(detections_by_image, f, indent=4)

        print(f"Saved calibration JSON to: {self.output_json}")
        print(f"Saved counts-by-image JSON to: {counts_json_path}")
        print(f"Saved detections-by-image JSON to: {detections_json_path}")

# %%
IMAGE_DIR = Path("development_set")

image_paths = sorted(
    [p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
)
''' If you want to select a specific image - If you run, pay attention to the output json files :)
TARGET_IMAGES = {
    "21a_png.rf.34ea0016904d75da861731071be2058f.jpg",
    "23a_png.rf.c51e6e9452d0f5b24425c277e7e93085.jpg",
    "123_png.rf.c77820fd3abbca5d6265af63564f9564.jpg",
    "127_png.rf.9f6501b0ba0e856cf6dd9e59f65ce574.jpg",
}
'''

image_paths = sorted(
    [p for p in IMAGE_DIR.glob("*") if p.name in TARGET_IMAGES]
)

print(f"Found {len(image_paths)} images")


def get_table_mask(img_bgr):
    mask, _ = segment_table(img_bgr)
    return mask

'''
annotator = BallCalibrationAnnotator(
    image_paths=image_paths,
    mask_fn=get_table_mask,
    output_json="color_calibration.json",
    use_white_balance=True
)
'''

# %% [markdown]
# # Final run code 

# %%
import json
from pathlib import Path
from collections import defaultdict, Counter

import cv2
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
import numpy as np


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = Path("development_set")
ANNOTATIONS_JSON = Path("ball_annotations_by_image_v2.json")
OUTPUT_JSON = Path("full_pipeline_predictions_v1.json")

VALID_LABELS = {
    "cue",
    "yellow_solid", "blue_solid", "red_solid", "purple_solid",
    "orange_solid", "green_solid", "maroon_solid", "black",
    "yellow_stripe", "blue_stripe", "red_stripe", "purple_stripe",
    "orange_stripe", "green_stripe", "maroon_stripe",
}
INVALID_LABELS = {"not_ball", "skip", "corner", "off_table"}

CHROMATIC_CLASSES = ["yellow", "blue", "red", "purple", "orange", "green", "maroon"]
REQUIRED_UNIQUE_LABELS = ["cue", "black"]

FEATURE_NAMES = [
    "hue_cos",
    "hue_sin",
    "s_median",
    "v_median",
    "white_ratio",
    "black_ratio",
    "colour_ratio",
    "h_std",
    "v_p90",
    "v_std",
    "s_p90",
    "stripe_score",
]

KNN_K = 5
COLOUR_CONFIDENCE_MIN_FOR_REASSIGN = 0.00
TYPE_CONFIDENCE_MIN_FOR_REASSIGN = 0.08

# Stage 1: validity rules
VALIDITY_THRESHOLDS = {
    "not_ball_white_ratio_max": 0.020,
    "not_ball_black_ratio_min": 0.180,
    "not_ball_h_std_max": 12.0,
    "suspect_white_ratio_center_min": 0.20,
    "suspect_center_minus_ring_min": 0.10,
    "ball_colour_ratio_min": 0.78,
    "ball_h_std_min": 18.0,
}

VALIDITY_WEIGHTS = {
    "very_low_white_ratio": -2.0,
    "high_black_ratio": -1.5,
    "low_h_std": -1.0,
    "high_white_ratio_center": -0.8,
    "high_center_minus_ring": -0.6,
    "good_colour_ratio": 0.8,
    "nonzero_white_ratio": 0.5,
    "good_h_std": 0.5,
}

VALIDITY_SCORE_THRESHOLDS = {
    "invalid_detection": -4.0,
    "suspect": -0.8,
}

# Stage 2: base colour rules
BLACK_RATIO_MIN_FOR_BLACK = 0.22
WHITE_RATIO_MIN_FOR_WHITE = 0.20
WHITE_VP90_MIN = 205.0
WHITE_S_MEDIAN_MAX = 170.0
WHITE_BLACK_RATIO_MAX = 0.08

# Stage 4: global consistency
GLOBAL_WEIGHTS = {
    "final_conf": 2.0,
    "base_conf": 0.5,
    "type_conf": 0.3,
    "validity_score": 0.8,
    "penalty_invalid": -100.0,
    "penalty_suspect": -1.5,
}



# ============================================================
# Generic helpers
# ============================================================

def safe_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        if np.isnan(value):
            return float(default)
    except Exception:
        pass
    return float(value)


def label_to_base_colour(label):
    if label == "cue":
        return "white"
    if label == "black":
        return "black"
    if label.endswith("_solid"):
        return label.replace("_solid", "")
    if label.endswith("_stripe"):
        return label.replace("_stripe", "")
    return None


def label_to_ball_type(label):
    if label == "cue":
        return "cue"
    if label == "black":
        return "black"
    if label.endswith("_solid"):
        return "solid"
    if label.endswith("_stripe"):
        return "stripe"
    return None


def final_label_from_parts(base_colour, ball_type):
    if base_colour == "white":
        return "cue"
    if base_colour == "black":
        return "black"
    if base_colour is None or ball_type is None:
        return "not_ball"
    return f"{base_colour}_{ball_type}"


def circular_mean_hue_opencv(h_values):
    clean = []
    for v in h_values:
        if v is None:
            continue
        try:
            if np.isnan(v):
                continue
        except Exception:
            pass
        clean.append(float(v))

    if not clean:
        return None

    angles = np.asarray(clean, dtype=float) * 2.0 * np.pi / 180.0
    angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    if angle < 0:
        angle += 2.0 * np.pi
    return float(angle * 180.0 / (2.0 * np.pi))


def circular_hue_diff(h1, h2):
    diff = abs(float(h1) - float(h2))
    return min(diff, 180.0 - diff)


# ============================================================
# Load annotations
# ============================================================

def load_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_detection_row(det, filename):
    stats = det.get("stats", {})

    white_ratio_center = safe_float(stats.get("white_ratio_center"), 0.0)
    white_ratio_ring = safe_float(stats.get("white_ratio_ring"), 0.0)

    row = {
        "filename": filename,
        "detection_index": int(det.get("detection_index", -1)),
        "x": int(det.get("x", -1)),
        "y": int(det.get("y", -1)),
        "r": int(det.get("r", -1)),
        "label_gt": det.get("label"),

        "white_ratio": stats.get("white_ratio"),
        "black_ratio": stats.get("black_ratio"),
        "colour_ratio": stats.get("colour_ratio"),
        "h_median": stats.get("h_median"),
        "s_median": stats.get("s_median"),
        "v_median": stats.get("v_median"),
        "h_p10": stats.get("h_p10"),
        "h_p90": stats.get("h_p90"),
        "h_std": stats.get("h_std"),
        "s_p90": stats.get("s_p90"),
        "v_p90": stats.get("v_p90"),
        "v_std": stats.get("v_std"),
        "stripe_score": stats.get("stripe_score"),
        "white_ratio_center": stats.get("white_ratio_center"),
        "white_ratio_ring": stats.get("white_ratio_ring"),
        "white_center_minus_ring": white_ratio_center - white_ratio_ring,
    }

    return row


# ============================================================
# Stage 2 training set for base colour
# Uses your current feature set and rules
# ============================================================

def build_base_colour_training_rows(annotations_by_image):
    rows = []

    for filename, payload in annotations_by_image.items():
        for det in payload.get("detections", []):
            label = det.get("label")
            if label not in VALID_LABELS:
                continue
            if not det.get("is_real_ball", False):
                continue

            row = build_detection_row(det, filename)
            row["base_colour"] = det.get("base_colour") or label_to_base_colour(label)
            row["ball_type"] = det.get("ball_type") or label_to_ball_type(label)
            rows.append(row)

    return rows


def robust_filter_group(rows, colour_name):
    if not rows:
        return []

    if colour_name == "white":
        vals_white = [safe_float(r["white_ratio"], 0.0) for r in rows]
        vals_v = [safe_float(r["v_median"], 0.0) for r in rows]
        q_white = np.quantile(vals_white, 0.05)
        q_v = np.quantile(vals_v, 0.05)

        return [
            r for r in rows
            if safe_float(r["white_ratio"], 0.0) >= q_white
            and safe_float(r["v_median"], 0.0) >= q_v
        ]

    if colour_name == "black":
        vals_black = [safe_float(r["black_ratio"], 0.0) for r in rows]
        q_black = np.quantile(vals_black, 0.10)

        return [r for r in rows if safe_float(r["black_ratio"], 0.0) >= q_black]

    vals_s = [safe_float(r["s_median"], 0.0) for r in rows]
    vals_v = [safe_float(r["v_median"], 0.0) for r in rows]

    s_lo, s_hi = np.quantile(vals_s, 0.05), np.quantile(vals_s, 0.95)
    v_lo, v_hi = np.quantile(vals_v, 0.05), np.quantile(vals_v, 0.95)

    filtered = [
        r for r in rows
        if s_lo <= safe_float(r["s_median"], 0.0) <= s_hi
        and v_lo <= safe_float(r["v_median"], 0.0) <= v_hi
    ]

    h_vals = [
        safe_float(r["h_median"], 0.0)
        for r in filtered
        if r.get("h_median") is not None
    ]
    if not h_vals:
        return filtered

    h_center = circular_mean_hue_opencv(h_vals)
    if h_center is None:
        return filtered

    dists = [circular_hue_diff(safe_float(r["h_median"], 0.0), h_center) for r in filtered]
    thr = np.quantile(dists, 0.90)

    out = []
    for r in filtered:
        dist = circular_hue_diff(safe_float(r["h_median"], 0.0), h_center)
        if dist <= thr:
            out.append(r)

    return out


def filter_training_rows(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["base_colour"]].append(r)

    out = []
    for colour_name, group_rows in grouped.items():
        out.extend(robust_filter_group(group_rows, colour_name))

    return out


def build_feature_vector_from_row(row):
    h = row["h_median"] if row["h_median"] is not None else None

    h_std = safe_float(row.get("h_std"), 0.0)
    s = safe_float(row.get("s_median"), 0.0)
    s_p90 = safe_float(row.get("s_p90"), 0.0)
    v = safe_float(row.get("v_median"), 0.0)
    v_p90 = safe_float(row.get("v_p90"), 0.0)
    v_std = safe_float(row.get("v_std"), 0.0)

    white_ratio = safe_float(row.get("white_ratio"), 0.0)
    black_ratio = safe_float(row.get("black_ratio"), 0.0)
    colour_ratio = safe_float(row.get("colour_ratio"), 0.0)
    stripe_score = safe_float(row.get("stripe_score"), 0.0)

    if h is None:
        hue_cos = 0.0
        hue_sin = 0.0
    else:
        angle = 2.0 * np.pi * float(h) / 180.0
        hue_cos = np.cos(angle)
        hue_sin = np.sin(angle)

    return np.array([
        hue_cos,
        hue_sin,
        s,
        v,
        white_ratio,
        black_ratio,
        colour_ratio,
        h_std,
        v_p90,
        v_std,
        s_p90,
        stripe_score,
    ], dtype=float)


def build_training_matrix(training_rows):
    X = np.vstack([build_feature_vector_from_row(r) for r in training_rows])
    y = np.asarray([r["base_colour"] for r in training_rows], dtype=object)

    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    Xz = (X - feature_mean) / feature_std
    return Xz, y, feature_mean, feature_std


# ============================================================
# Stage 1: validity
# ============================================================


def predict_validity(row):
    """
    Soft validity scoring only.

    Strategy:
    - no hard invalid_detection
    - everything gets a score
    - strong balls get higher score
    - suspicious detections get lower score
    - dark balls are explicitly protected
    - optional table-like penalty can be used if available
    """

    white_ratio = safe_float(row.get("white_ratio"), 0.0)
    black_ratio = safe_float(row.get("black_ratio"), 0.0)
    colour_ratio = safe_float(row.get("colour_ratio"), 0.0)
    h_std = safe_float(row.get("h_std"), 0.0)
    v_std = safe_float(row.get("v_std"), 0.0)
    s_median = safe_float(row.get("s_median"), 0.0)

    white_ratio_center = safe_float(row.get("white_ratio_center"), 0.0)
    white_ratio_ring = safe_float(row.get("white_ratio_ring"), 0.0)
    white_center_minus_ring = safe_float(
        row.get("white_center_minus_ring"),
        white_ratio_center - white_ratio_ring
    )

    # Optional future feature:
    # high value = region looks too similar to table cloth
    table_like_ratio = safe_float(row.get("table_like_ratio"), 0.0)

    reasons = []
    score = 0.0

    # =========================================================
    # 1. Explicit cue protection
    # =========================================================
    likely_cue = (
        white_ratio >= 0.30
        and white_ratio_center >= 0.70
        and black_ratio <= 0.03
    )

    if likely_cue:
        score += 3.0
        reasons.append("likely_cue_pattern")

    # =========================================================
    # 2. Explicit dark-ball protection
    # =========================================================
    likely_dark_ball = (
        black_ratio >= 0.20
        and colour_ratio >= 0.60
        and v_std >= 30.0
    )
    

    if likely_dark_ball:
        score += 1.5
        reasons.append("likely_dark_ball")
    
    likely_uniform_real_ball = (
    colour_ratio >= 0.97
    and black_ratio <= 0.05
    and white_ratio_center <= 0.02
    and white_ratio <= 0.02
    and v_std >= 18.0
)

    if likely_uniform_real_ball:
        score += 1.2
        reasons.append("likely_uniform_real_ball")

    likely_dark_artifact = (
        black_ratio >= 0.30
        and h_std <= 10.0
        and white_ratio_center <= 0.02
    )

    if likely_dark_artifact:
        score -= 2.0
        reasons.append("likely_dark_artifact")

    # =========================================================
    # 3. Positive evidence for real ball
    # =========================================================
    if white_ratio_center >= 0.02:
        score += 1.0
        reasons.append("nonzero_white_center")

    if white_ratio >= 0.02:
        score += 0.5
        reasons.append("nonzero_white_ratio")

    if h_std >= 12.0:
        score += 0.6
        reasons.append("good_h_std")

    if h_std >= 20.0:
        score += 0.6
        reasons.append("very_good_h_std")

    if v_std >= 20.0:
        score += 0.8
        reasons.append("good_v_std")

    if v_std >= 40.0:
        score += 0.5
        reasons.append("very_good_v_std")

    # Small positive only
    if colour_ratio >= 0.75:
        score += 0.2
        reasons.append("good_colour_ratio_small")

    # =========================================================
    # 4. Suspicious evidence
    # =========================================================


    if h_std < 10.0 and not likely_uniform_real_ball and not likely_dark_ball:
        score -= 1.0
        reasons.append("low_h_std")

    if h_std < 5.0 and not likely_uniform_real_ball and not likely_dark_ball:
        score -= 1.2
        reasons.append("very_low_h_std")

    # High saturation + very flat region can be suspicious
    if s_median >= 180.0 and h_std < 8.0:
        score -= 0.8
        reasons.append("high_sat_flat_region")

    # White concentrated too much in center can be suspicious
    # but do not punish cue-like cases
    if white_ratio_center < 0.01 and not likely_uniform_real_ball:
        score -= 1.0
        reasons.append("very_low_white_center")

    if white_ratio < 0.01 and not likely_uniform_real_ball:
        score -= 0.6
        reasons.append("very_low_white_ratio")

    if white_center_minus_ring > 0.10 and not likely_cue:
        score -= 0.4
        reasons.append("high_center_minus_ring")

    # Dark region penalty only if it does NOT look like a dark ball
    if black_ratio > 0.22 and not likely_dark_ball:
        score -= 0.5
        reasons.append("high_black_ratio_without_dark_ball_support")

    # Optional future penalty for table-like detections
    if table_like_ratio >= 0.50:
        score -= 1.2
        reasons.append("high_table_like_ratio")

    if table_like_ratio >= 0.70:
        score -= 1.0
        reasons.append("very_high_table_like_ratio")
    likely_black_ball = (
    black_ratio >= 0.14
    and colour_ratio >= 0.60
    and v_std >= 30.0
    and white_ratio_center <= 0.03
)
    if likely_black_ball:
        score += 1.8
        reasons.append("likely_black_ball")

    # =========================================================
    # 5. Final soft class
    # =========================================================
    # We keep the same API, but avoid hard invalids.
    if score <= -2.0:
        validity_class = "suspect"
    else:
        validity_class = "valid_ball"

    return {
        "validity_class": validity_class,
        "validity_score": float(score),
        "validity_reasons": reasons,
    }


# ============================================================
# Stage 2: base colour
# ============================================================

def apply_explicit_base_colour_rules(row):
    black_ratio = safe_float(row.get("black_ratio"), 0.0)
    white_ratio = safe_float(row.get("white_ratio"), 0.0)
    colour_ratio = safe_float(row.get("colour_ratio"), 0.0)
    h_median = safe_float(row.get("h_median"), 0.0)
    v_median = safe_float(row.get("v_median"), 0.0)
    s_median = safe_float(row.get("s_median"), 0.0)
    v_p90 = safe_float(row.get("v_p90"), v_median)
    ball_type = row.get("ball_type_hint")
    # Add these two — were missing
    white_ratio_center = safe_float(row.get("white_ratio_center"), 0.0)
    white_ratio_ring   = safe_float(row.get("white_ratio_ring"), 0.0)

    # Black
    if black_ratio >= 0.12 and white_ratio <= 0.12 and v_median <= 190:
        return "black"

    # Yellow stripe
    if (
        ball_type == "stripe"
        and 0.20 <= white_ratio <= 0.27
        and black_ratio <= 0.02
        and colour_ratio >= 0.72
        and h_median <= 95
    ):
        return "yellow"

    # White/cue — white must be spread across ball, not just a center glare
    if (
        white_ratio >= 0.15
        and v_p90 >= 190
        and s_median <= 190
        and black_ratio <= 0.10
        and white_ratio_center >= 0.10
        and white_ratio >= white_ratio_center * 0.6
    ):
        return "white"

    # Chromatic hue rules to reduce red/maroon/orange confusion
    s_min_chromatic = 100
    if s_median >= s_min_chromatic and colour_ratio >= 0.55:
        if h_median <= 7 or h_median >= 173:
            return "red"
        if 7 < h_median <= 16 and v_median <= 160:
            return "maroon"
        if 8 < h_median <= 22 and v_median > 130:
            return "orange"

    return None
# ============================================================
# Stage 3: ball type
# ============================================================

def predict_ball_type_with_probabilities(row, predicted_base_colour):
    if predicted_base_colour == "white":
        return {"top1": "cue", "ranked": [("cue", 1.0)], "confidence": 1.0, "source": "forced"}
    if predicted_base_colour == "black":
        return {"top1": "black", "ranked": [("black", 1.0)], "confidence": 1.0, "source": "forced"}

    stripe_score       = safe_float(row.get("stripe_score"), 0.0)
    white_ratio        = safe_float(row.get("white_ratio"), 0.0)
    white_ratio_center = safe_float(row.get("white_ratio_center"), 0.0)
    white_ratio_ring   = safe_float(row.get("white_ratio_ring"), 0.0)
    h_std              = safe_float(row.get("h_std"), 0.0)
    colour_ratio       = safe_float(row.get("colour_ratio"), 0.0)

    # Purple-specific guard: h_std is the strongest signal (F1=0.82)
    # solid has low h_std (median=10.6), stripe has higher (median=19.9)
    if predicted_base_colour == "purple":
        if h_std < 16.0 and colour_ratio > 0.93:
            score = -5.0  # almost certainly solid — specular glare fooling stripe_score
        else:
            score = -3.5 * stripe_score  # original formula for true stripes

    # All other colours: original formula unchanged
    else:
        score = 0.0
        score -= 3.5 * stripe_score
        if stripe_score < -0.03 and white_ratio_ring > 0.01:
            score += 1.5 * (white_ratio_center - white_ratio_ring)
        score += 0.8 * (white_ratio - 0.10)

    prob_stripe = 1.0 / (1.0 + np.exp(-6.0 * score))
    prob_solid  = 1.0 - prob_stripe

    ranked = sorted(
        [("stripe", prob_stripe), ("solid", prob_solid)],
        key=lambda x: x[1], reverse=True
    )
    return {
        "top1": ranked[0][0],
        "ranked": ranked,
        "confidence": float(ranked[0][1]),
        "source": "rule_score",
    }

# ============================================================
# Local prediction bundle
# ============================================================

def build_ranked_final_predictions(base_colour_pred, ball_type_pred):
    options = []

    for colour_cls, colour_prob in base_colour_pred["ranked"]:
        type_pred_alt = ball_type_pred if colour_cls == base_colour_pred["top1"] else None

        if colour_cls in ("white", "black"):
            ranked_types = [("cue", 1.0)] if colour_cls == "white" else [("black", 1.0)]
        else:
            if type_pred_alt is None:
                ranked_types = [("solid", 0.5), ("stripe", 0.5)]
            else:
                ranked_types = type_pred_alt["ranked"]

        for type_cls, type_prob in ranked_types:
            final_label = final_label_from_parts(colour_cls, type_cls)
            joint_prob = float(colour_prob) * float(type_prob)
            options.append({
                "final_label": final_label,
                "base_colour": colour_cls,
                "ball_type": type_cls,
                "joint_prob": joint_prob,
            })

    dedup = {}
    for item in sorted(options, key=lambda d: d["joint_prob"], reverse=True):
        if item["final_label"] not in dedup:
            dedup[item["final_label"]] = item

    return list(dedup.values())
    
def predict_base_colour_with_probabilities(row, Xz_train, y_train, feature_mean, feature_std, k=KNN_K):
    rule_pred = apply_explicit_base_colour_rules(row)
    if rule_pred is not None:
        return {
            "top1": rule_pred,
            "ranked": [(rule_pred, 1.0)],
            "confidence": 1.0,
            "source": "rule",
        }

    x = build_feature_vector_from_row(row)
    xz = (x - feature_mean) / feature_std

    dists = np.linalg.norm(Xz_train - xz, axis=1)
    nn_idx = np.argsort(dists)[:k]

    votes = defaultdict(float)
    for idx in nn_idx:
        cls = y_train[idx]
        if cls in ("white", "black"):
            continue
        votes[cls] += 1.0 / max(dists[idx], 1e-6)

    if not votes:
    # all neighbours were white/black — this detection is likely cue or black
        top_cls = y_train[nn_idx[0]]
        return {"top1": top_cls, "ranked": [(top_cls, 1.0)], "confidence": 1.0, "source": "knn_fallback"}

    total = sum(votes.values())
    ranked = sorted(
        [(cls, score / total) for cls, score in votes.items()],
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "top1": ranked[0][0],
        "ranked": ranked,
        "confidence": float(ranked[0][1]),
        "source": "knn",
    }


def predict_single_detection(row, Xz_train, y_train, feature_mean, feature_std):
    validity = predict_validity(row)

    base_colour_pred = predict_base_colour_with_probabilities(
        row, Xz_train, y_train, feature_mean, feature_std
    )

    ball_type_pred = predict_ball_type_with_probabilities(
        row, base_colour_pred["top1"]
    )

    ranked_final = build_ranked_final_predictions(base_colour_pred, ball_type_pred)
    pred_final_label = ranked_final[0]["final_label"] if ranked_final else "not_ball"
    pred_final_conf = ranked_final[0]["joint_prob"] if ranked_final else 0.0

    out = dict(row)
    out.update(validity)
    out["pred_base_colour"] = base_colour_pred["top1"]
    out["pred_base_colour_conf"] = float(base_colour_pred["confidence"])
    out["pred_ball_type"] = ball_type_pred["top1"]
    out["pred_ball_type_conf"] = float(ball_type_pred["confidence"])
    out["pred_final_label"] = pred_final_label
    out["pred_final_conf"] = float(pred_final_conf)
    out["ranked_final_predictions"] = ranked_final
    return out


# ============================================================
# Stage 4: global consistency
# ============================================================

def combined_strength(row):
    penalty = 0.0

    if row["validity_class"] == "suspect":
        penalty += -1.5

    return (
        GLOBAL_WEIGHTS["final_conf"] * float(row["pred_final_conf"]) +
        GLOBAL_WEIGHTS["base_conf"] * float(row["pred_base_colour_conf"]) +
        GLOBAL_WEIGHTS["type_conf"] * float(row["pred_ball_type_conf"]) +
        GLOBAL_WEIGHTS["validity_score"] * float(row["validity_score"]) +
        penalty
    )


def get_allowed_count_for_label(label):
    if label == "not_ball":
        return 999999
    if label == "cue":
        return 1
    if label == "black":
        return 1
    return 1


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

        # First preference: valid_ball
        tier = 0 if row["validity_class"] == "valid_ball" else (1 if row["validity_class"] == "suspect" else 2)
        score = row["strength"] + 2.0 * prob
        candidates.append((tier, score, idx, prob))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], -x[1]))
    _, _, idx, prob = candidates[0]
    return idx, prob


def try_reassign_row(row, assigned_labels, forbidden_current_label=None):
    counts = Counter(assigned_labels)

    for option in row["ranked_final_predictions"]:
        cand = option["final_label"]
        prob = float(option["joint_prob"])

        if cand == forbidden_current_label:
            continue

        if cand == "not_ball":
            continue

        if counts[cand] < get_allowed_count_for_label(cand):
            return cand, prob, "reassigned"

    return row["final_global_label"], row["final_global_conf"], "kept_duplicate"

def force_required_labels(rows):
    for required_label in REQUIRED_UNIQUE_LABELS:
        winner = choose_required_candidate(rows, required_label)
        if winner is None:
            continue

        winner_idx, winner_prob = winner
        rows[winner_idx]["final_global_label"] = required_label
        rows[winner_idx]["final_global_conf"] = winner_prob
        rows[winner_idx]["global_resolution"] = f"forced_required_{required_label}"

    return rows


def resolve_overflow_conflicts(rows):
    changed = True

    while changed:
        changed = False

        grouped = defaultdict(list)
        for idx, row in enumerate(rows):
            label = row["final_global_label"]
            if label != "not_ball":
                grouped[label].append((idx, row))

        for label, items in grouped.items():
            allowed = get_allowed_count_for_label(label)
            if len(items) <= allowed:
                continue

            items = sorted(items, key=lambda x: x[1]["strength"], reverse=True)
            overflow = items[allowed:]
            overflow_indices = {idx for idx, _ in overflow}

            kept_labels = [
                rows[i]["final_global_label"]
                for i in range(len(rows))
                if rows[i]["final_global_label"] != "not_ball" and i not in overflow_indices
            ]

            for idx, row in overflow:
                new_label, new_prob, mode = try_reassign_row(
                    row,
                    kept_labels,
                    forbidden_current_label=label
                )

                if new_label != row["final_global_label"]:
                    rows[idx]["final_global_label"] = new_label
                    rows[idx]["final_global_conf"] = new_prob
                    rows[idx]["global_resolution"] = mode
                    changed = True

    return rows


def resolve_global_consistency(pred_rows):
    rows = [dict(r) for r in pred_rows]

    for row in rows:
        row["strength"] = combined_strength(row)

    for row in rows:
        row["final_global_label"] = row["pred_final_label"]
        row["final_global_conf"] = float(row["pred_final_conf"])
        row["global_resolution"] = "kept_top1"

    rows = force_required_labels(rows)
    rows = resolve_overflow_conflicts(rows)
    rows = force_required_labels(rows)
    rows = resolve_overflow_conflicts(rows)

    # Final weak-candidate demotion
    for row in rows:
        if row["final_global_label"] in ("cue", "black"):
            continue

        if (
    row["validity_class"] == "suspect"
    and row["strength"] < -1
    and row["final_global_label"] not in ("cue", "black")
):
            row["final_global_label"] = "not_ball"
            row["final_global_conf"] = 0.0
            row["global_resolution"] = "weak_suspect_demoted"



    # Strong invalid final filter (extreme negatives only)

    for row in rows:
        if row["final_global_label"] in ("cue", "black"):
            continue

        if row["validity_score"] <= -3.5:
            row["final_global_label"] = "not_ball"
            row["final_global_conf"] = 0.0
            row["global_resolution"] = "final_strong_invalid"


    return sorted(rows, key=lambda d: (d["x"], d["y"]))


# ============================================================
# Per-image prediction
# ============================================================

def predict_image_detections(filename, detections, Xz_train, y_train, feature_mean, feature_std):
    pred_rows = []

    for det in detections:
        row = build_detection_row(det, filename)
        row["ball_type_hint"] = det.get("ball_type")
        pred = predict_single_detection(row, Xz_train, y_train, feature_mean, feature_std)
        pred_rows.append(pred)

    pred_rows = resolve_global_consistency(pred_rows)
    return pred_rows


# ============================================================
# Reporting
# ============================================================

def print_counts(title, items):
    print(f"\n{title}")
    counts = Counter(items)
    for key in sorted(counts.keys()):
        print(f"{key:25s} {counts[key]}")


def print_debug_summary(all_rows):
    print_counts(
        "Final global labels:",
        [r["final_global_label"] for r in all_rows]
    )
    print_counts(
        "Global resolution modes:",
        [r["global_resolution"] for r in all_rows]
    )
    print_counts(
        "Validity classes:",
        [r["validity_class"] for r in all_rows]
    )


# ============================================================
# Main run
# ============================================================

def run_full_pipeline(
    annotations_json=ANNOTATIONS_JSON,
    output_json=OUTPUT_JSON
):
    annotations_by_image = load_annotations(annotations_json)

    training_rows = build_base_colour_training_rows(annotations_by_image)
    training_rows = filter_training_rows(training_rows)
    Xz_train, y_train, feature_mean, feature_std = build_training_matrix(training_rows)

    output = {}
    all_rows = []

    for filename, payload in annotations_by_image.items():
        detections = payload.get("detections", [])  # todas, sem filtro
    
        preds = predict_image_detections(
            filename,
            detections,
            Xz_train, y_train, feature_mean, feature_std,
    )

        output[filename] = {
            "predictions": preds
        }
        all_rows.extend(preds)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print_debug_summary(all_rows)
    return output, all_rows, (Xz_train, y_train, feature_mean, feature_std)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


output, all_rows, model_bundle = run_full_pipeline(
    annotations_json=ANNOTATIONS_JSON,
    output_json=OUTPUT_JSON,
)
def evaluate_final_global(all_rows):
    y_true = []
    y_pred = []

    for row in all_rows:
        gt = "valid_ball" if row["label_gt"] in VALID_LABELS else "not_valid_ball"
        pred = "not_valid_ball" if row["final_global_label"] == "not_ball" else "valid_ball"

        y_true.append(gt)
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_true, y_pred), 4))
    print("Weighted F1:", round(f1_score(y_true, y_pred, average="weighted"), 4))
    print("Macro F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
    print("Recall not_valid_ball:", round(
        recall_score(y_true, y_pred, pos_label="not_valid_ball"), 4
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["not_valid_ball", "valid_ball"]))

    return y_true, y_pred

for row in all_rows:
    gt = "valid_ball" if row["label_gt"] in VALID_LABELS else "not_valid_ball"
    pred = row["final_global_label"]
    if gt == "valid_ball" and pred == "not_ball":
        print(
            f"FN | resolution={row['global_resolution']} "
            f"| validity_score={row['validity_score']:.2f} "
            f"| strength={row['strength']:.2f} "
            f"| validity_class={row['validity_class']}"
        )

# Depois o evaluate
evaluate_final_global(all_rows)

# %%
def evaluate_colour_and_type(all_rows):
    # Filtra só bolas reais com label válido
    valid_rows = [r for r in all_rows if r["label_gt"] in VALID_LABELS]

    # ── Base colour ──────────────────────────────────────────
    y_true_colour = [label_to_base_colour(r["label_gt"]) for r in valid_rows]
    y_pred_colour = [r["pred_base_colour"] for r in valid_rows]

    correct_colour = sum(t == p for t, p in zip(y_true_colour, y_pred_colour))
    print(f"\n{'='*50}")
    print(f"BASE COLOUR accuracy: {correct_colour}/{len(valid_rows)} "
          f"= {correct_colour/len(valid_rows):.4f}")

    # Erros por cor
    errors_colour = defaultdict(list)
    for r, t, p in zip(valid_rows, y_true_colour, y_pred_colour):
        if t != p:
            errors_colour[t].append(p)

    if errors_colour:
        print("\nErros de base colour (true → pred):")
        for true_col, preds in sorted(errors_colour.items()):
            from collections import Counter
            print(f"  {true_col:12s} → {dict(Counter(preds))}")
    else:
        print("Sem erros de base colour!")

    # ── Ball type (stripe vs solid) ──────────────────────────
    # Só bolas cromáticas (excluir cue e black)
    type_rows = [
        r for r in valid_rows
        if label_to_ball_type(r["label_gt"]) in ("solid", "stripe")
    ]
    y_true_type = [label_to_ball_type(r["label_gt"]) for r in type_rows]
    y_pred_type = [r["pred_ball_type"] for r in type_rows]

    correct_type = sum(t == p for t, p in zip(y_true_type, y_pred_type))
    print(f"\n{'='*50}")
    print(f"BALL TYPE (stripe/solid) accuracy: {correct_type}/{len(type_rows)} "
          f"= {correct_type/len(type_rows):.4f}")

    # Breakdown por tipo
    stripe_rows = [(t, p) for t, p in zip(y_true_type, y_pred_type) if t == "stripe"]
    solid_rows  = [(t, p) for t, p in zip(y_true_type, y_pred_type) if t == "solid"]

    stripe_correct = sum(t == p for t, p in stripe_rows)
    solid_correct  = sum(t == p for t, p in solid_rows)

    print(f"  Stripe recall: {stripe_correct}/{len(stripe_rows)} = "
          f"{stripe_correct/max(len(stripe_rows),1):.4f}")
    print(f"  Solid  recall: {solid_correct}/{len(solid_rows)} = "
          f"{solid_correct/max(len(solid_rows),1):.4f}")

    # Erros de tipo com detalhe de cor
    print("\nErros de tipo (label_gt → pred_ball_type):")
    type_errors = [
        r for r, t, p in zip(type_rows, y_true_type, y_pred_type) if t != p
    ]
    if type_errors:
        for r in type_errors[:20]:
            print(f"  {r['label_gt']:20s} | stripe_score={r.get('stripe_score', 'N/A')} "
                  f"| white_ratio={r.get('white_ratio', 0):.3f} "
                  f"| white_ring={r.get('white_ratio_ring', 0):.3f} "
                  f"| white_center={r.get('white_ratio_center', 0):.3f}")
    else:
        print("  Sem erros de tipo!")

    # ── Label final completo ─────────────────────────────────
    y_true_final = [r["label_gt"] for r in valid_rows]
    y_pred_final = [r["final_global_label"] for r in valid_rows]
    correct_final = sum(t == p for t, p in zip(y_true_final, y_pred_final))
    print(f"\n{'='*50}")
    print(f"LABEL FINAL completo accuracy: {correct_final}/{len(valid_rows)} "
          f"= {correct_final/len(valid_rows):.4f}")

    # Erros finais agrupados
    final_errors = defaultdict(list)
    for t, p in zip(y_true_final, y_pred_final):
        if t != p:
            final_errors[t].append(p)
    if final_errors:
        print("\nErros de label final (true → pred):")
        for true_lbl, preds in sorted(final_errors.items()):
            from collections import Counter
            print(f"  {true_lbl:25s} → {dict(Counter(preds))}")

evaluate_colour_and_type(all_rows)

print("\n=== VALIDITY EVALUATION ===")
for row in all_rows:
    gt = row["label_gt"]
    pred = row["final_global_label"]
    if gt == "not_ball" and pred != "not_ball":
        print(f"FP | pred={pred} | validity_score={row['validity_score']:.2f} | strength={row['strength']:.2f}")
    if gt in VALID_LABELS and pred == "not_ball":
        print(f"FN | gt={gt} | resolution={row['global_resolution']} | validity_score={row['validity_score']:.2f}")

not_ball_gt = [r for r in all_rows if r["label_gt"] == "not_ball"]
not_ball_correct = sum(r["final_global_label"] == "not_ball" for r in not_ball_gt)
valid_demoted = sum(r["final_global_label"] == "not_ball" for r in all_rows if r["label_gt"] in VALID_LABELS)
print(f"\nnot_ball correctly rejected: {not_ball_correct}/{len(not_ball_gt)}")
print(f"valid balls wrongly demoted to not_ball: {valid_demoted}")


# Debug: check what the white rule is actually doing on cue balls
print("=== WHITE RULE DEBUG ===")
cue_rows = [r for r in all_rows if r["label_gt"] == "cue"]
print(f"Total cue balls: {len(cue_rows)}")

for r in cue_rows:
    white_ratio        = safe_float(r.get("white_ratio"), 0.0)
    v_p90              = safe_float(r.get("v_p90"), 0.0)
    s_median           = safe_float(r.get("s_median"), 0.0)
    black_ratio        = safe_float(r.get("black_ratio"), 0.0)
    pred_colour        = r.get("pred_base_colour")
    final_label        = r.get("final_global_label")

    passed_white = (
        white_ratio >= 0.18
        and v_p90 >= 195
        and s_median <= 175
        and black_ratio <= 0.09
    )

    if not passed_white or pred_colour != "white":
        print(
            f"MISSED | pred={pred_colour:10s} final={final_label:20s} "
            f"| white_ratio={white_ratio:.3f} v_p90={v_p90:.1f} "
            f"s_median={s_median:.1f} black_ratio={black_ratio:.3f} "
            f"| rule_pass={passed_white}"
        )

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %%
"""
Hill Climbing Optimizer v3
==========================
Key improvements over v2:
  - KNN built separately for each split using training images only
  - Multiple splits (N_SPLITS seeds) — objective = mean score across splits
  - Multiple restarts per stage — random perturbation to escape local optima
  - Larger search space with coarser steps for faster exploration
  - Rebalanced objective weights (colour is the real bottleneck)
"""

import json
import csv
import random
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ── paths ─────────────────────────────────────────────────────────────────────
ANNOTATIONS_JSON = Path("ball_annotations_by_image_v2.json")
GROUND_TRUTH_CSV = Path("ground_truth_counts_excel.csv")

VALID_LABELS = {
    "cue",
    "yellow_solid", "blue_solid", "red_solid", "purple_solid",
    "orange_solid", "green_solid", "maroon_solid", "black",
    "yellow_stripe", "blue_stripe", "red_stripe", "purple_stripe",
    "orange_stripe", "green_stripe", "maroon_stripe",
}
INVALID_LABELS = {"not_ball", "skip", "corner", "off_table"}
REQUIRED_UNIQUE_LABELS = ["cue", "black"]

# ── multi-split config ────────────────────────────────────────────────────────
N_SPLITS = 6
VAL_RATIO = 0.25
TEST_RATIO = 0.15
SPLIT_SEEDS = [42, 7, 13, 99, 2024, 17]

# ── hill climbing config ──────────────────────────────────────────────────────

N_RESTARTS = 3
MAX_ITER = 500
PATIENCE = 50
PERTURB_FRAC = 0.25

# Stage-1 stronger search
STAGE1_N_RESTARTS = 7
STAGE1_MAX_ITER = 900
STAGE1_PATIENCE = 120
STAGE1_PERTURB_FRAC = 0.35

# ── objective weights ─────────────────────────────────────────────────────────
OBJ_WEIGHTS = {
    "validity_recall_balls": 1.5,
    "validity_fp_rate": -1.2,
    "colour_accuracy": 3.0,
    "type_accuracy": 1.0,
    "final_f1_macro": 2.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    "s1_cue_white_ratio_min": 0.30,
    "s1_cue_white_center_min": 0.70,
    "s1_cue_black_ratio_max": 0.03,
    "s1_cue_score_bonus": 3.0,

    "s1_dark_black_ratio_min": 0.20,
    "s1_dark_colour_ratio_min": 0.60,
    "s1_dark_v_std_min": 30.0,
    "s1_dark_score_bonus": 1.5,

    "s1_uniform_colour_ratio_min": 0.97,
    "s1_uniform_black_ratio_max": 0.05,
    "s1_uniform_white_center_max": 0.02,
    "s1_uniform_white_ratio_max": 0.02,
    "s1_uniform_v_std_min": 18.0,
    "s1_uniform_score_bonus": 1.2,

    "s1_artifact_black_ratio_min": 0.30,
    "s1_artifact_h_std_max": 10.0,
    "s1_artifact_white_center_max": 0.02,
    "s1_artifact_score_penalty": -2.0,

    "s1_pos_white_center_min": 0.02,
    "s1_pos_white_center_bonus": 1.0,
    "s1_pos_white_ratio_min": 0.02,
    "s1_pos_white_ratio_bonus": 0.5,
    "s1_pos_h_std_good_min": 12.0,
    "s1_pos_h_std_good_bonus": 0.6,
    "s1_pos_h_std_vgood_min": 20.0,
    "s1_pos_h_std_vgood_bonus": 0.6,
    "s1_pos_v_std_good_min": 20.0,
    "s1_pos_v_std_good_bonus": 0.8,
    "s1_pos_v_std_vgood_min": 40.0,
    "s1_pos_v_std_vgood_bonus": 0.5,
    "s1_pos_colour_ratio_min": 0.75,
    "s1_pos_colour_ratio_bonus": 0.2,

    "s1_sus_h_std_low_thr": 13.0,
    "s1_sus_h_std_low_penalty": -1.0,
    "s1_sus_h_std_vlow_thr": 5.0,
    "s1_sus_h_std_vlow_penalty": -1.2,
    "s1_sus_sat_thr": 180.0,
    "s1_sus_sat_h_std_thr": 8.0,
    "s1_sus_sat_penalty": -0.8,
    "s1_sus_white_center_vlow_thr": 0.01,
    "s1_sus_white_center_vlow_pen": -1.0,
    "s1_sus_white_ratio_vlow_thr": 0.01,
    "s1_sus_white_ratio_vlow_pen": -0.6,
    "s1_sus_center_minus_ring_thr": 0.10,
    "s1_sus_center_minus_ring_pen": -0.4,
    "s1_sus_black_ratio_thr": 0.22,
    "s1_sus_black_ratio_pen": -0.5,

    "s1_black_black_ratio_min": 0.14,
    "s1_black_colour_ratio_min": 0.60,
    "s1_black_v_std_min": 30.0,
    "s1_black_white_center_max": 0.03,
    "s1_black_score_bonus": 1.8,

    "s1_suspect_score_thr": -2.0,

    "s2_black_black_ratio_min": 0.12,
    "s2_black_white_ratio_max": 0.12,
    "s2_black_v_median_max": 190.0,

    "s2_yellow_white_ratio_lo": 0.20,
    "s2_yellow_white_ratio_hi": 0.27,
    "s2_yellow_black_ratio_max": 0.02,
    "s2_yellow_colour_ratio_min": 0.72,
    "s2_yellow_h_median_max": 100.0,

    "s2_white_white_ratio_min": 0.15,
    "s2_white_v_p90_min": 190.0,
    "s2_white_s_median_max": 190.0,
    "s2_white_black_ratio_max": 0.10,
    "s2_white_center_min": 0.10,
    "s2_white_center_ratio": 0.60,

    "s2_chrom_s_min": 100.0,
    "s2_chrom_colour_ratio_min": 0.55,
    "s2_red_h_max": 7.0,
    "s2_red_h_min_wrap": 173.0,
    "s2_maroon_h_lo": 7.0,
    "s2_maroon_h_hi": 16.0,
    "s2_maroon_v_max": 160.0,
    "s2_orange_h_lo": 8.0,
    "s2_orange_h_hi": 22.0,
    "s2_orange_v_min": 130.0,

    "s3_stripe_score_coef": 3.5,
    "s3_white_ratio_ring_thr": 0.01,
    "s3_center_ring_diff_coef": 1.5,
    "s3_white_ratio_offset": 0.10,
    "s3_white_ratio_coef": 1.2,
    "s3_sigmoid_scale": 6.0,

    "s3_purple_h_std_thr": 16.0,
    "s3_purple_colour_ratio_thr": 0.88,
    "s3_purple_solid_score": -5.0,
    "s3_purple_stripe_coef": 3.5,

    "s4_w_final_conf": 2.0,
    "s4_w_base_conf": 0.5,
    "s4_w_type_conf": 0.3,
    "s4_w_validity_score": 0.8,
    "s4_suspect_penalty": -1.5,
    "s4_weak_suspect_strength_thr": -1.0,
    "s4_strong_invalid_score_thr": -3.5,
}

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH SPACE
# ═══════════════════════════════════════════════════════════════════════════════

SEARCH_SPACE = {
    "s1_cue_white_ratio_min": (0.10, 0.55, 0.05),
    "s1_cue_white_center_min": (0.30, 0.95, 0.05),
    "s1_cue_black_ratio_max": (0.01, 0.15, 0.02),
    "s1_cue_score_bonus": (1.0, 6.0, 0.5),

    "s1_dark_black_ratio_min": (0.08, 0.40, 0.04),
    "s1_dark_colour_ratio_min": (0.30, 0.85, 0.05),
    "s1_dark_v_std_min": (10.0, 55.0, 5.0),
    "s1_dark_score_bonus": (0.5, 4.0, 0.5),

    "s1_uniform_colour_ratio_min": (0.80, 0.99, 0.03),
    "s1_uniform_v_std_min": (8.0, 40.0, 4.0),
    "s1_uniform_score_bonus": (0.3, 3.0, 0.3),

    "s1_artifact_black_ratio_min": (0.10, 0.55, 0.05),
    "s1_artifact_h_std_max": (3.0, 25.0, 2.0),
    "s1_artifact_score_penalty": (-5.0, -0.5, 0.5),

    "s1_pos_white_center_bonus": (0.2, 2.5, 0.3),
    "s1_pos_white_ratio_bonus": (0.1, 1.5, 0.2),
    "s1_pos_h_std_good_bonus": (0.1, 2.0, 0.3),
    "s1_pos_v_std_good_bonus": (0.2, 2.5, 0.3),
    "s1_pos_colour_ratio_bonus": (0.0, 1.0, 0.2),

    "s1_sus_h_std_low_thr": (5.0, 22.0, 2.0),
    "s1_sus_h_std_low_penalty": (-3.0, -0.2, 0.4),
    "s1_sus_h_std_vlow_penalty": (-4.0, -0.3, 0.5),
    "s1_sus_white_center_vlow_pen": (-3.0, -0.2, 0.4),
    "s1_sus_white_ratio_vlow_pen": (-2.0, -0.1, 0.3),
    "s1_sus_center_minus_ring_pen": (-1.5, -0.1, 0.2),
    "s1_sus_black_ratio_pen": (-2.0, -0.1, 0.3),
    "s1_sus_sat_penalty": (-2.0, -0.1, 0.3),

    "s1_black_black_ratio_min": (0.06, 0.30, 0.02),
    "s1_black_colour_ratio_min": (0.30, 0.85, 0.05),
    "s1_black_v_std_min": (10.0, 55.0, 5.0),
    "s1_black_score_bonus": (0.5, 4.0, 0.5),

    "s1_suspect_score_thr": (-5.0, -0.5, 0.5),

    "s2_black_black_ratio_min": (0.04, 0.30, 0.02),
    "s2_black_white_ratio_max": (0.04, 0.30, 0.02),
    "s2_black_v_median_max": (140.0, 230.0, 10.0),

    "s2_yellow_white_ratio_lo": (0.08, 0.35, 0.03),
    "s2_yellow_white_ratio_hi": (0.18, 0.45, 0.03),
    "s2_yellow_black_ratio_max": (0.01, 0.10, 0.01),
    "s2_yellow_colour_ratio_min": (0.50, 0.95, 0.05),
    "s2_yellow_h_median_max": (60.0, 130.0, 5.0),

    "s2_white_white_ratio_min": (0.05, 0.35, 0.02),
    "s2_white_v_p90_min": (150.0, 230.0, 5.0),
    "s2_white_s_median_max": (120.0, 240.0, 10.0),
    "s2_white_black_ratio_max": (0.02, 0.25, 0.02),
    "s2_white_center_min": (0.02, 0.40, 0.02),
    "s2_white_center_ratio": (0.20, 0.95, 0.05),

    "s2_chrom_s_min": (50.0, 180.0, 10.0),
    "s2_chrom_colour_ratio_min": (0.25, 0.80, 0.05),
    "s2_red_h_max": (3.0, 15.0, 1.0),
    "s2_red_h_min_wrap": (160.0, 179.0, 2.0),
    "s2_maroon_h_lo": (4.0, 12.0, 1.0),
    "s2_maroon_h_hi": (10.0, 28.0, 2.0),
    "s2_maroon_v_max": (110.0, 200.0, 10.0),
    "s2_orange_h_lo": (5.0, 18.0, 1.0),
    "s2_orange_h_hi": (15.0, 35.0, 2.0),
    "s2_orange_v_min": (80.0, 180.0, 10.0),

    "s3_stripe_score_coef": (1.0, 8.0, 0.5),
    "s3_center_ring_diff_coef": (0.5, 4.0, 0.5),
    "s3_white_ratio_offset": (0.02, 0.25, 0.02),
    "s3_white_ratio_coef": (0.2, 2.5, 0.2),
    "s3_sigmoid_scale": (2.0, 14.0, 1.0),

    "s3_purple_h_std_thr": (6.0, 30.0, 2.0),
    "s3_purple_colour_ratio_thr": (0.75, 0.99, 0.03),
    "s3_purple_solid_score": (-10.0, -1.0, 1.0),
    "s3_purple_stripe_coef": (1.0, 8.0, 0.5),

    "s4_w_final_conf": (0.5, 5.0, 0.5),
    "s4_w_base_conf": (0.1, 2.0, 0.2),
    "s4_w_type_conf": (0.0, 1.5, 0.1),
    "s4_w_validity_score": (0.1, 2.5, 0.2),
    "s4_suspect_penalty": (-4.0, -0.2, 0.4),
    "s4_weak_suspect_strength_thr": (-4.0, 1.0, 0.5),
    "s4_strong_invalid_score_thr": (-8.0, -0.5, 0.5),
}

STAGE_PARAMS = {
    1: [k for k in SEARCH_SPACE if k.startswith("s1_")],
    2: [k for k in SEARCH_SPACE if k.startswith("s2_")],
    3: [k for k in SEARCH_SPACE if k.startswith("s3_")],
    4: [k for k in SEARCH_SPACE if k.startswith("s4_")],
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def gt_final_label(label):
    return label if label in VALID_LABELS else "not_ball"

def safe_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        if np.isnan(value):
            return float(default)
    except Exception:
        pass
    return float(value)

def label_to_base_colour(label):
    if label == "cue":
        return "white"
    if label == "black":
        return "black"
    if label.endswith("_solid"):
        return label.replace("_solid", "")
    if label.endswith("_stripe"):
        return label.replace("_stripe", "")
    return None

def label_to_ball_type(label):
    if label == "cue":
        return "cue"
    if label == "black":
        return "black"
    if label.endswith("_solid"):
        return "solid"
    if label.endswith("_stripe"):
        return "stripe"
    return None

def final_label_from_parts(base_colour, ball_type):
    if base_colour == "white":
        return "cue"
    if base_colour == "black":
        return "black"
    if base_colour is None or ball_type is None:
        return "not_ball"
    return f"{base_colour}_{ball_type}"

def circular_mean_hue(h_values):
    clean = [float(v) for v in h_values if v is not None and not np.isnan(v)]
    if not clean:
        return None
    angles = np.asarray(clean) * 2.0 * np.pi / 180.0
    angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    if angle < 0:
        angle += 2.0 * np.pi
    return float(angle * 180.0 / (2.0 * np.pi))

def circular_hue_diff(h1, h2):
    diff = abs(float(h1) - float(h2))
    return min(diff, 180.0 - diff)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ground_truth_csv(csv_path):
    skip_cols = {"filename", "ball_count", "", None}
    gt = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            gt[fname] = {}
            for k, v in row.items():
                if k is None:
                    continue
                k = k.strip()
                if k in skip_cols or k.startswith("Unnamed"):
                    continue
                v = (v or "").strip()
                if v:
                    try:
                        gt[fname][k] = int(v)
                    except ValueError:
                        pass
    return gt

def build_detection_row(det, filename):
    stats = det.get("stats", {})
    wrc = safe_float(stats.get("white_ratio_center"), 0.0)
    wrr = safe_float(stats.get("white_ratio_ring"), 0.0)
    return {
        "filename": filename,
        "detection_index": int(det.get("detection_index", -1)),
        "x": int(det.get("x", -1)),
        "y": int(det.get("y", -1)),
        "r": int(det.get("r", -1)),
        "label_gt": det.get("label"),
        "white_ratio": stats.get("white_ratio"),
        "black_ratio": stats.get("black_ratio"),
        "colour_ratio": stats.get("colour_ratio"),
        "h_median": stats.get("h_median"),
        "s_median": stats.get("s_median"),
        "v_median": stats.get("v_median"),
        "h_p10": stats.get("h_p10"),
        "h_p90": stats.get("h_p90"),
        "h_std": stats.get("h_std"),
        "s_p90": stats.get("s_p90"),
        "v_p90": stats.get("v_p90"),
        "v_std": stats.get("v_std"),
        "stripe_score": stats.get("stripe_score"),
        "white_ratio_center": stats.get("white_ratio_center"),
        "white_ratio_ring": stats.get("white_ratio_ring"),
        "white_center_minus_ring": wrc - wrr,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# KNN — built per split using TRAIN images only
# ═══════════════════════════════════════════════════════════════════════════════

def _robust_filter_group(rows, colour_name):
    if not rows:
        return []

    if colour_name == "white":
        qw = np.quantile([safe_float(r["white_ratio"]) for r in rows], 0.05)
        qv = np.quantile([safe_float(r["v_median"]) for r in rows], 0.05)
        return [
            r for r in rows
            if safe_float(r["white_ratio"]) >= qw
            and safe_float(r["v_median"]) >= qv
        ]

    if colour_name == "black":
        qb = np.quantile([safe_float(r["black_ratio"]) for r in rows], 0.10)
        return [r for r in rows if safe_float(r["black_ratio"]) >= qb]

    s_vals = [safe_float(r["s_median"]) for r in rows]
    v_vals = [safe_float(r["v_median"]) for r in rows]
    s_lo, s_hi = np.quantile(s_vals, 0.05), np.quantile(s_vals, 0.95)
    v_lo, v_hi = np.quantile(v_vals, 0.05), np.quantile(v_vals, 0.95)

    filtered = [
        r for r in rows
        if s_lo <= safe_float(r["s_median"]) <= s_hi
        and v_lo <= safe_float(r["v_median"]) <= v_hi
    ]

    h_vals = [safe_float(r["h_median"]) for r in filtered if r.get("h_median") is not None]
    if not h_vals:
        return filtered

    h_center = circular_mean_hue(h_vals)
    if h_center is None:
        return filtered

    dists = [circular_hue_diff(safe_float(r["h_median"]), h_center) for r in filtered]
    thr = np.quantile(dists, 0.90)
    return [r for r, d in zip(filtered, dists) if d <= thr]

def _feature_vec(row):
    h = row["h_median"]
    if h is None:
        hc, hs = 0.0, 0.0
    else:
        angle = 2.0 * np.pi * float(h) / 180.0
        hc, hs = np.cos(angle), np.sin(angle)

    return np.array([
        hc, hs,
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

def build_knn_from_images(annotations_by_image, image_list):
    rows = []
    for filename in image_list:
        payload = annotations_by_image.get(filename, {})
        for det in payload.get("detections", []):
            label = det.get("label")
            if label not in VALID_LABELS:
                continue
            if not det.get("is_real_ball", False):
                continue

            row = build_detection_row(det, filename)
            row["base_colour"] = det.get("base_colour") or label_to_base_colour(label)
            row["ball_type"] = det.get("ball_type") or label_to_ball_type(label)
            rows.append(row)

    if not rows:
        raise ValueError("No valid training rows found for KNN.")

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["base_colour"]].append(r)

    filtered = []
    for colour_name, group in grouped.items():
        filtered.extend(_robust_filter_group(group, colour_name))

    if not filtered:
        raise ValueError("No filtered training rows remained for KNN.")

    X = np.vstack([_feature_vec(r) for r in filtered])
    y = np.asarray([r["base_colour"] for r in filtered], dtype=object)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0

    Xz = (X - mean) / std
    return Xz, y, mean, std

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITS
# ═══════════════════════════════════════════════════════════════════════════════

def make_splits(annotations_by_image):
    all_images = sorted(annotations_by_image.keys())
    n = len(all_images)

    rng0 = random.Random(0)
    shuffled = all_images[:]
    rng0.shuffle(shuffled)

    n_test = max(1, int(n * TEST_RATIO))
    test_images = shuffled[:n_test]
    remaining = shuffled[n_test:]

    n_val = max(1, int(len(remaining) * VAL_RATIO))

    splits = []
    for seed in SPLIT_SEEDS[:N_SPLITS]:
        rng = random.Random(seed)
        pool = remaining[:]
        rng.shuffle(pool)

        val_images = pool[:n_val]
        train_images = pool[n_val:]

        splits.append({
            "train_images": train_images,
            "val_images": val_images,
        })

    return splits, test_images

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_validity(row, p):
    wr = safe_float(row.get("white_ratio"))
    br = safe_float(row.get("black_ratio"))
    cr = safe_float(row.get("colour_ratio"))
    hs = safe_float(row.get("h_std"))
    vs = safe_float(row.get("v_std"))
    sm = safe_float(row.get("s_median"))
    wrc = safe_float(row.get("white_ratio_center"))
    wrr = safe_float(row.get("white_ratio_ring"))
    cmr = safe_float(row.get("white_center_minus_ring"), wrc - wrr)
    score = 0.0

    likely_cue = (
        wr >= p["s1_cue_white_ratio_min"]
        and wrc >= p["s1_cue_white_center_min"]
        and br <= p["s1_cue_black_ratio_max"]
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
        cr >= p["s1_uniform_colour_ratio_min"]
        and br <= p["s1_uniform_black_ratio_max"]
        and wrc <= p["s1_uniform_white_center_max"]
        and wr <= p["s1_uniform_white_ratio_max"]
        and vs >= p["s1_uniform_v_std_min"]
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
        br >= p["s1_black_black_ratio_min"]
        and cr >= p["s1_black_colour_ratio_min"]
        and vs >= p["s1_black_v_std_min"]
        and wrc <= p["s1_black_white_center_max"]
    )
    if likely_black_ball:
        score += p["s1_black_score_bonus"]

    vc = "suspect" if score <= p["s1_suspect_score_thr"] else "valid_ball"
    return {"validity_class": vc, "validity_score": float(score)}

def apply_colour_rules(row, p):
    br = safe_float(row.get("black_ratio"))
    wr = safe_float(row.get("white_ratio"))
    cr = safe_float(row.get("colour_ratio"))
    hm = safe_float(row.get("h_median"))
    vm = safe_float(row.get("v_median"))
    sm = safe_float(row.get("s_median"))
    vp = safe_float(row.get("v_p90"), vm)
    wrc = safe_float(row.get("white_ratio_center"))
    bt = row.get("ball_type_hint")

    if br >= p["s2_black_black_ratio_min"] and wr <= p["s2_black_white_ratio_max"] and vm <= p["s2_black_v_median_max"]:
        return "black"

    if (
        bt == "stripe"
        and p["s2_yellow_white_ratio_lo"] <= wr <= p["s2_yellow_white_ratio_hi"]
        and br <= p["s2_yellow_black_ratio_max"]
        and cr >= p["s2_yellow_colour_ratio_min"]
        and hm <= p["s2_yellow_h_median_max"]
    ):
        return "yellow"

    if (
        wr >= p["s2_white_white_ratio_min"]
        and vp >= p["s2_white_v_p90_min"]
        and sm <= p["s2_white_s_median_max"]
        and br <= p["s2_white_black_ratio_max"]
        and wrc >= p["s2_white_center_min"]
        and wr >= wrc * p["s2_white_center_ratio"]
    ):
        return "white"

    if sm >= p["s2_chrom_s_min"] and cr >= p["s2_chrom_colour_ratio_min"]:
        if hm <= p["s2_red_h_max"] or hm >= p["s2_red_h_min_wrap"]:
            return "red"
        if p["s2_maroon_h_lo"] < hm <= p["s2_maroon_h_hi"] and vm <= p["s2_maroon_v_max"]:
            return "maroon"
        if p["s2_orange_h_lo"] < hm <= p["s2_orange_h_hi"] and vm > p["s2_orange_v_min"]:
            return "orange"

    return None

def predict_base_colour(row, Xz_train, y_train, feature_mean, feature_std, p, k=5):
    rule = apply_colour_rules(row, p)
    if rule is not None:
        return {"top1": rule, "ranked": [(rule, 1.0)], "confidence": 1.0}

    x = _feature_vec(row)
    xz = (x - feature_mean) / feature_std
    dists = np.linalg.norm(Xz_train - xz, axis=1)
    nn_idx = np.argsort(dists)[:k]

    votes = defaultdict(float)
    for idx in nn_idx:
        cls = y_train[idx]
        if cls in ("white", "black"):
            continue
        votes[cls] += 1.0 / max(dists[idx], 1e-6)

    if not votes:
        top_cls = y_train[nn_idx[0]]
        return {"top1": top_cls, "ranked": [(top_cls, 1.0)], "confidence": 1.0}

    total = sum(votes.values())
    ranked = sorted([(c, s / total) for c, s in votes.items()], key=lambda x: x[1], reverse=True)
    return {"top1": ranked[0][0], "ranked": ranked, "confidence": float(ranked[0][1])}

def predict_ball_type(row, colour, p):
    if colour == "white":
        return {"top1": "cue", "ranked": [("cue", 1.0)], "confidence": 1.0}
    if colour == "black":
        return {"top1": "black", "ranked": [("black", 1.0)], "confidence": 1.0}

    ss = safe_float(row.get("stripe_score"))
    wr = safe_float(row.get("white_ratio"))
    wrc = safe_float(row.get("white_ratio_center"))
    wrr = safe_float(row.get("white_ratio_ring"))
    hs = safe_float(row.get("h_std"))
    cr = safe_float(row.get("colour_ratio"))

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
    prob_solid = 1.0 - prob_stripe
    ranked = sorted([("stripe", prob_stripe), ("solid", prob_solid)], key=lambda x: x[1], reverse=True)
    return {"top1": ranked[0][0], "ranked": ranked, "confidence": float(ranked[0][1])}

def build_ranked_final(base_pred, type_pred):
    options = []
    for colour_cls, colour_prob in base_pred["ranked"]:
        if colour_cls in ("white", "black"):
            rtypes = [("cue", 1.0)] if colour_cls == "white" else [("black", 1.0)]
        elif colour_cls == base_pred["top1"]:
            rtypes = type_pred["ranked"]
        else:
            rtypes = [("solid", 0.5), ("stripe", 0.5)]

        for type_cls, type_prob in rtypes:
            options.append({
                "final_label": final_label_from_parts(colour_cls, type_cls),
                "base_colour": colour_cls,
                "ball_type": type_cls,
                "joint_prob": float(colour_prob) * float(type_prob),
            })

    dedup = {}
    for item in sorted(options, key=lambda d: d["joint_prob"], reverse=True):
        if item["final_label"] not in dedup:
            dedup[item["final_label"]] = item

    return list(dedup.values())

def combined_strength(row, p):
    penalty = p["s4_suspect_penalty"] if row["validity_class"] == "suspect" else 0.0
    return (
        p["s4_w_final_conf"] * float(row["pred_final_conf"])
        + p["s4_w_base_conf"] * float(row["pred_base_colour_conf"])
        + p["s4_w_type_conf"] * float(row["pred_ball_type_conf"])
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
        tier = 0 if row["validity_class"] == "valid_ball" else 1
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
        row["strength"] = combined_strength(row, p)
        row["final_global_label"] = row["pred_final_label"]
        row["final_global_conf"] = float(row["pred_final_conf"])
        row["global_resolution"] = "kept_top1"

    for _ in range(2):
        for req in REQUIRED_UNIQUE_LABELS:
            winner = choose_required(rows, req)
            if winner is None:
                continue
            widx, wprob = winner
            rows[widx]["final_global_label"] = req
            rows[widx]["final_global_conf"] = wprob
            rows[widx]["global_resolution"] = f"forced_{req}"

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

                items = sorted(items, key=lambda x: x[1]["strength"], reverse=True)
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
                        rows[idx]["final_global_conf"] = np_
                        rows[idx]["global_resolution"] = mode
                        changed = True

    for row in rows:
        if row["final_global_label"] in ("cue", "black"):
            continue

        if row["validity_class"] == "suspect" and row["strength"] < p["s4_weak_suspect_strength_thr"]:
            row["final_global_label"] = "not_ball"
            row["final_global_conf"] = 0.0
            row["global_resolution"] = "weak_suspect_demoted"
        elif row["validity_score"] <= p["s4_strong_invalid_score_thr"]:
            row["final_global_label"] = "not_ball"
            row["final_global_conf"] = 0.0
            row["global_resolution"] = "final_strong_invalid"

    return rows

def predict_single(row, Xz_train, y_train, feature_mean, feature_std, p):
    validity = predict_validity(row, p)
    base_pred = predict_base_colour(row, Xz_train, y_train, feature_mean, feature_std, p)
    type_pred = predict_ball_type(row, base_pred["top1"], p)
    ranked = build_ranked_final(base_pred, type_pred)

    out = dict(row)
    out.update(validity)
    out["pred_base_colour"] = base_pred["top1"]
    out["pred_base_colour_conf"] = float(base_pred["confidence"])
    out["pred_ball_type"] = type_pred["top1"]
    out["pred_ball_type_conf"] = float(type_pred["confidence"])
    out["pred_final_label"] = ranked[0]["final_label"] if ranked else "not_ball"
    out["pred_final_conf"] = ranked[0]["joint_prob"] if ranked else 0.0
    out["ranked_final_predictions"] = ranked
    return out

def run_on_images(image_list, annotations_by_image, Xz_train, y_train, feature_mean, feature_std, p):
    all_rows = []
    for filename in image_list:
        payload = annotations_by_image.get(filename, {})
        pred_rows = []

        for det in payload.get("detections", []):
            row = build_detection_row(det, filename)
            row["ball_type_hint"] = det.get("ball_type")
            pred_rows.append(predict_single(row, Xz_train, y_train, feature_mean, feature_std, p))

        all_rows.extend(resolve_global(pred_rows, p))

    return all_rows

# ═══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_objective(all_rows):
    valid_gt = [r for r in all_rows if r["label_gt"] in VALID_LABELS]
    invalid_gt = [r for r in all_rows if r["label_gt"] in INVALID_LABELS]

    # Stage-1 / validity side
    tp_valid = sum(1 for r in valid_gt if r["final_global_label"] != "not_ball")
    fp_valid = sum(1 for r in invalid_gt if r["final_global_label"] != "not_ball")

    recall_balls = tp_valid / max(len(valid_gt), 1)
    fp_rate = fp_valid / max(len(invalid_gt), 1)

    # Subtask metrics: only real valid balls
    colour_correct = sum(
        1 for r in valid_gt
        if label_to_base_colour(r["label_gt"]) == r["pred_base_colour"]
    )
    colour_acc = colour_correct / max(len(valid_gt), 1)

    type_rows = [
        r for r in valid_gt
        if label_to_ball_type(r["label_gt"]) in ("solid", "stripe")
    ]
    type_correct = sum(
        1 for r in type_rows
        if label_to_ball_type(r["label_gt"]) == r["pred_ball_type"]
    )
    type_acc = type_correct / max(len(type_rows), 1)

    # Final global metrics: all detections after Stage 4
    y_true_all = [gt_final_label(r["label_gt"]) for r in all_rows]
    y_pred_all = [r["final_global_label"] for r in all_rows]

    final_acc_all = sum(
        1 for yt, yp in zip(y_true_all, y_pred_all) if yt == yp
    ) / max(len(y_true_all), 1)

    labels_all = sorted(set(y_true_all) | set(y_pred_all))
    per_f1_all = []
    for lbl in labels_all:
        tp = sum(1 for t, p_ in zip(y_true_all, y_pred_all) if t == lbl and p_ == lbl)
        fp = sum(1 for t, p_ in zip(y_true_all, y_pred_all) if t != lbl and p_ == lbl)
        fn = sum(1 for t, p_ in zip(y_true_all, y_pred_all) if t == lbl and p_ != lbl)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_f1_all.append(2 * prec * rec / max(prec + rec, 1e-9))

    f1_macro_all = float(np.mean(per_f1_all)) if per_f1_all else 0.0

    score = (
        OBJ_WEIGHTS["validity_recall_balls"] * recall_balls
        + OBJ_WEIGHTS["validity_fp_rate"] * fp_rate
        + OBJ_WEIGHTS["colour_accuracy"] * colour_acc
        + OBJ_WEIGHTS["type_accuracy"] * type_acc
        + OBJ_WEIGHTS["final_f1_macro"] * f1_macro_all
    )

    return score, {
        "recall_balls": round(recall_balls, 4),
        "fp_rate": round(fp_rate, 4),
        "colour_acc_valid_only": round(colour_acc, 4),
        "type_acc_valid_only": round(type_acc, 4),
        "final_acc_all": round(final_acc_all, 4),
        "f1_macro_all": round(f1_macro_all, 4),
        "objective": round(score, 5),
    }

def compute_eval_metrics(all_rows):
    valid_gt = [r for r in all_rows if r["label_gt"] in VALID_LABELS]
    invalid_gt = [r for r in all_rows if r["label_gt"] in INVALID_LABELS]

    recall_balls = sum(
        1 for r in valid_gt if r["final_global_label"] != "not_ball"
    ) / max(len(valid_gt), 1)

    fp_rate = sum(
        1 for r in invalid_gt if r["final_global_label"] != "not_ball"
    ) / max(len(invalid_gt), 1)

    invalid_rejection_rate = sum(
        1 for r in invalid_gt if r["final_global_label"] == "not_ball"
    ) / max(len(invalid_gt), 1)

    colour_acc_valid_only = sum(
        1 for r in valid_gt
        if label_to_base_colour(r["label_gt"]) == r["pred_base_colour"]
    ) / max(len(valid_gt), 1)

    type_rows = [
        r for r in valid_gt
        if label_to_ball_type(r["label_gt"]) in ("solid", "stripe")
    ]
    type_acc_valid_only = sum(
        1 for r in type_rows
        if label_to_ball_type(r["label_gt"]) == r["pred_ball_type"]
    ) / max(len(type_rows), 1)

    final_label_acc_valid_only = sum(
        1 for r in valid_gt if r["final_global_label"] == r["label_gt"]
    ) / max(len(valid_gt), 1)

    y_true_valid = [r["label_gt"] for r in valid_gt]
    y_pred_valid = [r["final_global_label"] for r in valid_gt]
    labels_valid = sorted(set(y_true_valid) | set(y_pred_valid))

    per_f1_valid = []
    for lbl in labels_valid:
        tp = sum(1 for t, p_ in zip(y_true_valid, y_pred_valid) if t == lbl and p_ == lbl)
        fp = sum(1 for t, p_ in zip(y_true_valid, y_pred_valid) if t != lbl and p_ == lbl)
        fn = sum(1 for t, p_ in zip(y_true_valid, y_pred_valid) if t == lbl and p_ != lbl)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_f1_valid.append(2 * prec * rec / max(prec + rec, 1e-9))

    f1_macro_valid_only = float(np.mean(per_f1_valid)) if per_f1_valid else 0.0

    y_true_all = [gt_final_label(r["label_gt"]) for r in all_rows]
    y_pred_all = [r["final_global_label"] for r in all_rows]

    final_acc_all = sum(
        1 for yt, yp in zip(y_true_all, y_pred_all) if yt == yp
    ) / max(len(y_true_all), 1)

    labels_all = sorted(set(y_true_all) | set(y_pred_all))
    per_f1_all = []
    for lbl in labels_all:
        tp = sum(1 for t, p_ in zip(y_true_all, y_pred_all) if t == lbl and p_ == lbl)
        fp = sum(1 for t, p_ in zip(y_true_all, y_pred_all) if t != lbl and p_ == lbl)
        fn = sum(1 for t, p_ in zip(y_true_all, y_pred_all) if t == lbl and p_ != lbl)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_f1_all.append(2 * prec * rec / max(prec + rec, 1e-9))

    f1_macro_all = float(np.mean(per_f1_all)) if per_f1_all else 0.0

    return {
        "recall_balls": round(recall_balls, 4),
        "fp_rate": round(fp_rate, 4),
        "invalid_rejection_rate": round(invalid_rejection_rate, 4),
        "colour_acc_valid_only": round(colour_acc_valid_only, 4),
        "type_acc_valid_only": round(type_acc_valid_only, 4),
        "final_label_acc_valid_only": round(final_label_acc_valid_only, 4),
        "f1_macro_valid_only": round(f1_macro_valid_only, 4),
        "final_acc_all": round(final_acc_all, 4),
        "f1_macro_all": round(f1_macro_all, 4),
    }

def evaluate_multi_split(p, splits, annotations_by_image):
    scores = []

    for split in splits:
        train_images = split["train_images"]
        val_images = split["val_images"]

        Xz_train, y_train, feature_mean, feature_std = build_knn_from_images(
            annotations_by_image,
            train_images
        )

        rows = run_on_images(
            val_images,
            annotations_by_image,
            Xz_train,
            y_train,
            feature_mean,
            feature_std,
            p
        )
        score, _ = compute_objective(rows)
        scores.append(score)

    return float(np.mean(scores)), float(np.std(scores))

# ═══════════════════════════════════════════════════════════════════════════════
# CSV GROUND TRUTH EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

CSV_LABEL_MAP = {
    "0_white_cue": "cue",
    "1_yellow_solid": "yellow_solid",
    "2_blue_solid": "blue_solid",
    "3_red_solid": "red_solid",
    "4_purple_solid": "purple_solid",
    "5_orange_solid": "orange_solid",
    "6_green_solid": "green_solid",
    "7_maroon_solid": "maroon_solid",
    "8_black_8ball": "black",
    "9_yellow_stripe": "yellow_stripe",
    "10_blue_stripe": "blue_stripe",
    "11_red_stripe": "red_stripe",
    "12_purple_stripe": "purple_stripe",
    "13_orange_stripe": "orange_stripe",
    "14_green_stripe": "green_stripe",
    "15_maroon_stripe": "maroon_stripe",
}

def evaluate_against_csv(all_rows, csv_path):
    gt_data = load_ground_truth_csv(csv_path)
    preds_by_image = defaultdict(list)

    for r in all_rows:
        preds_by_image[r["filename"]].append(r["final_global_label"])

    total_tp, total_fp, total_fn = 0, 0, 0

    for filename, gt_row in gt_data.items():
        pred_counts = Counter(preds_by_image.get(filename, []))
        for csv_col, label in CSV_LABEL_MAP.items():
            gt_count = gt_row.get(csv_col, 0)
            pred_count = pred_counts.get(label, 0)
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# HILL CLIMBING
# ═══════════════════════════════════════════════════════════════════════════════

def clamp(name, value):
    lo, hi, step = SEARCH_SPACE[name]
    value = max(lo, min(hi, value))
    steps = round((value - lo) / step)
    return round(lo + steps * step, 8)

def get_neighbours(params, param_names):
    neighbours = []
    for name in param_names:
        if name not in SEARCH_SPACE:
            continue
        lo, hi, step = SEARCH_SPACE[name]
        cur = params[name]
        for delta in [-step, +step]:
            nv = clamp(name, cur + delta)
            if nv != cur:
                np_ = dict(params)
                np_[name] = nv
                neighbours.append((name, nv, np_))
    return neighbours

def perturb_params(params, param_names, frac=PERTURB_FRAC, seed=None):
    rng = random.Random(seed)
    new_p = dict(params)
    to_perturb = rng.sample(param_names, max(1, int(len(param_names) * frac)))

    for name in to_perturb:
        if name not in SEARCH_SPACE:
            continue
        lo, hi, step = SEARCH_SPACE[name]
        n_steps = round((hi - lo) / step)
        chosen = rng.randint(0, n_steps)
        new_p[name] = round(lo + chosen * step, 8)

    return new_p

def hill_climb_single(
    start_params,
    param_names,
    splits,
    annotations_by_image,
    max_iter=MAX_ITER,
    patience=PATIENCE,
    label="",
    perturb_frac=PERTURB_FRAC,
):
    current = dict(start_params)
    best_score, best_std = evaluate_multi_split(current, splits, annotations_by_image)
    best_params = dict(current)
    no_improve = 0

    for iteration in range(1, max_iter + 1):
        neighbours = get_neighbours(current, param_names)
        random.shuffle(neighbours)

        improved = False
        for name, new_val, candidate in neighbours:
            score, std = evaluate_multi_split(candidate, splits, annotations_by_image)
            if score > best_score:
                best_score = score
                best_std = std
                best_params = dict(candidate)
                current = dict(candidate)
                improved = True
                no_improve = 0
                print(
                    f"    {label} iter {iteration:4d} | {name}={new_val:.4f} | "
                    f"score={score:.5f} ± {std:.4f}"
                )
                break

        if not improved:
            no_improve += 1
        if no_improve >= patience:
            break

    return best_params, best_score, best_std

def hill_climb_stage_with_restarts(
    stage_id,
    param_names,
    start_params,
    splits,
    annotations_by_image,
    n_restarts=N_RESTARTS,
    max_iter=MAX_ITER,
    patience=PATIENCE,
    perturb_frac=PERTURB_FRAC,
):
    print(f"\n{'=' * 65}")
    print(f"STAGE {stage_id} | {len(param_names)} params | {n_restarts} restarts")
    print(f"{'=' * 65}")

    best_params, best_score, best_std = hill_climb_single(
        start_params,
        param_names,
        splits,
        annotations_by_image,
        max_iter=max_iter,
        patience=patience,
        label=f"S{stage_id}-R0",
        perturb_frac=perturb_frac,
    )
    print(f"  Restart 0 done | best={best_score:.5f} ± {best_std:.4f}")

    for restart in range(1, n_restarts):
        perturbed = perturb_params(
            best_params,
            param_names,
            frac=perturb_frac,
            seed=restart * 1000 + stage_id
        )
        p, s, sd = hill_climb_single(
            perturbed,
            param_names,
            splits,
            annotations_by_image,
            max_iter=max_iter,
            patience=patience,
            label=f"S{stage_id}-R{restart}",
            perturb_frac=perturb_frac,
        )
        print(f"  Restart {restart} done | score={s:.5f} ± {sd:.4f}")
        if s > best_score:
            best_score = s
            best_std = sd
            best_params = dict(p)
            print(f"  *** New best: {best_score:.5f} ***")

    print(f"Stage {stage_id} final best: {best_score:.5f} ± {best_std:.4f}")
    return best_params, best_score

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("Loading annotations...")
    annotations_by_image = load_annotations(ANNOTATIONS_JSON)

    print(f"\nBuilding {N_SPLITS} val splits + fixed test set...")
    splits, test_images = make_splits(annotations_by_image)
    print(f"  Val splits: {[len(s['val_images']) for s in splits]} images each")
    print(f"  Train splits: {[len(s['train_images']) for s in splits]} images each")
    print(f"  Test images: {len(test_images)}")

    print("\nBaseline (DEFAULT params, multi-split)...")
    baseline_score, baseline_std = evaluate_multi_split(
        DEFAULT_PARAMS,
        splits,
        annotations_by_image
    )
    print(f"  Baseline: {baseline_score:.5f} ± {baseline_std:.4f}")

    optimized = dict(DEFAULT_PARAMS)

    # Stage 1: stronger search, same global objective
    optimized, _ = hill_climb_stage_with_restarts(
        stage_id=1,
        param_names=STAGE_PARAMS[1],
        start_params=optimized,
        splits=splits,
        annotations_by_image=annotations_by_image,
        n_restarts=STAGE1_N_RESTARTS,
        max_iter=STAGE1_MAX_ITER,
        patience=STAGE1_PATIENCE,
        perturb_frac=STAGE1_PERTURB_FRAC,
    )

    # Remaining stages: standard search
    for stage_id in [2, 3, 4]:
        optimized, _ = hill_climb_stage_with_restarts(
            stage_id=stage_id,
            param_names=STAGE_PARAMS[stage_id],
            start_params=optimized,
            splits=splits,
            annotations_by_image=annotations_by_image,
            n_restarts=N_RESTARTS,
            max_iter=MAX_ITER,
            patience=PATIENCE,
            perturb_frac=PERTURB_FRAC,
        )

    print(f"\n{'#' * 65}")
    print("JOINT FINE-TUNING (all params)")
    print(f"{'#' * 65}")
    all_param_names = list(SEARCH_SPACE.keys())
    optimized, joint_score = hill_climb_stage_with_restarts(
        stage_id=0,
        param_names=all_param_names,
        start_params=optimized,
        splits=splits,
        annotations_by_image=annotations_by_image,
        n_restarts=N_RESTARTS + 1,
        max_iter=MAX_ITER,
        patience=PATIENCE,
        perturb_frac=PERTURB_FRAC,
    )

    print(f"\n{'=' * 65}")
    print("TEST SET EVALUATION (JSON, held-out)")
    print(f"{'=' * 65}")

    train_for_test = sorted(set(annotations_by_image.keys()) - set(test_images))
    print(f"Building KNN for held-out test using {len(train_for_test)} non-test images...")
    Xz_test_train, y_test_train, mean_test_train, std_test_train = build_knn_from_images(
        annotations_by_image,
        train_for_test
    )
    print(f"  KNN training samples for test: {len(y_test_train)}")

    test_rows_opt = run_on_images(
        test_images,
        annotations_by_image,
        Xz_test_train,
        y_test_train,
        mean_test_train,
        std_test_train,
        optimized
    )
    test_rows_def = run_on_images(
        test_images,
        annotations_by_image,
        Xz_test_train,
        y_test_train,
        mean_test_train,
        std_test_train,
        DEFAULT_PARAMS
    )

    test_score_opt, _ = compute_objective(test_rows_opt)
    test_score_def, _ = compute_objective(test_rows_def)

    test_metrics_opt = compute_eval_metrics(test_rows_opt)
    test_metrics_def = compute_eval_metrics(test_rows_def)

    baseline_rows_preview = []
    for split in splits[:1]:
        Xz_b, y_b, mean_b, std_b = build_knn_from_images(
            annotations_by_image,
            split["train_images"]
        )
        baseline_rows_preview = run_on_images(
            split["val_images"],
            annotations_by_image,
            Xz_b,
            y_b,
            mean_b,
            std_b,
            DEFAULT_PARAMS
        )
        break

    baseline_preview_metrics = compute_eval_metrics(baseline_rows_preview)
    print(f"  Baseline preview metrics: {baseline_preview_metrics}")

    print(f"  Optimized: {test_score_opt:.5f}  {test_metrics_opt}")
    print(f"  Default:   {test_score_def:.5f}  {test_metrics_def}")
    print(f"  Improvement: {test_score_opt - test_score_def:+.5f}")

    if GROUND_TRUTH_CSV.exists():
        print(f"\n{'=' * 65}")
        print("CSV GROUND TRUTH EVALUATION (all images, diagnostic only)")
        print(f"{'=' * 65}")

        all_images = list(annotations_by_image.keys())
        Xz_all, y_all, mean_all, std_all = build_knn_from_images(
            annotations_by_image,
            all_images
        )

        all_rows_opt = run_on_images(
            all_images,
            annotations_by_image,
            Xz_all,
            y_all,
            mean_all,
            std_all,
            optimized
        )
        all_rows_def = run_on_images(
            all_images,
            annotations_by_image,
            Xz_all,
            y_all,
            mean_all,
            std_all,
            DEFAULT_PARAMS
        )

        csv_opt = evaluate_against_csv(all_rows_opt, GROUND_TRUTH_CSV)
        csv_def = evaluate_against_csv(all_rows_def, GROUND_TRUTH_CSV)

        print(f"  Optimized: P={csv_opt['precision']:.4f}  R={csv_opt['recall']:.4f}  F1={csv_opt['f1']:.4f}")
        print(f"  Default:   P={csv_def['precision']:.4f}  R={csv_def['recall']:.4f}  F1={csv_def['f1']:.4f}")
        print(f"  F1 improvement: {csv_opt['f1'] - csv_def['f1']:+.4f}")

    changes = {
        k: {"default": DEFAULT_PARAMS[k], "optimized": optimized[k]}
        for k in optimized
        if abs(optimized[k] - DEFAULT_PARAMS[k]) > 1e-9
    }

    print(f"\n{'=' * 65}")
    print(f"PARAMETER CHANGES ({len(changes)} changed):")
    print(f"{'=' * 65}")
    if changes:
        for k, v in sorted(changes.items()):
            direction = "↑" if v["optimized"] > v["default"] else "↓"
            print(f"  {direction} {k:48s} {v['default']:>9.4f} → {v['optimized']:>9.4f}")
    else:
        print("  No parameters changed.")

    results = {
        "baseline_val_score": baseline_score,
        "baseline_val_std": baseline_std,
        "optimized_val_score": joint_score,
        "test_score_optimized": test_score_opt,
        "test_score_default": test_score_def,
        "test_metrics_opt": test_metrics_opt,
        "test_metrics_def": test_metrics_def,
        "optimized_params": optimized,
        "param_changes": changes,
        "stage1_search_config": {
            "n_restarts": STAGE1_N_RESTARTS,
            "max_iter": STAGE1_MAX_ITER,
            "patience": STAGE1_PATIENCE,
            "perturb_frac": STAGE1_PERTURB_FRAC,
        },
        "default_search_config": {
            "n_restarts": N_RESTARTS,
            "max_iter": MAX_ITER,
            "patience": PATIENCE,
            "perturb_frac": PERTURB_FRAC,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out_path = Path("hill_climbing_results_v3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Total time: {results['elapsed_seconds']}s")

    return optimized, results

if __name__ == "__main__":
    main()

# %%
def expand_corners(corners, expand_px=20):
    """
    Push each corner away from the quadrilateral center.
    """
    corners = np.array(corners, dtype=np.float32)
    center = corners.mean(axis=0)

    expanded = []
    for pt in corners:
        direction = pt - center
        norm = np.linalg.norm(direction) + 1e-9
        direction = direction / norm
        expanded.append(pt + direction * expand_px)

    return np.array(expanded, dtype=np.float32)



def draw_table_outline(img, corners, color=(0, 255, 0), thickness=3):
    """
    Draw the detected table quadrilateral on the original image.
    """
    vis = img.copy()
    corners_int = corners.astype(int)

    # draw polygon
    for i in range(4):
        p1 = tuple(corners_int[i])
        p2 = tuple(corners_int[(i + 1) % 4])
        cv2.line(vis, p1, p2, color, thickness)

    # draw corner points and labels
    names = ["TL", "TR", "BR", "BL"]
    for name, pt in zip(names, corners_int):
        cv2.circle(vis, tuple(pt), 7, (0, 0, 255), -1)
        cv2.putText(
            vis,
            name,
            (pt[0] + 8, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return vis

def mask_from_corners(img_shape, corners):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
    return mask


def get_top_view(img, corners, expand_px=20, table_ratio=None):
    corners_exp = expand_corners(corners, expand_px=expand_px)
    tl, tr, br, bl = corners_exp

    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    W = int(max(width_top, width_bottom))
    
    if table_ratio is not None:
        # Force a specific ratio (e.g. 2.0 for standard pool table)
        H = int(W * table_ratio)
    else:
        # Use the ACTUAL height from the detected corners
        H = int(max(height_left, height_right))

    dst = np.array([
        [0,     0    ],
        [W - 1, 0    ],
        [W - 1, H - 1],
        [0,     H - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners_exp, dst)
    warped = cv2.warpPerspective(img, M, (W, H))

    h, w = warped.shape[:2]
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped, M, corners_exp



for path in image_paths[:15]:
    img = cv2.imread(path)
    mask, corners = segment_table(img)

    corners = order_corners(corners)
    warped, _, corners_used = get_top_view(img, corners, expand_px=20, table_ratio=2.0)

    outlined = draw_table_outline(img, corners_used)
    used_mask = mask_from_corners(img.shape, corners_used)
    masked_img = cv2.bitwise_and(img, img, mask=used_mask)

    show(
        [outlined, warped],
        ["Detected table","Top View"],
        figsize=(24, 8)
    )



