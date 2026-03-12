# UNABLE — Known Limitations & Setbacks

This document lists everything that **cannot be fully achieved** with the current information, data, or approach, along with known setbacks and caveats.

---

## 1. Exact JSON Input/Output Schema

**Status:** ✅ RESOLVED — Example JSON found in `example_json/output_example.json`

The output format uses normalized bounding box coordinates (0–1 range):
```json
[
  {
    "image_path": "development_set/image.jpg",
    "num_balls": 15,
    "balls": [
      {"number": 13, "xmin": 0.397, "xmax": 0.416, "ymin": 0.257, "ymax": 0.288}
    ]
  }
]
```

**Note:** No **input** JSON example was provided. The pipeline currently reads images from a directory. If a specific input JSON pointing to image paths is required, this would need to be adapted.

---

## 2. Solid vs Stripe Distinction

**Status:** ⚠️ PARTIAL — Heuristic-based, not guaranteed

Pool balls #1–7 are **solids** (single color) and #9–15 are **stripes** (color + white band). Distinguishing them requires detecting the presence/absence of a white stripe band.

- **Current approach:** We compute the ratio of white pixels to colored pixels in each ball ROI. Stripes should have a higher white ratio (~25%+).
- **Limitation:** Under poor lighting, shadows, reflections, or at small scales, the white ratio can be unreliable. Similarly-colored balls (e.g., #3 red solid vs #11 red stripe) may be misclassified.
- **Mitigation ideas:**
  - Use the number printed on each ball (OCR) — but this is extremely difficult at low resolution
  - Use texture analysis to detect the stripe pattern
  - Use a reference lookup based on known ball sizes in the image

---

## 3. Color Confusion Between Similar Balls

**Status:** ⚠️ KNOWN LIMITATION

Several ball pairs have very similar colors:
- **Red (#3/#11) vs Maroon (#7/#15):** Under warm lighting, these are nearly indistinguishable
- **Yellow (#1/#9) vs Orange (#5/#13):** Overlap in the hue range ~15–30
- **Purple (#4/#12) vs Blue (#2/#10):** Purple-blue boundary is ambiguous

**Mitigation:** The HSV ranges in the code are tuned for the dataset but may need adjustment for the 10 evaluation images.

---

## 4. Occluded or Partially Visible Balls

**Status:** ⚠️ KNOWN LIMITATION

- Balls partially hidden behind other balls or near pockets may not be fully circular
- `HoughCircles` requires a mostly-complete circle to detect — partial occlusions can cause missed detections
- Balls touching the table cushion may be partially cut off

**Mitigation:** Lowering `param2` in `HoughCircles` increases sensitivity but also increases false positives.

---

## 5. Table Corner Detection Accuracy

**Status:** ⚠️ VARIES BY IMAGE

The top-view transformation depends on accurate corner detection. Issues:
- **Pockets** can distort the contour shape, making polygon approximation give >4 points
- **Shadows** or **rails** may be included in the table mask
- **Camera angles** that are too oblique can cause poor perspective correction

**Fallback:** If `approxPolyDP` doesn't return exactly 4 points, we fall back to `minAreaRect` (bounding rotated rectangle), which is less accurate but more robust.

---

## 6. Evaluation on Unseen Images

**Status:** ℹ️ INHERENT RISK

The pipeline is tuned on the 50 development images. The 10 evaluation images may have:
- Different lighting conditions
- Different camera angles
- Different table cloth colors (not all tables are the same blue)
- Different image resolutions

**Mitigation:** Parameters should be kept as general as possible. The HSV exploration in Section 1 helps identify the range of conditions in the dataset.

---

## 7. Green Ball vs Table Felt Confusion

**Status:** ⚠️ KNOWN LIMITATION

The **green ball (#6/#14)** has a hue very close to the **table felt** (both are in the green/teal range). This can cause:
- The green ball being masked out during table detection
- Difficulty isolating the green ball's color from the background

**Mitigation:** Ball detection uses shape (circles), not color, so the ball should still be found. But color classification may struggle.

---

## 8. Single-File Final Script

**Status:** ℹ️ NOT YET DONE

The guidelines require a **single Python file** as the final deliverable. The current notebook is a development/exploration tool. The notebook functions would need to be extracted into a standalone `main.py` script.

---

## 9. Report

**Status:** ℹ️ NOT YET DONE

A **2-page maximum report** presenting methodology and results is required but has not been generated. The notebook itself contains extensive documentation that can serve as a basis.

---

## 10. Number Printed on Ball (OCR Approach)

**Status:** ❌ NOT FEASIBLE WITH ALLOWED LIBRARIES

Reading the number printed on each ball would be the most reliable identification method, but:
- The numbers are very small in most images
- OCR requires either `tesseract` / `pytesseract` (not in allowed libraries) or a trained model
- OpenCV's template matching could theoretically work but would require templates for each number at various scales and rotations

**Decision:** We rely on color-based identification as specified in the guidelines ("Ball numbers may be inferred based on colour").
