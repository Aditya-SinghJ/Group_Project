"""
severity.py — Advanced Severity Classification for Road Defect Detection
Place this in the same folder as app.py
"""

def classify_severity(x1: int, y1: int, x2: int, y2: int,
                      img_w: int, img_h: int,
                      label: str = "pothole"):
    """
    Classify pothole severity based on bounding-box area
    relative to image size.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        Bounding box coordinates
    img_w, img_h : int
        Image dimensions
    label : str
        Detected object label (default: pothole)

    Returns
    -------
    severity  : str   — "Low", "Medium", "High", or "Ignore"
    ratio_pct : float — bbox area as % of image area
    area_px   : int   — bbox area in pixels
    """

    # ── Step 1: Ignore non-potholes (optional but useful)
    if label.lower() != "pothole":
        return "Ignore", 0.0, 0

    # ── Step 2: Validate bounding box
    if x2 < x1 or y2 < y1:
        print("⚠ Warning: Invalid bounding box detected")
        return "Low", 0.0, 0

    # ── Step 3: Safe area calculation
    width  = max(0, x2 - x1)
    height = max(0, y2 - y1)
    area_px = width * height

    # ── Step 4: Prevent divide-by-zero
    img_area = max(img_w * img_h, 1)

    # ── Step 5: Area ratio (%)
    ratio_pct = round((area_px / img_area) * 100, 4)

    # ── Step 6: Improved thresholds (more realistic)
    if ratio_pct < 0.5:
        severity = "Low"
    elif ratio_pct < 3.0:
        severity = "Medium"
    else:
        severity = "High"

    return severity, ratio_pct, area_px


# ─────────────────────────────────────────────────────────────

def get_severity_color_cv2(severity: str):
    """Return BGR colour tuple for OpenCV drawing."""
    return {
        "Low":    (34, 197, 94),     # green
        "Medium": (30, 150, 255),    # orange
        "High":   (0, 0, 220),       # red
        "Ignore": (200, 200, 200),   # gray
    }.get(severity, (200, 200, 200))


def get_severity_color_hex(severity: str):
    """Return hex colour string for HTML/CSS."""
    return {
        "Low":    "#22c55e",
        "Medium": "#f97316",
        "High":   "#dc2626",
        "Ignore": "#aaaaaa",
    }.get(severity, "#aaaaaa")


def get_severity_emoji(severity: str):
    """Return emoji for severity level."""
    return {
        "Low":    "🟢",
        "Medium": "🟠",
        "High":   "🔴",
        "Ignore": "⚪",
    }.get(severity, "⚪")


# ── Extra: Severity Score (0–100) for analytics/dashboard ─────

def get_severity_score(ratio_pct: float):
    """
    Convert ratio percentage into a score (0–100).
    Useful for graphs, analytics, dashboards.
    """
    score = min(int(ratio_pct * 10), 100)
    return score


# ── Quick self-test ──────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (10, 10, 30, 25, 1280, 720, "pothole"),   # Low
        (100, 100, 350, 240, 1280, 720, "pothole"),  # Medium
        (50, 50, 700, 500, 1280, 720, "pothole"),    # High
        (100, 100, 50, 50, 1280, 720, "pothole"),    # Invalid
        (10, 10, 200, 200, 1280, 720, "crack"),      # Ignore
    ]

    print("=" * 50)
    print("SEVERITY CLASSIFICATION TEST")
    print("=" * 50)

    for x1, y1, x2, y2, w, h, label in tests:
        sev, pct, area = classify_severity(x1, y1, x2, y2, w, h, label)
        score = get_severity_score(pct)

        print(f"Box: ({x1},{y1},{x2},{y2}) | Label: {label}")
        print(f"→ Severity: {sev} {get_severity_emoji(sev)}")
        print(f"→ Area: {area}px | Ratio: {pct}% | Score: {score}")
        print("-" * 50)