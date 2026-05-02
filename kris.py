"""
================================================================
AI-Driven Urban Road Defect Detection and Analytics Dashboard
app.py | Clean & Production-Ready Version
================================================================
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from severity import (
    classify_severity,
    get_severity_color_cv2,
)

# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="Road Defect Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# CONSTANTS & COLORS
# ================================================================
DB_PATH = "road_defects.db"
SEV_COLORS = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}

# ================================================================
# DATABASE
# ================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name   TEXT    NOT NULL,
            total        INTEGER DEFAULT 0,
            low_count    INTEGER DEFAULT 0,
            medium_count INTEGER DEFAULT 0,
            high_count   INTEGER DEFAULT 0,
            timestamp    TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def db_save(name, total, low, med, high):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO detections(image_name,total,low_count,medium_count,high_count,timestamp) VALUES(?,?,?,?,?,?)",
        (name, total, low, med, high, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()

def db_fetch():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)
    conn.close()
    return df

def db_clear():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM detections")
    conn.commit()
    conn.close()

def db_summary(df):
    if df.empty:
        return 0, 0, 0, 0, 0
    return (
        len(df),
        int(df["total"].sum()),
        int(df["low_count"].sum()),
        int(df["medium_count"].sum()),
        int(df["high_count"].sum()),
    )

init_db()

# ================================================================
# MODEL LOADING
# ================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists("best.pt"):
        st.error("❌  best.pt not found — place it in the same folder as app.py")
        st.stop()
    return YOLO("best.pt")

with st.spinner("⚡ Loading YOLOv8 model…"):
    model = load_model()

# ================================================================
# DETECTION PIPELINE
# ================================================================
def run_detection(pil_image, conf, draw_labels):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]
    canvas = img_bgr.copy()
    results = model.predict(source=img_rgb, conf=conf, verbose=False)
    boxes = results[0].boxes
    detections = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = round(float(box.conf[0]), 3)
        sev, pct, area = classify_severity(x1, y1, x2, y2, W, H)
        color_bgr = get_severity_color_cv2(sev)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)
        
        if draw_labels:
            label = f"{sev} {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            ly1 = max(y1 - th - 10, 0)
            cv2.rectangle(canvas, (x1, ly1), (x1 + tw + 8, y1), color_bgr, -1)
            cv2.putText(canvas, label, (x1 + 4, max(y1 - 4, th + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (10, 10, 10), 1, cv2.LINE_AA)
                        
        detections.append({
            "severity": sev, "confidence": confidence,
            "area_px": area, "ratio_pct": pct,
        })
        
    annotated_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections

# ================================================================
# CHART HELPERS 
# ================================================================
@st.cache_data
def chart_advanced_trend(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center")
    else:
        x = range(len(df))
        ax.plot(x, df["total"], color="#3b82f6", linewidth=2, marker="o", markersize=6, label="Total Intensity")
        ax.plot(x, df["high_count"], color=SEV_COLORS["High"], linewidth=2, linestyle="--", label="Critical Level")
        ax.set_xticks(x[::max(1, len(df)//5)])
        ax.set_xticklabels([str(ts)[:10] for ts in df["timestamp"]][::max(1, len(df)//5)])
    ax.set_title("Detection Trend Over Time", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

@st.cache_data
def chart_premium_donut(counts):
    fig, ax = plt.subplots(figsize=(6, 6))
    nz = {k: v for k, v in counts.items() if v > 0}
    if not nz:
        ax.text(0.5, 0.5, "Surface Intact", ha="center")
    else:
        colors = [SEV_COLORS[k] for k in nz.keys()]
        wedges, texts, autotexts = ax.pie(nz.values(), labels=nz.keys(), colors=colors, 
                                          autopct='%1.1f%%', startangle=140, pctdistance=0.85,
                                          wedgeprops={'width': 0.4})
        ax.text(0, 0, f"{sum(nz.values())}\nTotal", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title("Severity Composition", fontweight="bold")
    return fig

@st.cache_data
def chart_stacked_bar(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "Awaiting scans...", ha="center")
    else:
        indices = range(len(df))
        ax.bar(indices, df["low_count"], label="Low", color=SEV_COLORS["Low"], alpha=0.9)
        ax.bar(indices, df["medium_count"], bottom=df["low_count"], label="Med", color=SEV_COLORS["Medium"], alpha=0.9)
        ax.bar(indices, df["high_count"], bottom=df["low_count"]+df["medium_count"], label="High", color=SEV_COLORS["High"], alpha=0.9)
        ax.set_xticks(indices)
        ax.set_xticklabels([n[:12]+".." if len(n)>12 else n for n in df["image_name"]], rotation=30, ha="right")
    ax.set_title("Historical Severity Breakdown", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

@st.cache_data
def get_top_k(df, k):
    return df.nlargest(k, "total") if not df.empty else pd.DataFrame()

def chart_top_k(df, k=5):
    top = get_top_k(df, k)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if top.empty:
        ax.text(0.5, 0.5, "No data available", ha="center")
    else:
        names = [n[:15]+".." if len(n)>15 else n for n in top["image_name"]]
        ax.barh(names, top["total"], color="#f59e0b", alpha=0.8, height=0.6)
        ax.invert_yaxis()

    ax.set_title(f"Top {k} Scans with Most Defects", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig

# ================================================================
# UI HELPERS
# ================================================================
def render_stat_cards(n_images, total, low, med, high):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Scanned Images", n_images)
    col2.metric("Total Detections", total)
    col3.metric("Low Severity", low)
    col4.metric("Med. Severity", med)
    col5.metric("High Severity", high)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.title("🛣️ RoadScan AI")
    st.caption("MIET JAMMU · CSE 2026")
    st.divider()

    st.subheader("Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.40, 0.05,
                            help="Minimum confidence for a detection to be shown")
    draw_labels = st.toggle("Draw Labels on Image", value=True)

    st.divider()
    st.subheader("History Controls")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Refresh", use_container_width=True):
            pass # Streamlit handles the rerun
    with col_b:
        if st.button("🗑️ Clear DB", use_container_width=True):
            db_clear()
            st.success("Database cleared.")

    st.divider()
    st.subheader("Data Export")
    df_export = db_fetch()
    if not df_export.empty:
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=convert_df_to_csv(df_export),
            file_name=f"road_scan_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("No data to export.")

    st.divider()
    st.caption("YOLOv8 · OpenCV · Streamlit · SQLite")

# ================================================================
# MAIN LAYOUT
# ================================================================
st.title("AI-Driven Urban Road Defect Detection")
st.markdown("##### YOLOv8 Analytics Dashboard")
st.write("") # Spacing

# Tabs
tab_detect, tab_history, tab_analytics = st.tabs([
    "🔍 DETECTION", "📋 HISTORY", "📊 ANALYTICS"
])

# ──────────────────────────────────────────────
# TAB 1 — DETECTION
# ──────────────────────────────────────────────
with tab_detect:
    df_hist = db_fetch()
    n_img, tot, lo, me, hi = db_summary(df_hist)
    render_stat_cards(n_img, tot, lo, me, hi)
    st.divider()

    st.subheader("Upload Road Images")
    uploaded_files = st.file_uploader(
        "Drop one or more road images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.info("📷 Upload one or more road images above to begin AI detection.")
    else:
        if st.button(f"⚡ Run Detection on {len(uploaded_files)} Image(s)", type="primary"):
            all_detections = []
            batch_stats = []

            for idx, uf in enumerate(uploaded_files):
                pil_img = Image.open(uf)
                fname = uf.name

                with st.spinner(f"Analyzing {fname}…"):
                    annotated, detections = run_detection(pil_img, conf_thresh, draw_labels)

                all_detections.extend(detections)
                c_low  = sum(1 for d in detections if d["severity"] == "Low")
                c_med  = sum(1 for d in detections if d["severity"] == "Medium")
                c_high = sum(1 for d in detections if d["severity"] == "High")
                
                db_save(fname, len(detections), c_low, c_med, c_high)
                batch_stats.append({
                    "image_name": fname, "low_count": c_low, 
                    "medium_count": c_med, "high_count": c_high
                })

                st.subheader(f"📸 {fname} - {len(detections)} defect(s)")

                col_orig, col_ann = st.columns(2)
                with col_orig:
                    st.image(pil_img, caption="Original", use_container_width=True)
                with col_ann:
                    st.image(annotated, caption="Annotated", use_container_width=True)

                # Native Severity Alerts
                if not detections:
                    st.success("✅ **No Defects Detected** — Road surface appears intact.")
                elif c_high > 0:
                    st.error(f"🚨 **Critical** — {c_high} high-severity defect(s) detected. Immediate action required.")
                elif c_med > 0:
                    st.warning(f"⚠️ **Moderate** — {c_med} medium-severity defect(s) found. Schedule repair soon.")
                else:
                    st.info(f"🔵 **Minor** — {c_low} low-severity defect(s) noted. Monitor condition.")

                if detections:
                    with st.expander(f"📋 Detailed Results — {fname}", expanded=(len(uploaded_files) == 1)):
                        tc1, tc2 = st.columns([2, 1])
                        with tc1:
                            det_df = pd.DataFrame(detections)
                            display_df = det_df.copy()
                            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
                            display_df["ratio_pct"] = display_df["ratio_pct"].apply(lambda x: f"{x:.2f}%")
                            display_df["area_px"] = display_df["area_px"].apply(lambda x: f"{x:,} px²")
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                        with tc2:
                            counts = {"Low": c_low, "Medium": c_med, "High": c_high}
                            st.pyplot(chart_premium_donut(counts), use_container_width=True)

                    st.download_button(
                        label=f"📥 Export {fname} Results as CSV",
                        data=convert_df_to_csv(det_df),
                        file_name=f"detections_{fname.rsplit('.', 1)[0]}.csv",
                        mime="text/csv",
                        key=f"csv_{idx}_{fname}",
                    )
                st.divider()

            # Batch Session Analytics
            if len(uploaded_files) > 1 and all_detections:
                st.subheader("Batch Session Analytics")
                
                s_low = sum(1 for d in all_detections if d["severity"] == "Low")
                s_med = sum(1 for d in all_detections if d["severity"] == "Medium")
                s_high = sum(1 for d in all_detections if d["severity"] == "High")
                
                render_stat_cards(len(uploaded_files), len(all_detections), s_low, s_med, s_high)

                sc1, sc2 = st.columns(2)
                with sc1:
                    s_counts = {"Low": s_low, "Medium": s_med, "High": s_high}
                    st.pyplot(chart_premium_donut(s_counts), use_container_width=True)
                with sc2:
                    st.pyplot(chart_stacked_bar(pd.DataFrame(batch_stats)), use_container_width=True)

                st.download_button(
                    label="📥 Export Batch Results as CSV",
                    data=convert_df_to_csv(pd.DataFrame(all_detections)),
                    file_name=f"batch_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="csv_batch",
                )

# ──────────────────────────────────────────────
# TAB 2 — HISTORY
# ──────────────────────────────────────────────
with tab_history:
    df_hist = db_fetch()
    n_img, tot, lo, me, hi = db_summary(df_hist)
    render_stat_cards(n_img, tot, lo, me, hi)
    st.divider()

    st.subheader("Detection Log")

    if df_hist.empty:
        st.info("📋 No records yet. Run detection on some images to populate this log.")
    else:
        st.download_button(
            label="📥 Export Full History as CSV",
            data=convert_df_to_csv(df_hist),
            file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="csv_history_tab",
        )
        
        def get_dominant(row):
            m = max(row["low_count"], row["medium_count"], row["high_count"])
            if m == row["high_count"] and m > 0: return "🔴 High"
            if m == row["medium_count"] and m > 0: return "🟡 Medium"
            if m == 0: return "None"
            return "🟢 Low"

        df_display = df_hist.copy()
        df_display["Dominant Severity"] = df_display.apply(get_dominant, axis=1)
        
        st.dataframe(
            df_display[["id", "image_name", "total", "low_count", "medium_count", "high_count", "Dominant Severity", "timestamp"]],
            use_container_width=True,
            hide_index=True
        )

# ──────────────────────────────────────────────
# TAB 3 — ANALYTICS
# ──────────────────────────────────────────────
with tab_analytics:
    df_hist = db_fetch()
    n_img, tot, lo, me, hi = db_summary(df_hist)
    render_stat_cards(n_img, tot, lo, me, hi)
    st.divider()

    st.subheader("Visual Analytics")

    if df_hist.empty:
        st.info("📊 No data to visualize. Run detection on road images to populate charts.")
    else:
        st.download_button(
            label="📥 Export Analytics Data as CSV",
            data=convert_df_to_csv(df_hist),
            file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="csv_analytics_tab",
        )

        st.write("**Detection Trend & Per-Scan Breakdown**")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.pyplot(chart_advanced_trend(df_hist[::-1].reset_index(drop=True)), use_container_width=True)
        with r1c2:
            st.pyplot(chart_stacked_bar(df_hist[::-1].reset_index(drop=True)), use_container_width=True)

        st.write("**Severity Distribution & Top Analysis**")
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            counts = {"Low": lo, "Medium": me, "High": hi}
            st.pyplot(chart_premium_donut(counts), use_container_width=True)
        with r2c2:
            st.pyplot(chart_top_k(df_hist), use_container_width=True)

        st.write("**Top Affected Scans**")
        top5 = df_hist.nlargest(5, "high_count")[
            ["image_name", "total", "high_count", "medium_count", "low_count", "timestamp"]
        ]
        st.dataframe(
            top5.rename(columns={
                "image_name": "Image", "total": "Total",
                "high_count": "🔴 High", "medium_count": "🟡 Medium",
                "low_count": "🟢 Low", "timestamp": "Time",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        avg_per_scan = round(tot / max(n_img, 1), 1)
        high_rate = round(hi / max(tot, 1) * 100, 1)
        worst_scan = df_hist.loc[df_hist["high_count"].idxmax(), "image_name"] if hi > 0 else "—"

        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Avg Defects / Scan", avg_per_scan)
        km2.metric("High Severity Rate", f"{high_rate}%")
        km3.metric("Total Scans", n_img)
        km4.metric("Worst Scan", worst_scan[:20] + "…" if len(str(worst_scan)) > 20 else worst_scan)
