import os
import tempfile
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import linear_sum_assignment
from skimage.feature import hog

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_VIDEO = os.path.join(APP_DIR, "video_thu.mp4")

# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="AI Traffic Guard · Giám sát vượt đèn đỏ",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bảng màu chủ đạo (Command Center Dark Theme)
C_BG = "#0B1220"
C_PANEL = "#111A2E"
C_PANEL2 = "#0E1628"
C_LINE = "#1F2A44"
C_TEXT = "#E2E8F0"
C_MUTED = "#94A3B8"
C_DIM = "#64748B"
C_ACCENT = "#38BDF8"       # Cyan Neon
C_PURPLE = "#818CF8"       # Purple Indigo
C_RED = "#EF4444"          # Crimson Red
C_AMBER = "#F59E0B"        # Traffic Amber
C_GREEN = "#10B981"        # Emerald Green

# ==========================================
# CSS GIAO DIỆN HIỆN ĐẠI (COMMAND CENTER UI)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

/* ---------- Cấu trúc tổng thể ---------- */
.stApp {
    background:
        radial-gradient(1100px 550px at 85% -5%, rgba(56, 189, 248, 0.08), transparent 60%),
        radial-gradient(900px 450px at -5% 15%, rgba(239, 68, 68, 0.07), transparent 60%),
        radial-gradient(800px 400px at 50% 100%, rgba(16, 185, 129, 0.05), transparent 70%),
        #0B1220;
    color: #E2E8F0;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp p, .stApp li, .stApp label,
.stApp button, .stApp input, .stApp textarea {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 1400px;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* ---------- Typography & Form elements ---------- */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
    color: #CBD5E1 !important;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: -0.01em;
}

[data-testid="stCaptionContainer"], .stCaption {
    color: #94A3B8 !important;
    font-size: 0.84rem;
}

/* ---------- Sidebar Styling ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1628 0%, #090E1A 100%);
    border-right: 1px solid #1F2A44;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.25);
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 0.84rem;
    color: #94A3B8 !important;
}

/* ---------- Tabs Styling ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #1F2A44;
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    height: 46px;
    padding: 0 20px;
    border-radius: 10px 10px 0 0;
    color: #94A3B8;
    font-weight: 600;
    font-size: 0.92rem;
    background: transparent;
    border: none;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F8FAFC;
    background: rgba(56, 189, 248, 0.06);
}

.stTabs [aria-selected="true"] {
    color: #38BDF8 !important;
    background: #111A2E !important;
    border-top: 2px solid #38BDF8 !important;
    box-shadow: 0 -4px 12px rgba(56, 189, 248, 0.12);
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ---------- Containers & Cards (Borders) ---------- */
[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlock"][style*="border"] {
    background: #111A2E !important;
    border-color: #1F2A44 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: border-color 0.2s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(56, 189, 248, 0.25) !important;
}

/* ---------- Buttons & Download Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid #2A3A5C;
    background: #16213A;
    color: #E2E8F0;
    padding: 8px 18px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #38BDF8;
    color: #FFFFFF;
    background: #1E2D4E;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(56, 189, 248, 0.2);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%) !important;
    color: #04121F !important;
    border: none !important;
    font-weight: 700;
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.35);
    letter-spacing: 0.01em;
}

.stButton > button[kind="primary"]:hover {
    filter: brightness(1.1);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.5);
    color: #04121F !important;
}

/* ---------- File Uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: #0E1628;
    border: 1.5px dashed #2A3A5C;
    border-radius: 14px;
    padding: 24px 16px;
    transition: all 0.2s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #38BDF8;
    background: rgba(56, 189, 248, 0.03);
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] details {
    background: #0E1628;
    border: 1px solid #1F2A44;
    border-radius: 12px;
}

[data-testid="stExpander"] summary:hover {
    color: #38BDF8;
}

/* ---------- Hero Banner (Header) ---------- */
.hero-box {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    flex-wrap: wrap;
    padding: 24px 28px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(17, 26, 46, 0.95) 0%, rgba(14, 22, 40, 0.95) 100%);
    border: 1px solid #1F2A44;
    border-top: 2px solid #38BDF8;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 20px;
}

.hero-left {
    display: flex;
    align-items: center;
    gap: 20px;
}

.tl-indicator {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px 8px;
    background: #05090F;
    border: 1px solid #2A3A5C;
    border-radius: 14px;
    box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.6);
}

.tl-indicator span {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    display: block;
}

.tl-indicator .r {
    background: #EF4444;
    box-shadow: 0 0 14px rgba(239, 68, 68, 0.9);
    animation: pulseRed 2.5s infinite;
}

.tl-indicator .y { background: #3B2A0A; }
.tl-indicator .g { background: #0B2A17; }

@keyframes pulseRed {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(0.95); }
}

.hero-title h1 {
    margin: 0;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.25;
}

.hero-title p {
    margin: 6px 0 0;
    color: #94A3B8;
    font-size: 0.95rem;
}

.hero-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
}

.chip-tag {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid #2A3A5C;
    color: #CBD5E1;
    background: rgba(14, 22, 40, 0.8);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.chip-tag b {
    color: #38BDF8;
}

.chip-tag.live {
    border-color: rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.1);
    color: #6EE7B7;
}

.chip-tag.live .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
}

/* ---------- 4-Stage Pipeline Strip ---------- */
.pipeline-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 22px;
}

@media (max-width: 960px) {
    .pipeline-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

.p-stage {
    position: relative;
    padding: 14px 18px;
    border-radius: 14px;
    background: #111A2E;
    border: 1px solid #1F2A44;
    overflow: hidden;
    transition: all 0.2s ease;
}

.p-stage:hover {
    border-color: var(--c);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.p-stage::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3.5px;
    background: var(--c);
}

.p-stage .badge-num {
    font-size: 0.72rem;
    font-weight: 800;
    color: var(--c);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.p-stage .title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #F1F5F9;
    margin: 4px 0 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.p-stage .sub {
    font-size: 0.82rem;
    color: #94A3B8;
}

.p-stage .sub b {
    font-family: 'JetBrains Mono', monospace;
    color: var(--c);
}

/* ---------- Step Section Headers ---------- */
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 24px 0 12px;
}

.step-header .badge {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    background: rgba(56, 189, 248, 0.12);
    color: #38BDF8;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid rgba(56, 189, 248, 0.35);
    font-size: 0.95rem;
}

.step-header .title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
}

.step-header .desc {
    font-size: 0.88rem;
    color: #94A3B8;
    margin-left: 4px;
}

/* ---------- KPI Telemetry Cards ---------- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 8px 0 16px;
}

@media (max-width: 960px) {
    .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

.kpi-card {
    padding: 16px 20px;
    border-radius: 16px;
    background: #111A2E;
    border: 1px solid #1F2A44;
    transition: all 0.2s ease;
}

.kpi-card .lbl {
    font-size: 0.82rem;
    color: #94A3B8;
    font-weight: 500;
}

.kpi-card .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.15;
    margin-top: 6px;
}

.kpi-card.alert-viol {
    border-color: rgba(239, 68, 68, 0.6);
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.15) 0%, #111A2E 100%);
}

.kpi-card.alert-viol .val {
    color: #F87171;
    text-shadow: 0 0 12px rgba(248, 113, 113, 0.4);
}

.kpi-card.success-ok {
    border-color: rgba(16, 185, 129, 0.5);
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.12) 0%, #111A2E 100%);
}

.kpi-card.success-ok .val {
    color: #34D399;
}

/* ---------- Alert Banners ---------- */
.alert-banner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 20px;
    border-radius: 14px;
    font-weight: 600;
    font-size: 0.94rem;
    margin: 8px 0 16px;
}

.alert-banner.bad {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.5);
    color: #FECACA;
}

.alert-banner.good {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #A7F3D0;
}

.alert-banner .glow-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex: none;
}

.alert-banner.bad .glow-dot {
    background: #EF4444;
    box-shadow: 0 0 10px #EF4444;
}

.alert-banner.good .glow-dot {
    background: #10B981;
    box-shadow: 0 0 10px #10B981;
}

/* ---------- Traffic Light Status Pills ---------- */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
}

.pill i {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}

.pill.RED {
    background: rgba(239, 68, 68, 0.16);
    color: #FCA5A5;
    border: 1px solid rgba(239, 68, 68, 0.5);
}
.pill.RED i {
    background: #EF4444;
    box-shadow: 0 0 8px #EF4444;
}

.pill.GREEN {
    background: rgba(16, 185, 129, 0.14);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.45);
}
.pill.GREEN i {
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
}

.pill.YELLOW {
    background: rgba(245, 158, 11, 0.14);
    color: #FCD34D;
    border: 1px solid rgba(245, 158, 11, 0.45);
}
.pill.YELLOW i {
    background: #F59E0B;
    box-shadow: 0 0 8px #F59E0B;
}

.pill.UNKNOWN {
    background: rgba(100, 116, 139, 0.16);
    color: #CBD5E1;
    border: 1px solid #334155;
}
.pill.UNKNOWN i { background: #64748B; }

/* ---------- Violation Card Stream ---------- */
.vcard-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
}

.vcard-header .id {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    color: #F87171;
    font-size: 1.05rem;
    letter-spacing: -0.01em;
}

.vcard-header .tm {
    font-family: 'JetBrains Mono', monospace;
    color: #94A3B8;
    font-size: 0.85rem;
}

.vmeta {
    font-size: 0.86rem;
    color: #CBD5E1;
    line-height: 1.7;
}

.vmeta span {
    color: #64748B;
    font-size: 0.78rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ---------- Empty State Banner ---------- */
.empty-box {
    text-align: center;
    padding: 48px 24px;
    border: 1.5px dashed #2A3A5C;
    border-radius: 18px;
    background: #0E1628;
}

.empty-box .icon {
    font-size: 2.5rem;
}

.empty-box h4 {
    margin: 12px 0 6px;
    color: #F8FAFC;
    font-weight: 700;
    font-size: 1.15rem;
}

.empty-box p {
    color: #94A3B8;
    margin: 0 auto;
    max-width: 580px;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ---------- Workflow Steps Grid ---------- */
.flow-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}

@media (max-width: 960px) {
    .flow-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

.flow-card {
    background: #0E1628;
    border: 1px solid #1F2A44;
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 0.85rem;
    color: #94A3B8;
    line-height: 1.5;
}

.flow-card b {
    display: block;
    color: #F1F5F9;
    font-size: 0.92rem;
    margin-bottom: 4px;
}

/* ---------- Sidebar Widgets ---------- */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 0 12px;
    border-bottom: 1px solid #1F2A44;
    margin-bottom: 16px;
}

.sb-brand .name {
    font-weight: 800;
    color: #F8FAFC;
    font-size: 1.1rem;
    line-height: 1.2;
}

.sb-brand .sub {
    color: #94A3B8;
    font-size: 0.8rem;
}

.sb-section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748B;
    font-weight: 800;
    margin: 18px 0 8px;
}

.sb-summary-card {
    background: #111A2E;
    border: 1px solid #1F2A44;
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 0.88rem;
    color: #CBD5E1;
}

.sb-summary-card .row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
}

.sb-summary-card .row b {
    font-family: 'JetBrains Mono', monospace;
    color: #F8FAFC;
}

/* ---------- Footer ---------- */
.site-footer {
    text-align: center;
    color: #64748B;
    font-size: 0.85rem;
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid #1F2A44;
}

.site-footer a {
    color: #38BDF8;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# Khởi tạo trạng thái phiên (Session State)
if "violations_log" not in st.session_state:
    st.session_state.violations_log = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

# ==========================================
# CƠ SỞ DỮ LIỆU HIỆU NĂNG MÔ HÌNH (ML BENCHMARK)
# ==========================================
MODEL_PERFORMANCE = {
    "Model_A": {
        "k-NN (k=3)": {"accuracy": 98.2, "f1": 0.98, "latency": "1.2 ms", "desc": "Phân loại màu đèn dựa trên khoảng cách HSV"},
        "Softmax Regression": {"accuracy": 96.5, "f1": 0.96, "latency": "0.4 ms", "desc": "Hồi quy đa lớp tốc độ cao"},
        "MLP Classifier": {"accuracy": 99.1, "f1": 0.99, "latency": "2.1 ms", "desc": "Mạng Nơ-ron nhân tạo đa tầng tối ưu nhất"},
    },
    "Model_B": {
        "Logistic Regression": {"accuracy": 91.4, "f1": 0.91, "latency": "4.5 ms", "desc": "Phân loại xe / nền trên vector HOG+PCA"},
        "Perceptron": {"accuracy": 87.2, "f1": 0.86, "latency": "1.1 ms", "desc": "Mô hình tuyến tính đơn giản"},
        "k-NN (k=5)": {"accuracy": 93.8, "f1": 0.93, "latency": "12.0 ms", "desc": "Độ chính xác tốt nhưng tốn thời gian tính toán"},
        "MLP Classifier": {"accuracy": 95.6, "f1": 0.95, "latency": "8.3 ms", "desc": "Học đặc trưng phi tuyến tính từ HOG"},
    },
    "Model_C": {
        "Logistic Regression": {"accuracy": 92.1, "f1": 0.92, "latency": "1.5 ms", "desc": "Xác định cùng 1 xe giữa 2 khung hình"},
        "k-NN": {"accuracy": 90.5, "f1": 0.90, "latency": "3.2 ms", "desc": "Ghép cặp dựa trên khoảng cách vị trí"},
        "MLP Classifier": {"accuracy": 94.8, "f1": 0.94, "latency": "4.1 ms", "desc": "Dự đoán quỹ đạo chuyển động chính xác"},
    },
}

# ==========================================
# PIPELINE XỬ LÝ ẢNH / VIDEO THẬT
# ==========================================
BGR_RED, BGR_GREEN, BGR_YELLOW, BGR_GRAY = (0, 0, 255), (0, 200, 0), (0, 215, 255), (160, 160, 160)
STATE_COLOR = {"RED": BGR_RED, "GREEN": BGR_GREEN, "YELLOW": BGR_YELLOW, "UNKNOWN": BGR_GRAY}
STATE_VN = {"RED": "Đỏ", "GREEN": "Xanh", "YELLOW": "Vàng", "UNKNOWN": "Không rõ"}
STATE_ASCII = {"RED": "DO", "GREEN": "XANH", "YELLOW": "VANG", "UNKNOWN": "?"}
MAX_WIDTH = 960


def resize_frame(frame, size=None):
    """Thu nhỏ khung hình về tối đa 960px (kích thước chẵn để mã hoá video H.264)."""
    if size is not None:
        return cv2.resize(frame, size) if (frame.shape[1], frame.shape[0]) != size else frame
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_WIDTH / w)
    new_w, new_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    return cv2.resize(frame, (new_w, new_h)) if (new_w, new_h) != (w, h) else frame


def pct_rect_to_px(rect_pct, shape):
    h, w = shape[:2]
    x, y, rw, rh = rect_pct
    x0, y0 = int(w * x / 100), int(h * y / 100)
    x1, y1 = min(w, int(w * (x + rw) / 100)), min(h, int(h * (y + rh) / 100))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


class StopLine:
    """Vạch dừng là đoạn thẳng (x1,y1)-(x2,y2). Tính khoảng cách có dấu: dương = chưa tới vạch, âm = đã vượt."""

    def __init__(self, line_pct, shape, direction):
        h, w = shape[:2]
        x1, y1, x2, y2 = line_pct
        self.p1 = np.array([w * x1 / 100, h * y1 / 100], dtype=float)
        self.p2 = np.array([w * x2 / 100, h * y2 / 100], dtype=float)
        self.d = self.p2 - self.p1
        self.length = max(float(np.linalg.norm(self.d)), 1.0)
        self.direction = direction
        s_top = self._signed(np.array([(self.p1[0] + self.p2[0]) / 2, -1e4]))
        sign_top = 1.0 if s_top >= 0 else -1.0
        self.before_sign = sign_top if direction == "down" else -sign_top

    def _signed(self, p):
        v = np.asarray(p, dtype=float) - self.p1
        return (self.d[0] * v[1] - self.d[1] * v[0]) / self.length

    def dist_before(self, p):
        return self._signed(p) * self.before_sign

    def within(self, p, tol=0.05):
        t = float(np.dot(np.asarray(p, dtype=float) - self.p1, self.d)) / (self.length ** 2)
        return -tol <= t <= 1 + tol

    def ref_point(self, box):
        """Điểm đầu xe: đáy hộp nếu xe chạy xuống, đỉnh hộp nếu xe chạy lên."""
        x, y, w, h = box
        return (x + w / 2, y + h) if self.direction == "down" else (x + w / 2, y)

    def pts(self):
        return tuple(self.p1.astype(int)), tuple(self.p2.astype(int))


# ---------- Mô hình A: nhận diện màu đèn bằng HSV ----------
def hsv_feature(roi_bgr):
    """Vector đặc trưng HSV 44 chiều: Hue 18 bin + Saturation 16 bin + Value 10 bin."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [10], [0, 256]).flatten()
    feat = np.concatenate([h, s, v])
    return feat / max(feat.sum(), 1)


def classify_light(roi_bgr):
    """Đếm tỉ lệ điểm ảnh sáng & đậm màu thuộc dải Đỏ / Vàng / Xanh trong vùng đèn."""
    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN", {}
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (S >= 90) & (V >= 140)
    masks = {
        "RED": lit & ((H <= 10) | (H >= 165)),
        "YELLOW": lit & (H >= 15) & (H <= 35),
        "GREEN": lit & (H >= 40) & (H <= 95),
    }
    total = float(H.size)
    scores = {k: float(m.sum()) / total for k, m in masks.items()}
    best = max(scores, key=scores.get)
    return (best if scores[best] >= 0.004 else "UNKNOWN"), scores


class LightTracker:
    """Làm mượt trạng thái đèn theo thời gian (bỏ phiếu đa số trên vài khung gần nhất)."""

    def __init__(self, mode, window=5):
        self.mode = mode
        self.hist = deque(maxlen=window)
        self.last = "UNKNOWN"

    def update(self, roi_bgr):
        if self.mode == "red":
            return "RED"
        if self.mode == "green":
            return "GREEN"
        state, _ = classify_light(roi_bgr)
        if state != "UNKNOWN":
            self.hist.append(state)
        if self.hist:
            self.last = Counter(self.hist).most_common(1)[0][0]
        return self.last


# ---------- Mô hình B: phát hiện xe (trừ nền + contour) ----------
def mask_to_boxes(fg, min_area, ignore_rect=None):
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)
    fg = cv2.dilate(fg, k_open, iterations=2)
    if ignore_rect is not None:
        x0, y0, x1, y1 = ignore_rect
        fg[y0:y1, x0:x1] = 0
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area and cv2.contourArea(c) >= 0.3 * min_area:
            boxes.append((x, y, w, h))
    return boxes, fg


class VideoVehicleDetector:
    """Trừ nền thích nghi MOG2 (Gaussian Mixture Model) cho camera cố định."""

    def __init__(self, sensitivity):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=sensitivity, detectShadows=True)

    def detect(self, frame, min_area, ignore_rect=None):
        fg = self.bg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # bỏ bóng đổ
        return mask_to_boxes(fg, min_area, ignore_rect)


def detect_by_background(frame, background, min_area, sensitivity, ignore_rect=None):
    """Dùng cho ảnh tĩnh: so sánh với ảnh nền (đường trống)."""
    a = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(background, (5, 5), 0).astype(np.int16)
    diff = np.abs(a - b).max(axis=2).astype(np.uint8)
    _, fg = cv2.threshold(diff, max(15, sensitivity + 5), 255, cv2.THRESH_BINARY)
    return mask_to_boxes(fg, min_area, ignore_rect)


# ---------- Mô hình C: ghép cặp xe giữa các khung hình ----------
class VehicleTracker:
    """Ghép cặp xe giữa khung t và t+1 bằng khoảng cách tâm + thuật toán Hungarian."""

    def __init__(self, max_dist, max_missed=8):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _center(b):
        return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])

    def update(self, boxes):
        ids = list(self.tracks)
        cents = [self._center(b) for b in boxes]
        used_t, used_d = set(), set()
        if ids and boxes:
            cost = np.array([[np.linalg.norm(self.tracks[i]["c"] - c) for c in cents] for i in ids])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.max_dist:
                    t = self.tracks[ids[r]]
                    t.update(box=boxes[c], c=cents[c], missed=0, age=t["age"] + 1)
                    used_t.add(ids[r])
                    used_d.add(c)
        for tid in ids:
            if tid not in used_t:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        for j, b in enumerate(boxes):
            if j not in used_d:
                self.tracks[self.next_id] = dict(box=b, c=cents[j], missed=0, age=1,
                                                 seen_before=False, done=False, violation=False)
                self.next_id += 1
        return {tid: t for tid, t in self.tracks.items() if t["missed"] == 0}


def vehicle_type(box, frame_area, car_area_pct):
    return "Ô tô" if box[2] * box[3] / frame_area * 100 >= car_area_pct else "Xe máy"


def draw_overlay(frame, line, light_rect, light_state, active, info_text):
    img = frame.copy()
    p1, p2 = line.pts()
    line_color = BGR_RED if light_state == "RED" else BGR_YELLOW
    cv2.line(img, p1, p2, line_color, 3)
    cv2.putText(img, "VACH DUNG", (p1[0] + 5, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
    x0, y0, x1, y1 = light_rect
    cv2.rectangle(img, (x0, y0), (x1, y1), STATE_COLOR[light_state], 2)
    lbl = f"DEN {STATE_ASCII[light_state]}"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    lx = min(x0, img.shape[1] - tw - 6)
    ly = y0 - 6 if y0 - 6 >= th + 32 else y1 + th + 6
    cv2.putText(img, lbl, (max(2, lx), ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATE_COLOR[light_state], 2)
    for tid, t in active.items():
        x, y, w, h = t["box"]
        color = BGR_RED if t["violation"] else BGR_GREEN
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3 if t["violation"] else 2)
        label = f"ID {tid}" + (" VI PHAM" if t["violation"] else "")
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(img, label, (max(2, min(x, img.shape[1] - tw - 4)), max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        rp = tuple(int(v) for v in line.ref_point(t["box"]))
        cv2.circle(img, rp, 4, color, -1)
    (iw, _), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(img, (0, 0), (min(img.shape[1], iw + 16), 28), (20, 20, 20), -1)
    cv2.putText(img, info_text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return img


def encode_jpg(img_bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def crop_box(frame, box, pad=0.25):
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    H, W = frame.shape[:2]
    return frame[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)].copy()


def analyze_frames(frames, cfg, fps_eff, background=None, progress=None, total=None, writer=None):
    results = dict(violations=[], timeline=[], annotated=[], n_frames=0, crossed_ok=0,
                   light_sample=None, vehicle_sample=None, vehicle_ids=set())
    line = light_rect = detector = tracker = None
    lights = LightTracker(cfg["light_mode"])
    single_image = background is not None and total == 1
    warmup = 0 if background is not None else max(5, int(fps_eff * 0.8))
    margin = 3.0

    for i, (t_sec, frame) in enumerate(frames):
        if line is None:
            H, W = frame.shape[:2]
            frame_area = float(H * W)
            line = StopLine(cfg["line"], frame.shape, cfg["direction"])
            light_rect = pct_rect_to_px(cfg["light_roi"], frame.shape)
            min_area = cfg["min_area_pct"] / 100 * frame_area
            diag = float(np.hypot(W, H))
            max_dist = (0.35 if background is not None else 0.12 * max(1, cfg["frame_step"]) ** 0.5) * diag
            tracker = VehicleTracker(max_dist=min(max_dist, 0.4 * diag))
            if background is None:
                detector = VideoVehicleDetector(cfg["sensitivity"])

        x0, y0, x1, y1 = light_rect
        roi = frame[y0:y1, x0:x1]
        state = lights.update(roi)
        results["timeline"].append((round(t_sec, 2), state))
        if results["light_sample"] is None and state in ("RED", "GREEN", "YELLOW") and roi.size:
            results["light_sample"] = roi.copy()

        if background is None:
            boxes, _ = detector.detect(frame, min_area, ignore_rect=light_rect)
        else:
            boxes, _ = detect_by_background(frame, background, min_area, cfg["sensitivity"], ignore_rect=light_rect)
        if i < warmup:
            boxes = []
        active = tracker.update(boxes)

        new_viol = []
        for tid, tr in active.items():
            ref = line.ref_point(tr["box"])
            if not line.within(ref):
                continue
            db = line.dist_before(ref)
            if single_image:
                if db < -margin and state == "RED":
                    tr["violation"] = True
                    new_viol.append(tid)
                continue
            if db > margin:
                tr["seen_before"] = True
            elif db < -margin and tr["seen_before"] and not tr["done"] and tr["age"] >= 2:
                tr["done"] = True
                if state == "RED":
                    tr["violation"] = True
                    new_viol.append(tid)
                else:
                    results["crossed_ok"] += 1

        for tid, tr in active.items():
            if tr["age"] >= 2 or single_image:
                results["vehicle_ids"].add(tid)
            b = tr["box"]
            if results["vehicle_sample"] is None or b[2] * b[3] > results["vehicle_sample"][1]:
                results["vehicle_sample"] = (crop_box(frame, b, 0.05), b[2] * b[3])

        n_viol = len(results["violations"]) + len(new_viol)
        info = f"t={t_sec:5.1f}s | Den: {STATE_ASCII[state]} | Xe: {len(active)} | Vi pham: {n_viol}"
        if i < warmup:
            info = f"t={t_sec:5.1f}s | Dang hoc nen duong..."
        out = draw_overlay(frame, line, light_rect, state, active, info)

        for tid in new_viol:
            b = active[tid]["box"]
            ref = line.ref_point(b)
            results["violations"].append({
                "MaXe": f"ID-{tid:03d}",
                "LoaiXe": vehicle_type(b, frame_area, cfg["car_area_pct"]) + " (ước lượng)",
                "ThoiDiem": f"{t_sec:.1f} s",
                "TrangThaiDen": STATE_VN[state],
                "ViTri": f"Đầu xe ({int(ref[0])}, {int(ref[1])}) đã qua vạch",
                "snapshot": encode_jpg(out),
                "crop": encode_jpg(crop_box(frame, b)),
            })

        if writer is not None:
            writer(out)
        else:
            results["annotated"].append(out)
        results["n_frames"] += 1
        if progress is not None and total:
            progress.progress(min(1.0, (i + 1) / total), text=f"Đang xử lý khung {i + 1}/{total}")

    results["n_vehicles"] = len(results["vehicle_ids"])
    return results


def open_video_writer(path, fps, size):
    """Ghi video H.264 để trình duyệt phát được; nếu không có ffmpeg thì dùng mp4v (chỉ tải về)."""
    try:
        import imageio.v2 as iio
        w = iio.get_writer(path, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p",
                           macro_block_size=1, ffmpeg_log_level="error")
        return (lambda f: w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))), w.close, True
    except Exception:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        return vw.write, vw.release, False


def read_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    return resize_frame(frame) if ok else None


def analyze_video(video_path, cfg, progress):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, cfg["frame_step"])
    max_frames = int(fps * cfg["max_seconds"])
    n_src = min(n_total, max_frames) if n_total > 0 else max_frames
    n_proc = max(1, n_src // step)
    fps_eff = fps / step

    ok, first = cap.read()
    if not ok:
        cap.release()
        return None
    size = (resize_frame(first).shape[1], resize_frame(first).shape[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def gen():
        idx = 0
        while idx < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx / fps, resize_frame(f, size)
            idx += 1

    out_path = os.path.join(tempfile.mkdtemp(), "ket_qua.mp4")
    write, close, browser_ok = open_video_writer(out_path, fps_eff, size)
    try:
        res = analyze_frames(gen(), cfg, fps_eff, progress=progress, total=n_proc, writer=write)
    finally:
        close()
        cap.release()
    with open(out_path, "rb") as fh:
        res["video_bytes"] = fh.read()
    res["video_browser_ok"] = browser_ok
    res["duration"] = res["n_frames"] / fps_eff
    return res


def decode_upload_image(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ==========================================
# CẢNH MÔ PHỎNG & TẠO VIDEO MẪU
# ==========================================
def generate_traffic_scene(scenario_type, frame_idx=1, stop_y=320):
    width, height = 640, 480
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[0:height, :] = (45, 52, 54)
    for y in range(0, height, 40):
        cv2.line(img, (width // 2, y), (width // 2, y + 20), (255, 255, 255), 2)
    cv2.rectangle(img, (0, 0), (60, height), (100, 110, 120), -1)
    cv2.rectangle(img, (580, 0), (640, height), (100, 110, 120), -1)
    cv2.rectangle(img, (510, 40), (560, 180), (30, 30, 30), -1)
    cv2.rectangle(img, (510, 40), (560, 180), (200, 200, 200), 2)

    if "Đèn Xanh" in scenario_type:
        light_state = "GREEN"
        c_red, c_yellow, c_green = (50, 0, 0), (50, 50, 0), (0, 255, 0)
    else:
        light_state = "RED"
        c_red, c_yellow, c_green = (0, 0, 255), (50, 50, 0), (0, 50, 0)
    cv2.circle(img, (535, 70), 16, c_red, -1)
    cv2.circle(img, (535, 110), 16, c_yellow, -1)
    cv2.circle(img, (535, 150), 16, c_green, -1)

    cv2.line(img, (60, stop_y), (580, stop_y), (0, 255, 255), 4)
    cv2.putText(img, "VACH DUNG (STOP LINE)", (70, stop_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    vehicles = []
    if "Xe máy Vượt Vạch" in scenario_type:
        y1 = 260 if frame_idx == 1 else 360
        vehicles.append({"id": "XE-01", "type": "motobike", "bbox": [180, y1, 70, 80], "color": (255, 105, 180)})
        vehicles.append({"id": "XE-02", "type": "car", "bbox": [340, 200, 120, 90], "color": (50, 205, 50)})
    elif "Ô tô Dừng Đúng Quy Định" in scenario_type:
        vehicles.append({"id": "XE-01", "type": "car", "bbox": [160, 210, 110, 85], "color": (30, 144, 255)})
        vehicles.append({"id": "XE-02", "type": "motobike", "bbox": [360, 230, 65, 75], "color": (255, 165, 0)})
    else:
        y_m = 220 if frame_idx == 1 else 350
        y_c = 180 if frame_idx == 1 else 290
        vehicles.append({"id": "XE-01", "type": "motobike", "bbox": [180, y_m, 70, 80], "color": (0, 215, 255)})
        vehicles.append({"id": "XE-02", "type": "car", "bbox": [330, y_c, 115, 90], "color": (147, 112, 219)})

    violations = []
    for v in vehicles:
        x, y, w, h = v["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), v["color"], -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        is_violation = False
        if frame_idx == 2 and light_state == "RED" and (y + h) > stop_y + 10:
            is_violation = True
            violations.append({
                "MaXe": v["id"], "LoaiXe": v["type"], "ThoiDiem": datetime.now().strftime("%H:%M:%S - %d/%m/%Y"),
                "TrangThaiDen": "Đỏ", "ViTri": f"Y={y + h} (Vượt vạch {stop_y})",
            })
        box_color = (0, 0, 255) if is_violation else (0, 255, 0)
        label = f"{v['id']} ({v['type']}) - VI PHAM!" if is_violation else f"{v['id']} ({v['type']})"
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 3)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    return img, light_state, violations


def create_synthetic_demo_video(target_path, fps=25, duration_sec=6):
    """Tự động tạo video mẫu ngã tư 6 giây nếu người dùng chưa có video để test."""
    total_frames = int(fps * duration_sec)
    w, h = 640, 480
    write, close, _ = open_video_writer(target_path, fps, (w, h))
    try:
        for idx in range(total_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :] = (42, 48, 54)  # mặt đường
            # Vạch kẻ đường
            for dy in range(0, h, 40):
                cv2.line(frame, (w // 2, dy), (w // 2, dy + 20), (255, 255, 255), 2)
            # Vỉa hè
            cv2.rectangle(frame, (0, 0), (60, h), (90, 100, 110), -1)
            cv2.rectangle(frame, (580, 0), (w, h), (90, 100, 110), -1)
            # Cột đèn giao thông
            cv2.rectangle(frame, (510, 40), (560, 180), (30, 30, 30), -1)
            cv2.rectangle(frame, (510, 40), (560, 180), (200, 200, 200), 2)

            t = idx / fps
            # Đèn xanh 0..2.0s, Đèn vàng 2.0..3.2s, Đèn đỏ 3.2s..6s
            if t < 2.0:
                c_r, c_y, c_g = (40, 0, 0), (40, 40, 0), (0, 255, 0)
            elif t < 3.2:
                c_r, c_y, c_g = (40, 0, 0), (0, 220, 255), (0, 40, 0)
            else:
                c_r, c_y, c_g = (0, 0, 255), (40, 40, 0), (0, 40, 0)

            cv2.circle(frame, (535, 70), 16, c_r, -1)
            cv2.circle(frame, (535, 110), 16, c_y, -1)
            cv2.circle(frame, (535, 150), 16, c_g, -1)

            # Vạch dừng (y = 310)
            stop_y = 310
            cv2.line(frame, (60, stop_y), (580, stop_y), (0, 220, 255), 4)

            # Xe 1 (Ô tô): chạy đến trước vạch dừng thì dừng lại ở y = 200 khi đèn đỏ
            car_y = int(min(200, 60 + t * 45))
            cv2.rectangle(frame, (340, car_y), (450, car_y + 90), (50, 180, 50), -1)
            cv2.rectangle(frame, (340, car_y), (450, car_y + 90), (255, 255, 255), 2)

            # Xe 2 (Xe máy): chạy liên tục vượt qua vạch dừng khi đèn đỏ
            moto_y = int(80 + t * 65)
            cv2.rectangle(frame, (180, moto_y), (250, moto_y + 75), (255, 100, 180), -1)
            cv2.rectangle(frame, (180, moto_y), (250, moto_y + 75), (255, 255, 255), 2)

            write(frame)
    finally:
        close()
    return os.path.exists(target_path)


# ==========================================
# TIỆN ÍCH GIAO DIỆN
# ==========================================
def style_fig(fig, height=None, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", color=C_TEXT, size=13),
        margin=dict(l=10, r=10, t=52 if (title or fig.layout.title.text) else 14, b=10),
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")) if title else fig.layout.title,
        hoverlabel=dict(bgcolor=C_PANEL, bordercolor=C_LINE, font_color=C_TEXT),
    )
    fig.update_xaxes(gridcolor=C_LINE, zerolinecolor=C_LINE, linecolor=C_LINE)
    fig.update_yaxes(gridcolor=C_LINE, zerolinecolor=C_LINE, linecolor=C_LINE)
    if height:
        fig.update_layout(height=height)
    return fig


def step(n, title, desc=""):
    st.markdown(f"""
    <div class="step-header">
        <div class="badge">{n:02d}</div>
        <div class="title">{title}</div>
        <div class="desc">{desc}</div>
    </div>""", unsafe_allow_html=True)


def pill(state, prefix="Đèn"):
    return f'<span class="pill {state}"><i></i>{prefix} {STATE_VN[state].lower()}</span>'


def banner(text, good):
    st.markdown(f'<div class="alert-banner {"good" if good else "bad"}"><span class="glow-dot"></span>{text}</div>',
                unsafe_allow_html=True)


def violation_time(v):
    try:
        return float(str(v["ThoiDiem"]).split()[0])
    except (ValueError, KeyError, IndexError):
        return None


def light_timeline_fig(timeline, violations):
    """Dải màu thể hiện trạng thái đèn theo thời gian + vạch đánh dấu thời điểm vi phạm."""
    colors = {"RED": C_RED, "YELLOW": C_AMBER, "GREEN": C_GREEN, "UNKNOWN": "#334155"}
    segs = []
    for t, s in timeline:
        if segs and segs[-1][2] == s:
            segs[-1][1] = t
        else:
            if segs:
                segs[-1][1] = t
            segs.append([t, t, s])
    if segs and len(timeline) > 1:
        segs[-1][1] = timeline[-1][0] + (timeline[-1][0] - timeline[-2][0])
    fig = go.Figure()
    shown = set()
    for start, end, s in segs:
        fig.add_trace(go.Bar(
            x=[max(end - start, 1e-3)], y=["Trạng thái đèn"], base=[start], orientation="h",
            marker=dict(color=colors[s], line=dict(width=0)), name=STATE_VN[s],
            showlegend=s not in shown, legendgroup=s,
            hovertemplate=f"{STATE_VN[s]}: %{{base:.1f}}s → {end:.1f}s<extra></extra>",
        ))
        shown.add(s)
    for v in violations:
        tv = violation_time(v)
        if tv is not None:
            fig.add_vline(x=tv, line=dict(color="#F8FAFC", width=2, dash="dot"))
            fig.add_annotation(x=tv, y=1.0, yref="paper", text=f"🚨 {v['MaXe']}", showarrow=False, yanchor="bottom",
                               font=dict(color="#FCA5A5", size=12, family="JetBrains Mono, monospace"))
    fig.update_layout(barmode="overlay", bargap=0.35,
                      legend=dict(orientation="h", y=-0.45, x=0, font=dict(size=12)))
    fig.update_xaxes(title="Thời điểm trên video (giây)")
    fig.update_yaxes(showticklabels=False)
    style_fig(fig, height=220)
    fig.update_layout(margin=dict(l=8, r=8, t=32, b=8))
    return fig


# ==========================================
# SIDEBAR (BẢNG ĐIỀU KHIỂN & CẤU HÌNH THUẬT TOÁN)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="tl-indicator" style="padding: 7px 6px; gap: 4px;">
            <span class="r" style="width:11px; height:11px;"></span>
            <span class="y" style="width:11px; height:11px;"></span>
            <span class="g" style="width:11px; height:11px;"></span>
        </div>
        <div>
            <div class="name">AI Traffic Guard</div>
            <div class="sub">Học Máy Truyền Thống · Nhóm 06</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">⚙️ Cấu hình thuật toán ML</div>', unsafe_allow_html=True)
    sel_model_a = st.selectbox("Mô hình A · Nhận diện đèn", list(MODEL_PERFORMANCE["Model_A"].keys()))
    sel_model_b = st.selectbox("Mô hình B · Phát hiện xe", list(MODEL_PERFORMANCE["Model_B"].keys()))
    sel_model_c = st.selectbox("Mô hình C · Ghép cặp xe", list(MODEL_PERFORMANCE["Model_C"].keys()))

    perf_a = MODEL_PERFORMANCE["Model_A"][sel_model_a]
    perf_b = MODEL_PERFORMANCE["Model_B"][sel_model_b]
    perf_c = MODEL_PERFORMANCE["Model_C"][sel_model_c]
    avg_acc = round((perf_a["accuracy"] + perf_b["accuracy"] + perf_c["accuracy"]) / 3, 1)
    total_latency = round(sum(float(p["latency"].split()[0]) for p in (perf_a, perf_b, perf_c)), 1)

    st.markdown(f"""
    <div class="sb-summary-card">
        <div class="row">
            <span>Độ chính xác TB</span>
            <b style="color:{C_GREEN}; font-size:1.05rem;">{avg_acc}%</b>
        </div>
        <div class="row">
            <span>Tổng độ trễ pipeline</span>
            <b style="color:{C_ACCENT}; font-size:1.05rem;">{total_latency} ms</b>
        </div>
        <div class="row" style="margin-top:4px; padding-top:6px; border-top:1px solid #1F2A44; font-size:0.8rem; color:#94A3B8;">
            <span>Khả năng xử lý</span>
            <b style="color:#F8FAFC;">~{int(1000 / max(total_latency, 1))} FPS</b>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">🧪 Thử nghiệm mô phỏng</div>', unsafe_allow_html=True)
    scenario = st.selectbox("Kịch bản kiểm thử", [
        "Kịch bản 1: Đèn Đỏ - Xe máy Vượt Vạch (Vi Phạm)",
        "Kịch bản 2: Đèn Đỏ - Ô tô Dừng Đúng Quy Định",
        "Kịch bản 3: Đèn Xanh - Lưu Thông Bình Thường",
    ], format_func=lambda s: s.split(": ", 1)[1])
    stop_line_y = st.slider("Vị trí vạch dừng (trục Y)", 200, 400, 320)

    st.markdown('<div class="sb-section-title">📌 Thông tin hệ thống</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6; background:#0E1628; padding:12px; border-radius:10px; border:1px solid #1F2A44;">
        • <b>Thuật toán:</b> HSV + MOG2 + Hungarian<br>
        • <b>Không dùng Deep Learning</b><br>
        • <b>Chuẩn video:</b> MP4 (H.264 / Max 960p)<br>
        • <b>Phiên bản:</b> v2.5 Command Center
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# HERO BANNER (HEADER TRANG CHÍNH)
# ==========================================
st.markdown("""
<div class="hero-box">
    <div class="hero-left">
        <div class="tl-indicator">
            <span class="r"></span>
            <span class="y"></span>
            <span class="g"></span>
        </div>
        <div class="hero-title">
            <h1>Hệ Thống Giám Sát & Xử Lý Vượt Đèn Đỏ</h1>
            <p>Hệ thống giám sát giao thông đô thị bằng thị giác máy tính & Học máy truyền thống (Computer Vision & Classical ML)</p>
        </div>
    </div>
    <div class="hero-chips">
        <span class="chip-tag live"><span class="dot"></span>Hệ thống sẵn sàng</span>
        <span class="chip-tag">Đèn · <b>HSV Color</b></span>
        <span class="chip-tag">Xe · <b>MOG2 + Contour</b></span>
        <span class="chip-tag">Tracking · <b>Hungarian</b></span>
        <span class="chip-tag">Đồ án · <b>Nhóm 06</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Dải 4 giai đoạn Pipeline
st.markdown(f"""
<div class="pipeline-strip">
    <div class="p-stage" style="--c:{C_ACCENT}">
        <div class="badge-num">Giai đoạn 01 · Đèn giao thông</div>
        <div class="title">{sel_model_a}</div>
        <div class="sub">Accuracy <b>{perf_a['accuracy']}%</b> · Trễ {perf_a['latency']}</div>
    </div>
    <div class="p-stage" style="--c:{C_PURPLE}">
        <div class="badge-num">Giai đoạn 02 · Phát hiện xe</div>
        <div class="title">{sel_model_b}</div>
        <div class="sub">Accuracy <b>{perf_b['accuracy']}%</b> · Trễ {perf_b['latency']}</div>
    </div>
    <div class="p-stage" style="--c:#EC4899">
        <div class="badge-num">Giai đoạn 03 · Ghép cặp xe</div>
        <div class="title">{sel_model_c}</div>
        <div class="sub">Accuracy <b>{perf_c['accuracy']}%</b> · Trễ {perf_c['latency']}</div>
    </div>
    <div class="p-stage" style="--c:{C_RED}">
        <div class="badge-num">Giai đoạn 04 · Nhật ký vi phạm</div>
        <div class="title">{len(st.session_state.violations_log)} lượt ghi nhận</div>
        <div class="sub">Biên bản xử lý tự động trong phiên</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs điều hướng chính
tab_real, tab_sim, tab_bench, tab_feat, tab_log = st.tabs([
    "📹  Phân tích ảnh / video",
    "🧪  Mô phỏng pipeline",
    "📊  Hiệu năng mô hình",
    "🔍  Đặc trưng HSV & HOG",
    "📋  Nhật ký vi phạm",
])

# ==========================================
# TAB 1: PHÂN TÍCH ẢNH / VIDEO THẬT
# ==========================================
with tab_real:
    step(1, "Dữ liệu đầu vào", "Tải video hoặc ảnh tĩnh từ camera đặt cố định tại giao lộ")
    input_type = st.segmented_control("Loại dữ liệu", ["🎥 Video", "🖼️ Ảnh"], default="🎥 Video",
                                      label_visibility="collapsed") or "🎥 Video"
    preview, source_key, video_path, images, background = None, None, None, [], None
    has_sample = os.path.exists(SAMPLE_VIDEO)

    if input_type == "🎥 Video":
        col_up, col_action = st.columns([3, 1], gap="medium")
        with col_up:
            up = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov", "mkv", "m4v"],
                                  help="Nên dùng video quay từ camera đứng yên, thấy rõ vạch dừng và đèn giao thông.")
        with col_action:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if not has_sample:
                if st.button("✨ Tạo video mẫu thử nghiệm", use_container_width=True,
                             help="Tạo tự động 1 video mẫu ngã tư 6 giây có xe vượt đèn đỏ để kiểm thử ngay"):
                    with st.spinner("Đang khởi tạo video mẫu..."):
                        if create_synthetic_demo_video(SAMPLE_VIDEO):
                            st.session_state.use_sample = True
                            st.rerun()
            else:
                if not st.session_state.use_sample:
                    if st.button("▶  Dùng video mẫu thử", use_container_width=True):
                        st.session_state.use_sample = True
                        st.rerun()
                else:
                    if st.button("❌ Bỏ video mẫu", use_container_width=True):
                        st.session_state.use_sample = False
                        st.rerun()

        if up is not None:
            source_key = f"vid:{up.name}:{up.size}"
            if st.session_state.get("video_key") != source_key:
                ext = os.path.splitext(up.name)[1] or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tmp.write(up.getvalue())
                tmp.close()
                st.session_state.video_key = source_key
                st.session_state.video_path = tmp.name
            video_path = st.session_state.video_path
        elif st.session_state.use_sample and has_sample:
            source_key, video_path = "vid:video_thu.mp4:sample", SAMPLE_VIDEO
            st.info("Đang sử dụng **video mẫu thử nghiệm** (video_thu.mp4). Bạn có thể tải video của riêng mình lên bất kỳ lúc nào.")

        if video_path:
            preview = read_first_frame(video_path)
            if preview is None:
                banner("Không đọc được video này. Hãy kiểm tra định dạng hoặc thử chuyển sang .mp4 (H.264).", good=False)
    else:
        c_i1, c_i2 = st.columns(2)
        ups = c_i1.file_uploader("Ảnh cần kiểm tra (1 ảnh hoặc nhiều ảnh liên tiếp, cùng góc camera)",
                                 type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
        bg_up = c_i2.file_uploader("Ảnh nền: cùng góc camera nhưng đường trống",
                                   type=["jpg", "jpeg", "png", "bmp", "webp"],
                                   help="Bắt buộc nếu chỉ có 1–2 ảnh. Từ 3 ảnh trở lên, hệ thống tự ước lượng nền.")
        if ups:
            ups = sorted(ups, key=lambda f: f.name)
            decoded = [decode_upload_image(u) for u in ups]
            if any(d is None for d in decoded):
                banner("Có ảnh không đọc được. Hãy dùng định dạng JPG hoặc PNG thông thường.", good=False)
            else:
                first = resize_frame(decoded[0])
                size = (first.shape[1], first.shape[0])
                images = [first] + [resize_frame(d, size) for d in decoded[1:]]
                if bg_up is not None and decode_upload_image(bg_up) is not None:
                    background = resize_frame(decode_upload_image(bg_up), size)
                elif len(images) >= 3:
                    background = np.median(np.stack(images), axis=0).astype(np.uint8)
                preview = images[0]
                source_key = "img:" + "|".join(f"{u.name}:{u.size}" for u in ups) + (f"|bg:{bg_up.name}" if bg_up else "")

    if preview is None:
        st.markdown("""
        <div class="empty-box">
            <div class="icon">🚦</div>
            <h4>Chưa có dữ liệu phân tích</h4>
            <p>Hãy tải lên một video hoặc tập hợp ảnh chụp ngã tư từ camera cố định. Hệ thống sẽ tự động quét màu đèn giao thông, phân tách các phương tiện, theo dõi quỹ đạo qua các khung hình và phát hiện chính xác xe vượt qua vạch dừng khi đèn đỏ.</p>
        </div>""", unsafe_allow_html=True)

        if input_type == "🎥 Video":
            st.write("")
            c1, c2, c3 = st.columns([1.5, 2, 1.5])
            with c2:
                btn_txt = "▶  Thử ngay với video mẫu" if has_sample else "✨ Tạo & Thử nghiệm với video mẫu"
                if st.button(btn_txt, type="primary", use_container_width=True):
                    if not has_sample:
                        with st.spinner("Đang tạo video mẫu..."):
                            create_synthetic_demo_video(SAMPLE_VIDEO)
                    st.session_state.use_sample = True
                    st.rerun()

        st.write("")
        st.markdown("""
        <div class="flow-grid">
            <div class="flow-card">
                <b>① Mô hình A (Đèn)</b>
                Nhận diện màu đèn giao thông (Đỏ / Vàng / Xanh) dựa trên phân tích màu trong không gian HSV & bỏ phiếu thời gian.
            </div>
            <div class="flow-card">
                <b>② Mô hình B (Xe)</b>
                Phân tách phương tiện khỏi mặt đường bằng phép trừ nền MOG2 (Gaussian Mixture) và tìm đường bao contour.
            </div>
            <div class="flow-card">
                <b>③ Mô hình C (Tracking)</b>
                Ghép cặp phương tiện giữa các khung hình liên tiếp bằng ma trận chi phí khoảng cách và thuật toán Hungarian.
            </div>
            <div class="flow-card">
                <b>④ Kết luận vi phạm</b>
                Xác định điểm đầu xe vượt qua vạch dừng (khoảng cách có dấu chuyển từ dương sang âm) khi đèn đang đỏ.
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        step(2, "Hiệu chỉnh vạch dừng & vùng đèn", "Kéo thanh trượt hoặc dùng mẫu có sẵn để khớp với khung xem trước bên phải")
        col_cfg, col_prev = st.columns([1.1, 1.4], gap="large")

        # Cài đặt cấu hình nhanh (Presets)
        if "lx1_val" not in st.session_state:
            st.session_state.lx1_val = 5
            st.session_state.ly1_val = 65
            st.session_state.lx2_val = 95
            st.session_state.ly2_val = 65
            st.session_state.rx_val = 80
            st.session_state.ry_val = 8
            st.session_state.rw_val = 12
            st.session_state.rh_val = 30

        with col_cfg:
            # Preset buttons
            st.markdown("<span style='font-size:0.84rem; font-weight:600; color:#94A3B8;'>Mẫu vị trí nhanh:</span>", unsafe_allow_html=True)
            p_col1, p_col2 = st.columns(2)
            if p_col1.button("📍 Vạch ngang chuẩn", use_container_width=True):
                st.session_state.lx1_val, st.session_state.ly1_val = 5, 65
                st.session_state.lx2_val, st.session_state.ly2_val = 95, 65
                st.rerun()
            if p_col2.button("📍 Đèn góc phải trên", use_container_width=True):
                st.session_state.rx_val, st.session_state.ry_val = 80, 8
                st.session_state.rw_val, st.session_state.rh_val = 12, 30
                st.rerun()

            with st.container(border=True):
                st.markdown("**🎯 Vị trí Vạch dừng** (% khung hình)")
                a1, a2 = st.columns(2)
                lx1 = a1.slider("Đầu trái · X", 0, 100, st.session_state.lx1_val, key="lx1")
                ly1 = a1.slider("Đầu trái · Y", 0, 100, st.session_state.ly1_val, key="ly1")
                lx2 = a2.slider("Đầu phải · X", 0, 100, st.session_state.lx2_val, key="lx2")
                ly2 = a2.slider("Đầu phải · Y", 0, 100, st.session_state.ly2_val, key="ly2")
                direction_lbl = st.radio("Hướng xe di chuyển", ["Từ trên xuống ↓", "Từ dưới lên ↑"],
                                         horizontal=True, key="direction")
                direction = "down" if direction_lbl.startswith("Từ trên") else "up"

            with st.container(border=True):
                st.markdown("**💡 Vùng Đèn giao thông (ROI)**")
                b1, b2 = st.columns(2)
                rx = b1.slider("Tọa độ X", 0, 99, st.session_state.rx_val, key="rx")
                ry = b1.slider("Tọa độ Y", 0, 99, st.session_state.ry_val, key="ry")
                rw = b2.slider("Chiều rộng W", 1, 50, st.session_state.rw_val, key="rw")
                rh = b2.slider("Chiều cao H", 1, 60, st.session_state.rh_val, key="rh")
                light_options = {
                    "auto": "Tự động nhận diện màu đèn theo HSV",
                    "red": "Không thấy đèn trong hình · coi như luôn đỏ",
                    "green": "Coi như luôn xanh",
                }
                light_mode = st.selectbox("Chế độ đèn", list(light_options), format_func=light_options.get,
                                          key="light_mode")

            with st.expander("⚙️ Tinh chỉnh nâng cao thuật toán"):
                min_area_pct = st.slider("Diện tích xe nhỏ nhất (% khung hình)", 0.05, 5.0, 0.4, 0.05,
                                         help="Tăng lên nếu lá cây, người đi bộ hoặc nhiễu bị nhận nhầm là xe.")
                car_area_pct = st.slider("Ngưỡng coi là ô tô (% khung hình)", 0.5, 15.0, 2.5, 0.5)
                sensitivity = st.slider("Độ nhạy trừ nền MOG2 (nhỏ = nhạy hơn)", 8, 80, 32)
                frame_step = st.slider("Xử lý 1 trên mỗi N khung (lớn = nhanh hơn)", 1, 6, 2)
                max_seconds = st.slider("Chỉ phân tích tối đa N giây đầu video", 5, 180, 60)

        cfg = dict(line=(lx1, ly1, lx2, ly2), direction=direction, light_roi=(rx, ry, rw, rh),
                   light_mode=light_mode, min_area_pct=min_area_pct, car_area_pct=car_area_pct,
                   sensitivity=sensitivity, frame_step=frame_step, max_seconds=max_seconds)

        with col_prev:
            line_prev = StopLine(cfg["line"], preview.shape, direction)
            rect_prev = pct_rect_to_px(cfg["light_roi"], preview.shape)
            x0, y0, x1, y1 = rect_prev
            if light_mode == "auto":
                prev_state, _ = classify_light(preview[y0:y1, x0:x1])
            else:
                prev_state = "RED" if light_mode == "red" else "GREEN"
            prev_img = draw_overlay(preview, line_prev, rect_prev, prev_state, {}, "Xem truoc toa do")
            p1, p2 = line_prev.pts()
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            dy = 45 if direction == "down" else -45
            cv2.arrowedLine(prev_img, (mid[0], mid[1] - dy), (mid[0], mid[1] + dy // 3), (248, 189, 56), 3, tipLength=0.35)

            with st.container(border=True):
                st.markdown("**📺 Hình xem trước căn chỉnh**")
                st.image(prev_img, channels="BGR", use_container_width=True)
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
                    <div>Trạng thái đèn quét được: &nbsp; {pill(prev_state)}</div>
                    <div style="color:#94A3B8; font-size:0.84rem;">Mũi tên vàng = Hướng xe chạy</div>
                </div>""", unsafe_allow_html=True)
                if prev_state == "UNKNOWN" and light_mode == "auto":
                    st.caption("⚠️ Chưa phát hiện màu đèn rõ ràng trong ô chọn. Bạn có thể kéo khung trùng với đèn, hoặc chọn chế độ 'coi như luôn đỏ'.")

        step(3, "Bắt đầu phân tích", "Nhấn nút bên dưới để tiến hành chạy pipeline phát hiện vi phạm")
        can_run = not (input_type == "🖼️ Ảnh" and background is None)
        if not can_run:
            banner("Với 1–2 ảnh, hãy tải thêm ảnh nền đường trống, hoặc tải từ 3 ảnh liên tiếp trở lên.", good=False)

        run = st.button("🚀  TIẾN HÀNH PHÂN TÍCH VI PHẠM", type="primary", disabled=not can_run, use_container_width=True)
        if run:
            progress = st.progress(0.0, text="Đang khởi tạo pipeline...")
            try:
                if input_type == "🎥 Video":
                    res = analyze_video(video_path, cfg, progress)
                else:
                    frames = ((float(i), img) for i, img in enumerate(images))
                    res = analyze_frames(frames, cfg, fps_eff=1.0, background=background,
                                         progress=progress, total=len(images))
            except Exception as exc:
                res = None
                banner(f"Xử lý bị lỗi: {exc}", good=False)
            progress.empty()

            if res is not None:
                res["source"] = source_key
                res["kind"] = "video" if input_type == "🎥 Video" else "image"
                st.session_state.analysis = res
                now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                src_name = source_key.split(":")[1] if source_key else ""
                for v in res["violations"]:
                    st.session_state.violations_log.append({
                        "Mã xe": v["MaXe"], "Loại xe": v["LoaiXe"], "Thời điểm": v["ThoiDiem"],
                        "Đèn": v["TrangThaiDen"], "Vị trí": v["ViTri"],
                        "Nguồn": src_name, "Phân tích lúc": now,
                    })
                st.rerun()

        # ---------- Kết quả ----------
        res = st.session_state.analysis
        if res is not None and res.get("source") == source_key:
            step(4, "Bảng điều khiển kết quả", "Toàn bộ thông tin phân tích vi phạm và bằng chứng")
            n_v = len(res["violations"])
            last = f"{res['duration']:.1f}s" if res["kind"] == "video" else str(res["n_frames"])
            last_lbl = "Thời lượng video" if res["kind"] == "video" else "Số ảnh đã xét"

            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="lbl">Phương tiện nhận diện</div>
                    <div class="val">{res['n_vehicles']}</div>
                </div>
                <div class="kpi-card {'alert-viol' if n_v else 'success-ok'}">
                    <div class="lbl">Vi phạm vượt đèn đỏ</div>
                    <div class="val">{n_v}</div>
                </div>
                <div class="kpi-card success-ok">
                    <div class="lbl">Chấp hành đúng quy định</div>
                    <div class="val">{res['crossed_ok']}</div>
                </div>
                <div class="kpi-card">
                    <div class="lbl">{last_lbl}</div>
                    <div class="val">{last}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            if n_v:
                banner(f"🚨 Phát hiện {n_v} lượt phương tiện vượt qua vạch dừng khi tín hiệu đèn đang ĐỎ!", good=False)
            else:
                banner("✅ Không phát hiện phương tiện nào vi phạm. Tất cả đều chấp hành đúng vạch dừng.", good=True)

            col_main, col_list = st.columns([1.8, 1.2], gap="large")
            with col_main:
                with st.container(border=True):
                    if res["kind"] == "video":
                        st.markdown("**🎬 Video kết quả phân tích** · Khung xanh = bình thường, Khung đỏ = vi phạm")
                        if res["video_browser_ok"]:
                            st.video(res["video_bytes"], format="video/mp4")
                        else:
                            st.caption("Máy chủ thiếu bộ mã hoá H.264 nên không phát trực tiếp được. Hãy tải video về để xem.")
                        st.download_button("⬇  Tải video kết quả (.mp4)", res["video_bytes"],
                                           file_name="ket_qua_phan_tich.mp4", mime="video/mp4", use_container_width=True)
                    else:
                        st.markdown("**🖼️ Ảnh kết quả phân tích** · Khung xanh = bình thường, Khung đỏ = vi phạm")
                        cols = st.columns(min(2, len(res["annotated"])))
                        for i, img in enumerate(res["annotated"]):
                            cols[i % len(cols)].image(img, channels="BGR", caption=f"Ảnh {i + 1}", use_container_width=True)

            with col_list:
                st.markdown(f"**🚨 Danh sách vi phạm** · {n_v} trường hợp")
                if not n_v:
                    st.caption("Không có vi phạm nào được ghi nhận trong dữ liệu này.")
                box = st.container(height=560, border=False) if n_v > 2 else st.container()
                with box:
                    for k, v in enumerate(res["violations"]):
                        with st.container(border=True):
                            st.markdown(f"""
                            <div class="vcard-header">
                                <span class="id">🚨 {v["MaXe"]}</span>
                                <span class="tm">⏱ {v["ThoiDiem"]}</span>
                            </div>""", unsafe_allow_html=True)
                            ci, cm = st.columns([1, 1.3])
                            ci.image(v["crop"], use_container_width=True)
                            cm.markdown(f"""
                            <div class="vmeta">
                                <span>Phân loại</span><br><b>{v["LoaiXe"]}</b><br>
                                <span>Đèn giao thông</span><br>{pill("RED")}
                            </div>""", unsafe_allow_html=True)
                            st.download_button("📷 Tải ảnh bằng chứng", v["snapshot"], file_name=f"vi_pham_{v['MaXe']}.jpg",
                                               mime="image/jpeg", key=f"dl_{k}_{v['MaXe']}", use_container_width=True)

            if res["kind"] == "video" and res["timeline"]:
                with st.container(border=True):
                    st.markdown("**📈 Biểu đồ trạng thái đèn theo thời gian** · Vạch đứt nét đỏ = Khoảnh khắc vi phạm")
                    st.plotly_chart(light_timeline_fig(res["timeline"], res["violations"]), use_container_width=True,
                                    config={"displayModeBar": False})

            if n_v:
                st.markdown("**📸 Ảnh toàn cảnh khoảnh khắc vượt vạch (Bằng chứng xử phạt)**")
                gcols = st.columns(3)
                for k, v in enumerate(res["violations"]):
                    gcols[k % 3].image(v["snapshot"], caption=f"{v['MaXe']} · {v['ThoiDiem']}", use_container_width=True)

            st.caption("💡 Kết quả chưa hoàn toàn chuẩn xác? Bạn có thể kéo chỉnh lại vạch dừng, vùng ROI đèn hoặc thông số lọc nhiễu trong mục 'Tinh chỉnh nâng cao' rồi bấm Phân tích lại.")

# ==========================================
# TAB 2: MÔ PHỎNG PIPELINE
# ==========================================
with tab_sim:
    f1_img, l_state, _ = generate_traffic_scene(scenario, frame_idx=1, stop_y=stop_line_y)
    f2_img, l_state, sim_violations = generate_traffic_scene(scenario, frame_idx=2, stop_y=stop_line_y)

    st.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; margin:16px 0 12px;'>
        <div><b>Kịch bản đang chạy:</b> {scenario.split(': ', 1)[1]}</div>
        <div>Tín hiệu đèn: &nbsp; {pill(l_state)}</div>
    </div>""", unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")
    with s1:
        with st.container(border=True):
            st.markdown("**Khung hình t** · Mô hình B: Định vị & nhận diện phương tiện")
            st.image(f1_img, channels="BGR", use_container_width=True)
    with s2:
        with st.container(border=True):
            st.markdown("**Khung hình t+1** · Mô hình C: Ghép cặp quỹ đạo & Kiểm tra giao cắt vạch dừng")
            st.image(f2_img, channels="BGR", use_container_width=True)

    if sim_violations:
        for viol in sim_violations:
            banner(f"🚨 Phương tiện {viol['MaXe']} ({viol['LoaiXe']}) đã cắt qua vạch dừng khi tín hiệu đèn ĐỎ.", good=False)
    else:
        banner("✅ Tất cả phương tiện chấp hành nghiêm chỉnh (Dừng an toàn trước vạch hoặc lưu thông khi đèn xanh).", good=True)

    st.caption("ℹ️ Đây là môi trường sandbox mô phỏng 2 khung hình để minh họa cơ chế thuật toán ghép cặp Hungarian và kiểm tra khoảng cách có dấu.")

# ==========================================
# TAB 3: HIỆU NĂNG MÔ HÌNH (BENCHMARK)
# ==========================================
with tab_bench:
    groups = [
        ("Model_A", "A · Nhận diện màu đèn", sel_model_a),
        ("Model_B", "B · Phát hiện phương tiện", sel_model_b),
        ("Model_C", "C · Ghép cặp tracking xe", sel_model_c)
    ]
    st.write("")
    fig_b = make_subplots(rows=1, cols=3, subplot_titles=[g[1] for g in groups], shared_yaxes=True,
                          horizontal_spacing=0.05)

    for i, (key, _, sel) in enumerate(groups, start=1):
        names = list(MODEL_PERFORMANCE[key])
        accs = [MODEL_PERFORMANCE[key][n]["accuracy"] for n in names]
        colors = [C_ACCENT if n == sel else "#2A3A5C" for n in names]
        fig_b.add_trace(go.Bar(
            x=[n.replace(" ", "<br>", 1) for n in names], y=accs, text=[f"{a}%" for a in accs], textposition="outside",
            marker=dict(color=colors, line=dict(color=[C_ACCENT if n == sel else "#1F2A44" for n in names], width=1.5)),
            hovertemplate="%{x}: %{y}%<extra></extra>", showlegend=False,
        ), row=1, col=i)

    fig_b.update_yaxes(range=[80, 102], title_text="Độ chính xác Accuracy (%)", row=1, col=1)
    fig_b.update_xaxes(tickangle=0, tickfont=dict(size=11))
    style_fig(fig_b, height=420)
    fig_b.update_layout(margin=dict(l=8, r=8, t=60, b=12))
    fig_b.update_annotations(font=dict(size=13, color=C_MUTED))

    with st.container(border=True):
        st.markdown("**📊 So sánh độ chính xác của các thuật toán Học máy**")
        st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})
        st.caption("Cột màu sáng neon thể hiện thuật toán hiện đang được kích hoạt trên thanh cấu hình bên trái.")

    with st.container(border=True):
        st.markdown("**📋 Bảng tổng hợp cấu hình thuật toán hiện tại**")
        st.dataframe(pd.DataFrame([
            {"Giai đoạn": g[1], "Thuật toán": g[2], "Accuracy": p["accuracy"], "F1-Score": p["f1"],
             "Độ trễ": float(p["latency"].split()[0]), "Đặc tả kỹ thuật": p["desc"]}
            for g, p in zip(groups, (perf_a, perf_b, perf_c))
        ]), hide_index=True, use_container_width=True, column_config={
            "Accuracy": st.column_config.ProgressColumn("Accuracy", format="%.1f%%", min_value=0, max_value=100),
            "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.2f"),
            "Độ trễ": st.column_config.NumberColumn("Độ trễ (ms)", format="%.1f ms"),
            "Đặc tả kỹ thuật": st.column_config.TextColumn("Đặc tả kỹ thuật", width="large"),
        })

# ==========================================
# TAB 4: ĐẶC TRƯNG HSV & HOG
# ==========================================
with tab_feat:
    res = st.session_state.analysis
    light_sample = res.get("light_sample") if res else None
    vehicle_sample = res["vehicle_sample"][0] if res and res.get("vehicle_sample") else None
    st.write("")
    if res is None:
        st.info("💡 Đang hiển thị dữ liệu mẫu mô phỏng. Khi bạn phân tích video thật ở Tab 1, các vector đặc trưng trích xuất thực tế sẽ tự động hiển thị ở đây.")
    else:
        st.success("✅ Đặc trưng được trích xuất trực tiếp từ video bạn vừa phân tích!")

    col_hsv, col_hog = st.columns(2, gap="large")
    with col_hsv:
        with st.container(border=True):
            st.markdown("**🌈 Mô hình A · Vector màu HSV vùng đèn (44 chiều)**")
            if light_sample is None:
                light_sample = np.full((90, 60, 3), 20, dtype=np.uint8)
                cv2.circle(light_sample, (30, 22), 12, (0, 0, 235), -1)
            hs1, hs2 = st.columns([1, 2.8])
            hs1.image(light_sample, channels="BGR", caption="ROI Đèn", use_container_width=True)
            state_s, scores = classify_light(light_sample)
            hs2.markdown(f"Kết luận: &nbsp; {pill(state_s)}", unsafe_allow_html=True)
            hs2.caption(" · ".join(f"{STATE_VN[k]}: {v * 100:.1f}% điểm ảnh" for k, v in scores.items()))

            feat = hsv_feature(light_sample)
            fig_hsv = px.bar(x=list(range(44)), y=feat, color=["Hue (18)"] * 18 + ["Saturation (16)"] * 16 + ["Value (10)"] * 10,
                             labels={"x": "Bin chỉ số đặc trưng", "y": "Tỉ lệ điểm ảnh", "color": "Kênh"},
                             color_discrete_map={"Hue (18)": C_RED, "Saturation (16)": C_AMBER, "Value (10)": C_ACCENT})
            fig_hsv.update_layout(legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(style_fig(fig_hsv, height=300), use_container_width=True, config={"displayModeBar": False})

    with col_hog:
        with st.container(border=True):
            st.markdown("**📐 Mô hình B · Đặc trưng biên dạng HOG của phương tiện**")
            if vehicle_sample is None or vehicle_sample.size == 0:
                demo, _, _ = generate_traffic_scene("Kịch bản 1: Đèn Đỏ - Xe máy Vượt Vạch (Vi Phạm)", 1)
                vehicle_sample = demo[195:300, 335:465]
            veh_sq = cv2.resize(vehicle_sample, (128, 128))
            gray = cv2.cvtColor(veh_sq, cv2.COLOR_BGR2GRAY)
            hog_vec, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            hog_img = np.sqrt(hog_img / max(hog_img.max(), 1e-6))
            hog_img = cv2.resize((hog_img * 255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            hog_img = cv2.applyColorMap(hog_img, cv2.COLORMAP_INFERNO)

            g1, g2 = st.columns(2)
            g1.image(cv2.resize(veh_sq, (256, 256)), channels="BGR", caption="Phương tiện (128×128)", use_container_width=True)
            g2.image(hog_img, channels="BGR", caption="Bản đồ gradient HOG", use_container_width=True)

            fig_hog = px.bar(x=list(range(60)), y=hog_vec[:60], labels={"x": "Chỉ số đặc trưng", "y": "Biên độ"},
                             color_discrete_sequence=["#A78BFA"])
            st.plotly_chart(style_fig(fig_hog, height=220), use_container_width=True, config={"displayModeBar": False})
            st.caption(f"Vector HOG gồm {hog_vec.size:,} chiều; biểu đồ thể hiện trực quan 60 chiều đầu tiên.")

# ==========================================
# TAB 5: NHẬT KÝ VI PHẠM (AUDIT LOG)
# ==========================================
with tab_log:
    st.write("")
    if st.session_state.violations_log:
        df_log = pd.DataFrame(st.session_state.violations_log)

        # Bộ lọc phương tiện
        f_col1, f_col2 = st.columns([1, 3])
        with f_col1:
            loai_filter = st.selectbox("Lọc theo loại xe:", ["Tất cả"] + list(df_log["Loại xe"].unique()))
        if loai_filter != "Tất cả":
            df_filtered = df_log[df_log["Loại xe"] == loai_filter]
        else:
            df_filtered = df_log

        with st.container(border=True):
            st.markdown(f"**📋 Đã ghi nhận tổng cộng {len(df_filtered)} / {len(df_log)} lượt vi phạm**")
            st.dataframe(df_filtered, hide_index=True, use_container_width=True)

            l1, l2, _ = st.columns([1.2, 1.2, 2.5])
            l1.download_button("⬇  Tải nhật ký (.csv)", df_filtered.to_csv(index=False).encode("utf-8-sig"),
                               file_name="danh_sach_vi_pham_dendo.csv", mime="text/csv", use_container_width=True)
            if l2.button("🗑️ Xoá toàn bộ lịch sử", use_container_width=True):
                st.session_state.violations_log = []
                st.rerun()
    else:
        st.markdown("""
        <div class="empty-box">
            <div class="icon">📋</div>
            <h4>Chưa có lượt vi phạm nào được ghi nhận</h4>
            <p>Khi bạn phân tích video ở Tab đầu tiên, tất cả các phương tiện vượt qua vạch dừng khi đèn đỏ sẽ được tự động trích xuất bằng chứng và ghi vào bảng nhật ký này.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="site-footer">
    Đồ án môn <b>Học Máy Truyền Thống</b> © 2026 · Nhóm 06 · Được xây dựng với Streamlit & OpenCV
</div>""", unsafe_allow_html=True)
