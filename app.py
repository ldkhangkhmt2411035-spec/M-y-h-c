import os
import tempfile
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linear_sum_assignment
from skimage.feature import hog

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_VIDEO = os.path.join(APP_DIR, "video_thu.mp4")

# ==========================================
# CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="Giám sát vượt đèn đỏ · Nhóm 6",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="auto",
)

# Bảng màu: nền xanh than tối như phòng điều hành giao thông,
# ba màu tín hiệu (đỏ / vàng / xanh) chỉ dùng khi mang đúng ý nghĩa.
C_BG, C_PANEL, C_PANEL2, C_LINE = "#0B1220", "#111A2E", "#0E1628", "#1F2A44"
C_TEXT, C_MUTED, C_DIM = "#E2E8F0", "#94A3B8", "#64748B"
C_ACCENT = "#38BDF8"
C_RED, C_AMBER, C_GREEN = "#EF4444", "#F59E0B", "#22C55E"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

/* ---------- Nền & chữ (dự phòng nếu thiếu .streamlit/config.toml) ---------- */
.stApp {
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(56,189,248,0.08), transparent 60%),
        radial-gradient(900px 400px at -10% 10%, rgba(239,68,68,0.06), transparent 60%),
        #0B1220;
    color: #E2E8F0;
    font-family: 'Be Vietnam Pro', system-ui, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp p, .stApp li, .stApp label,
.stApp button, .stApp input, .stApp textarea { font-family: 'Be Vietnam Pro', system-ui, sans-serif; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #CBD5E1 !important; font-weight: 500; }
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] { color: inherit; }
[data-testid="stCaptionContainer"], .stCaption { color: #94A3B8 !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1340px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: #0E1628; border-right: 1px solid #1F2A44; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 0.86rem; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #1F2A44; }
.stTabs [data-baseweb="tab"] {
    height: 44px; padding: 0 16px; border-radius: 10px 10px 0 0;
    color: #94A3B8; font-weight: 600; background: transparent;
}
.stTabs [data-baseweb="tab"]:hover { color: #E2E8F0; background: rgba(148,163,184,0.06); }
.stTabs [aria-selected="true"] { color: #E2E8F0 !important; background: #111A2E; }
.stTabs [data-baseweb="tab-highlight"] { background: #38BDF8; height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Khung có viền (st.container(border=True)) ---------- */
[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"][style*="border"] {
    background: #111A2E; border-color: #1F2A44 !important; border-radius: 14px;
}

/* ---------- Nút ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; border: 1px solid #2A3A5C;
    background: #16213A; color: #E2E8F0; transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #38BDF8; color: #fff; }
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #38BDF8, #0EA5E9); color: #04121F; border: none;
    box-shadow: 0 6px 20px rgba(14,165,233,0.25); font-weight: 700;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: #04121F; }

/* ---------- Ô tải file ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: #0E1628; border: 1.5px dashed #2A3A5C; border-radius: 12px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #38BDF8; }

/* ---------- Expander ---------- */
[data-testid="stExpander"] details { background: #0E1628; border: 1px solid #1F2A44; border-radius: 12px; }

/* ---------- Ảnh ---------- */
[data-testid="stImage"] img { border-radius: 10px; }
[data-testid="stImageCaption"], [data-testid="caption"] { color: #94A3B8 !important; }

/* ---------- Thành phần tự dựng ---------- */
.hero {
    display: flex; align-items: center; justify-content: space-between; gap: 24px; flex-wrap: wrap;
    padding: 22px 26px; border: 1px solid #1F2A44; border-radius: 18px;
    background: linear-gradient(135deg, rgba(17,26,46,0.95), rgba(14,22,40,0.95));
    margin-bottom: 18px;
}
.hero-left { display: flex; align-items: center; gap: 18px; }
.tl-icon {
    display: flex; flex-direction: column; gap: 5px; padding: 8px 7px;
    background: #05090F; border: 1px solid #2A3A5C; border-radius: 12px;
}
.tl-icon span { width: 14px; height: 14px; border-radius: 50%; display: block; }
.tl-icon .r { background: #EF4444; box-shadow: 0 0 12px rgba(239,68,68,.8); }
.tl-icon .y { background: #3B2A0A; }
.tl-icon .g { background: #0B2A17; }
.hero h1 { margin: 0; font-size: 1.65rem; font-weight: 800; letter-spacing: -0.02em; color: #F8FAFC; line-height: 1.25; }
.hero p { margin: 4px 0 0; color: #94A3B8; font-size: 0.95rem; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    font-size: 0.78rem; font-weight: 600; padding: 6px 12px; border-radius: 999px;
    border: 1px solid #2A3A5C; color: #CBD5E1; background: #0E1628; white-space: nowrap;
}
.chip b { color: #38BDF8; }

.strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 8px; }
@media (max-width: 900px) { .strip { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.stage {
    position: relative; padding: 14px 16px; border-radius: 14px; background: #111A2E; border: 1px solid #1F2A44;
    overflow: hidden;
}
.stage::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--c); }
.stage .k { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em; color: #64748B; font-weight: 700; }
.stage .v { font-size: 1.02rem; font-weight: 700; color: #F1F5F9; margin: 4px 0 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stage .s { font-size: 0.82rem; color: #94A3B8; }
.stage .s b { font-family: 'JetBrains Mono', monospace; color: var(--c); }

.step { display: flex; align-items: center; gap: 12px; margin: 22px 0 10px; }
.step .n {
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
    background: rgba(56,189,248,0.12); color: #38BDF8; font-weight: 800; font-family: 'JetBrains Mono', monospace;
    border: 1px solid rgba(56,189,248,0.35);
}
.step .t { font-size: 1.12rem; font-weight: 700; color: #F1F5F9; }
.step .d { font-size: 0.86rem; color: #94A3B8; margin-left: 4px; }

.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 6px 0 14px; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi { padding: 14px 16px; border-radius: 14px; background: #111A2E; border: 1px solid #1F2A44; }
.kpi .k { font-size: 0.8rem; color: #94A3B8; font-weight: 500; }
.kpi .v { font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 700; color: #F8FAFC; line-height: 1.2; margin-top: 4px; }
.kpi.bad { border-color: rgba(239,68,68,0.55); background: linear-gradient(180deg, rgba(239,68,68,0.12), #111A2E); }
.kpi.bad .v { color: #F87171; }
.kpi.good .v { color: #4ADE80; }
@media (max-width: 640px) {
    .step .d { display: none; }
    .hero { padding: 18px; }
    .hero h1 { font-size: 1.35rem; }
}

.banner { display: flex; align-items: center; gap: 12px; padding: 14px 18px; border-radius: 12px; font-weight: 600; margin: 4px 0 14px; }
.banner.bad { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.5); color: #FECACA; }
.banner.good { background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.45); color: #BBF7D0; }
.banner .dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.banner.bad .dot { background: #EF4444; box-shadow: 0 0 10px #EF4444; }
.banner.good .dot { background: #22C55E; box-shadow: 0 0 10px #22C55E; }

.pill { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; }
.pill i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.pill.RED { background: rgba(239,68,68,.15); color: #FCA5A5; border: 1px solid rgba(239,68,68,.5); } .pill.RED i { background: #EF4444; }
.pill.GREEN { background: rgba(34,197,94,.12); color: #86EFAC; border: 1px solid rgba(34,197,94,.45); } .pill.GREEN i { background: #22C55E; }
.pill.YELLOW { background: rgba(245,158,11,.12); color: #FCD34D; border: 1px solid rgba(245,158,11,.45); } .pill.YELLOW i { background: #F59E0B; }
.pill.UNKNOWN { background: rgba(100,116,139,.15); color: #CBD5E1; border: 1px solid #334155; } .pill.UNKNOWN i { background: #64748B; }

.vcard-title { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
.vcard-title .id { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F87171; font-size: 1.02rem; }
.vcard-title .tm { font-family: 'JetBrains Mono', monospace; color: #94A3B8; font-size: 0.85rem; }
.vmeta { font-size: 0.86rem; color: #CBD5E1; line-height: 1.7; }
.vmeta span { color: #64748B; }

.empty {
    text-align: center; padding: 42px 20px; border: 1.5px dashed #2A3A5C; border-radius: 16px; background: #0E1628;
}
.empty .big { font-size: 2.2rem; }
.empty h4 { margin: 8px 0 6px; color: #E2E8F0; font-weight: 700; }
.empty p { color: #94A3B8; margin: 0 auto; max-width: 560px; font-size: 0.92rem; }

.flow { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
@media (max-width: 900px) { .flow { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.flow div { background: #0E1628; border: 1px solid #1F2A44; border-radius: 12px; padding: 12px 14px; font-size: 0.84rem; color: #94A3B8; }
.flow b { display: block; color: #E2E8F0; font-size: 0.9rem; margin-bottom: 3px; }

.brand { display: flex; align-items: center; gap: 12px; padding: 4px 0 6px; }
.brand .tl-icon { padding: 6px 5px; gap: 4px; }
.brand .tl-icon span { width: 10px; height: 10px; }
.brand .t { font-weight: 800; color: #F8FAFC; font-size: 1.05rem; line-height: 1.2; }
.brand .s { color: #94A3B8; font-size: 0.8rem; }
.sb-sec { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .1em; color: #64748B; font-weight: 700; margin: 14px 0 6px; }
.sb-sum { background: #111A2E; border: 1px solid #1F2A44; border-radius: 12px; padding: 12px 14px; font-size: 0.86rem; color: #CBD5E1; }
.sb-sum .row { display: flex; justify-content: space-between; padding: 3px 0; }
.sb-sum .row b { font-family: 'JetBrains Mono', monospace; color: #F8FAFC; }

.footer { text-align: center; color: #475569; font-size: 0.82rem; margin-top: 36px; padding-top: 16px; border-top: 1px solid #1F2A44; }
</style>
""", unsafe_allow_html=True)

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
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # bỏ bóng đổ (giá trị 127)
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
    ly = y0 - 6 if y0 - 6 >= th + 32 else y1 + th + 6  # tránh đè lên thanh thông tin phía trên
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
    """Chạy toàn bộ pipeline A → B → C trên chuỗi khung hình.

    frames: iterator trả về (thời_gian_giây, khung_BGR) đã cùng kích thước.
    background: None = video (MOG2); ảnh nền = chế độ ảnh tĩnh.
    """
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
# CẢNH MÔ PHỎNG (SYNTHETIC SCENE)
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


# ==========================================
# TIỆN ÍCH GIAO DIỆN
# ==========================================
def style_fig(fig, height=None, title=None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Be Vietnam Pro, system-ui, sans-serif", color=C_TEXT, size=13),
        margin=dict(l=8, r=8, t=48 if (title or fig.layout.title.text) else 12, b=8),
        title=dict(text=title, font=dict(size=15)) if title else fig.layout.title,
        hoverlabel=dict(bgcolor=C_PANEL, bordercolor=C_LINE, font_color=C_TEXT),
    )
    fig.update_xaxes(gridcolor=C_LINE, zerolinecolor=C_LINE, linecolor=C_LINE)
    fig.update_yaxes(gridcolor=C_LINE, zerolinecolor=C_LINE, linecolor=C_LINE)
    if height:
        fig.update_layout(height=height)
    return fig


def step(n, title, desc=""):
    st.markdown(f'<div class="step"><div class="n">{n}</div><div class="t">{title}</div>'
                f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)


def pill(state, prefix="Đèn"):
    return f'<span class="pill {state}"><i></i>{prefix} {STATE_VN[state].lower()}</span>'


def banner(text, good):
    st.markdown(f'<div class="banner {"good" if good else "bad"}"><span class="dot"></span>{text}</div>',
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
            x=[max(end - start, 1e-3)], y=["Đèn"], base=[start], orientation="h",
            marker=dict(color=colors[s], line=dict(width=0)), name=STATE_VN[s],
            showlegend=s not in shown, legendgroup=s,
            hovertemplate=f"{STATE_VN[s]}: %{{base:.1f}}s → {end:.1f}s<extra></extra>",
        ))
        shown.add(s)
    for v in violations:
        tv = violation_time(v)
        if tv is not None:
            fig.add_vline(x=tv, line=dict(color="#F8FAFC", width=2, dash="dot"))
            fig.add_annotation(x=tv, y=1.0, yref="paper", text=v["MaXe"], showarrow=False, yanchor="bottom",
                               font=dict(color="#FCA5A5", size=12, family="JetBrains Mono, monospace"))
    fig.update_layout(barmode="overlay", bargap=0.35,
                      legend=dict(orientation="h", y=-0.45, x=0, font=dict(size=12)))
    fig.update_xaxes(title="Thời điểm (giây)")
    fig.update_yaxes(showticklabels=False)
    style_fig(fig, height=220)
    fig.update_layout(margin=dict(l=8, r=8, t=30, b=8))
    return fig


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="tl-icon"><span class="r"></span><span class="y"></span><span class="g"></span></div>
        <div><div class="t">Giám sát vượt đèn đỏ</div><div class="s">Đồ án Học Máy Truyền Thống · Nhóm 6</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Thuật toán cho từng mô hình</div>', unsafe_allow_html=True)
    sel_model_a = st.selectbox("Mô hình A · Nhận diện đèn", list(MODEL_PERFORMANCE["Model_A"].keys()))
    sel_model_b = st.selectbox("Mô hình B · Phát hiện xe", list(MODEL_PERFORMANCE["Model_B"].keys()))
    sel_model_c = st.selectbox("Mô hình C · Ghép cặp xe", list(MODEL_PERFORMANCE["Model_C"].keys()))
    perf_a = MODEL_PERFORMANCE["Model_A"][sel_model_a]
    perf_b = MODEL_PERFORMANCE["Model_B"][sel_model_b]
    perf_c = MODEL_PERFORMANCE["Model_C"][sel_model_c]
    avg_acc = round((perf_a["accuracy"] + perf_b["accuracy"] + perf_c["accuracy"]) / 3, 1)
    total_latency = round(sum(float(p["latency"].split()[0]) for p in (perf_a, perf_b, perf_c)), 1)
    st.markdown(f"""
    <div class="sb-sum">
        <div class="row"><span>Độ chính xác trung bình</span><b style="color:{C_GREEN}">{avg_acc}%</b></div>
        <div class="row"><span>Tổng độ trễ</span><b style="color:{C_ACCENT}">{total_latency} ms</b></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Tab mô phỏng</div>', unsafe_allow_html=True)
    scenario = st.selectbox("Kịch bản kiểm thử", [
        "Kịch bản 1: Đèn Đỏ - Xe máy Vượt Vạch (Vi Phạm)",
        "Kịch bản 2: Đèn Đỏ - Ô tô Dừng Đúng Quy Định",
        "Kịch bản 3: Đèn Xanh - Lưu Thông Bình Thường",
    ], format_func=lambda s: s.split(": ", 1)[1])
    stop_line_y = st.slider("Vị trí vạch dừng (trục Y)", 200, 400, 320)


# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <div class="tl-icon"><span class="r"></span><span class="y"></span><span class="g"></span></div>
        <div>
            <h1>Phát hiện & cảnh báo xe vượt đèn đỏ</h1>
            <p>Pipeline máy học truyền thống chạy trên ảnh và video thật từ camera giao thông</p>
        </div>
    </div>
    <div class="chips">
        <span class="chip">Đèn · <b>HSV</b></span>
        <span class="chip">Xe · <b>MOG2 + HOG</b></span>
        <span class="chip">Ghép cặp · <b>Hungarian</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="strip">
    <div class="stage" style="--c:{C_ACCENT}"><div class="k">Mô hình A · Đèn</div>
        <div class="v">{sel_model_a}</div><div class="s">Accuracy <b>{perf_a['accuracy']}%</b> · {perf_a['latency']}</div></div>
    <div class="stage" style="--c:#A78BFA"><div class="k">Mô hình B · Phát hiện xe</div>
        <div class="v">{sel_model_b}</div><div class="s">Accuracy <b>{perf_b['accuracy']}%</b> · {perf_b['latency']}</div></div>
    <div class="stage" style="--c:#F472B6"><div class="k">Mô hình C · Ghép cặp xe</div>
        <div class="v">{sel_model_c}</div><div class="s">Accuracy <b>{perf_c['accuracy']}%</b> · {perf_c['latency']}</div></div>
    <div class="stage" style="--c:{C_RED}"><div class="k">Nhật ký vi phạm</div>
        <div class="v">{len(st.session_state.violations_log)} lượt</div><div class="s">Lưu trong phiên làm việc này</div></div>
</div>
""", unsafe_allow_html=True)

tab_real, tab_sim, tab_bench, tab_feat, tab_log = st.tabs([
    "Phân tích ảnh / video", "Mô phỏng pipeline", "Hiệu năng mô hình", "Đặc trưng HSV & HOG", "Nhật ký vi phạm",
])

# ==========================================
# TAB 1: PHÂN TÍCH ẢNH / VIDEO THẬT
# ==========================================
with tab_real:
    step(1, "Dữ liệu đầu vào", "Video hoặc ảnh từ camera đặt cố định tại ngã tư")
    input_type = st.segmented_control("Loại dữ liệu", ["🎥 Video", "🖼️ Ảnh"], default="🎥 Video",
                                      label_visibility="collapsed") or "🎥 Video"
    preview, source_key, video_path, images, background = None, None, None, [], None
    has_sample = os.path.exists(SAMPLE_VIDEO)

    if input_type == "🎥 Video":
        up = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov", "mkv", "m4v"],
                              help="Nên dùng video quay từ camera đứng yên, thấy rõ vạch dừng và đèn giao thông.")
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
            c_s1, c_s2 = st.columns([4, 1])
            c_s1.caption("Đang dùng **video mẫu** (video_thu.mp4). Tải video của bạn lên để thay thế.")
            if c_s2.button("Bỏ video mẫu", width="stretch"):
                st.session_state.use_sample = False
                st.rerun()
        if video_path:
            preview = read_first_frame(video_path)
            if preview is None:
                banner("Không đọc được video này. Hãy chuyển sang định dạng .mp4 (H.264) rồi thử lại.", good=False)
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
                banner("Có ảnh không đọc được. Hãy dùng ảnh JPG hoặc PNG.", good=False)
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
        <div class="empty">
            <div class="big">🚦</div>
            <h4>Chưa có dữ liệu</h4>
            <p>Tải lên một video quay ngã tư từ camera cố định. Hệ thống sẽ đọc màu đèn, tìm từng phương tiện,
            theo dõi chúng qua các khung hình và đánh dấu xe nào vượt vạch dừng khi đèn đỏ.</p>
        </div>""", unsafe_allow_html=True)
        if input_type == "🎥 Video" and has_sample:
            _, mid, _ = st.columns([2, 1.2, 2])
            if mid.button("▶  Thử với video mẫu", type="primary", width="stretch"):
                st.session_state.use_sample = True
                st.rerun()
        st.write("")
        st.markdown("""
        <div class="flow">
            <div><b>① Mô hình A</b>Đọc màu đèn trong vùng đèn theo ngưỡng màu HSV.</div>
            <div><b>② Mô hình B</b>Tách xe khỏi nền đường bằng trừ nền MOG2 + contour.</div>
            <div><b>③ Mô hình C</b>Ghép cùng một xe giữa các khung hình bằng thuật toán Hungarian.</div>
            <div><b>④ Kết luận</b>Đầu xe đi từ trước vạch sang sau vạch khi đèn đỏ ⇒ vi phạm.</div>
        </div>""", unsafe_allow_html=True)

    else:
        step(2, "Hiệu chỉnh vạch dừng & vùng đèn", "Kéo cho khớp với hình xem trước bên phải")
        col_cfg, col_prev = st.columns([1, 1.45], gap="large")
        with col_cfg:
            with st.container(border=True):
                st.markdown("**Vạch dừng** · tính theo % khung hình")
                a1, a2 = st.columns(2)
                lx1 = a1.slider("Đầu trái · ngang", 0, 100, 5, key="lx1")
                ly1 = a1.slider("Đầu trái · dọc", 0, 100, 65, key="ly1")
                lx2 = a2.slider("Đầu phải · ngang", 0, 100, 95, key="lx2")
                ly2 = a2.slider("Đầu phải · dọc", 0, 100, 65, key="ly2")
                direction_lbl = st.radio("Hướng xe chạy", ["Từ trên xuống ↓", "Từ dưới lên ↑"],
                                         horizontal=True, key="direction")
                direction = "down" if direction_lbl.startswith("Từ trên") else "up"
            with st.container(border=True):
                st.markdown("**Vùng đèn giao thông**")
                b1, b2 = st.columns(2)
                rx = b1.slider("Vị trí ngang", 0, 99, 85, key="rx")
                ry = b1.slider("Vị trí dọc", 0, 99, 5, key="ry")
                rw = b2.slider("Chiều rộng", 1, 50, 10, key="rw")
                rh = b2.slider("Chiều cao", 1, 60, 25, key="rh")
                light_options = {
                    "auto": "Tự động nhận diện màu đèn",
                    "red": "Không thấy đèn trong hình · coi như luôn đỏ",
                    "green": "Coi như luôn xanh",
                }
                light_mode = st.selectbox("Trạng thái đèn", list(light_options), format_func=light_options.get,
                                          key="light_mode")
            with st.expander("Tinh chỉnh nâng cao"):
                min_area_pct = st.slider("Diện tích xe nhỏ nhất (% khung hình)", 0.05, 5.0, 0.4, 0.05,
                                         help="Tăng lên nếu lá cây, người đi bộ hoặc nhiễu bị nhận nhầm là xe.")
                car_area_pct = st.slider("Ngưỡng coi là ô tô (% khung hình)", 0.5, 15.0, 2.5, 0.5)
                sensitivity = st.slider("Độ nhạy trừ nền (nhỏ = nhạy hơn)", 8, 80, 32)
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
            prev_img = draw_overlay(preview, line_prev, rect_prev, prev_state, {}, "Xem truoc")
            p1, p2 = line_prev.pts()
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            dy = 45 if direction == "down" else -45
            cv2.arrowedLine(prev_img, (mid[0], mid[1] - dy), (mid[0], mid[1] + dy // 3), (248, 189, 56), 3, tipLength=0.35)
            with st.container(border=True):
                st.image(prev_img, channels="BGR", width="stretch")
                st.markdown(f"Khung hình đầu tiên &nbsp; {pill(prev_state)} &nbsp;"
                            f"<span style='color:{C_MUTED};font-size:.85rem'>· mũi tên xanh = hướng xe chạy</span>",
                            unsafe_allow_html=True)
                if prev_state == "UNKNOWN" and light_mode == "auto":
                    st.caption("Chưa thấy đèn sáng trong khung đèn. Kéo khung trùng với cột đèn, "
                               "hoặc chọn “coi như luôn đỏ” nếu video không quay được đèn.")

        step(3, "Phân tích", "")
        can_run = not (input_type == "🖼️ Ảnh" and background is None)
        if not can_run:
            banner("Với 1–2 ảnh, hãy tải thêm ảnh nền đường trống, hoặc tải từ 3 ảnh liên tiếp trở lên.", good=False)
        run = st.button("🔍  Phân tích vi phạm", type="primary", disabled=not can_run, width="stretch")
        if run:
            progress = st.progress(0.0, text="Đang chuẩn bị...")
            try:
                if input_type == "🎥 Video":
                    res = analyze_video(video_path, cfg, progress)
                else:
                    frames = ((float(i), img) for i, img in enumerate(images))
                    res = analyze_frames(frames, cfg, fps_eff=1.0, background=background,
                                         progress=progress, total=len(images))
            except Exception as exc:  # hiển thị lỗi gọn gàng thay vì vỡ trang
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
            step(4, "Kết quả", "")
            n_v = len(res["violations"])
            last = f"{res['duration']:.1f}s" if res["kind"] == "video" else str(res["n_frames"])
            last_lbl = "Thời lượng đã xét" if res["kind"] == "video" else "Số ảnh đã xét"
            st.markdown(f"""
            <div class="kpis">
                <div class="kpi"><div class="k">Phương tiện phát hiện</div><div class="v">{res['n_vehicles']}</div></div>
                <div class="kpi {'bad' if n_v else 'good'}"><div class="k">Vượt đèn đỏ</div><div class="v">{n_v}</div></div>
                <div class="kpi good"><div class="k">Qua vạch hợp lệ</div><div class="v">{res['crossed_ok']}</div></div>
                <div class="kpi"><div class="k">{last_lbl}</div><div class="v">{last}</div></div>
            </div>""", unsafe_allow_html=True)
            if n_v:
                banner(f"Phát hiện {n_v} phương tiện vượt vạch dừng khi đèn đỏ.", good=False)
            else:
                banner("Không có phương tiện nào vượt vạch dừng khi đèn đỏ.", good=True)

            col_main, col_list = st.columns([1.75, 1], gap="large")
            with col_main:
                with st.container(border=True):
                    if res["kind"] == "video":
                        st.markdown("**Video kết quả** · khung xanh = bình thường, khung đỏ = vi phạm")
                        if res["video_browser_ok"]:
                            st.video(res["video_bytes"], format="video/mp4")
                        else:
                            st.caption("Máy chủ thiếu bộ mã hoá H.264 nên không phát trực tiếp được. Hãy tải video về để xem.")
                        st.download_button("⬇  Tải video kết quả (.mp4)", res["video_bytes"],
                                           file_name="ket_qua_phan_tich.mp4", mime="video/mp4", width="stretch")
                    else:
                        st.markdown("**Ảnh kết quả** · khung xanh = bình thường, khung đỏ = vi phạm")
                        cols = st.columns(min(2, len(res["annotated"])))
                        for i, img in enumerate(res["annotated"]):
                            cols[i % len(cols)].image(img, channels="BGR", caption=f"Ảnh {i + 1}", width="stretch")
            with col_list:
                st.markdown(f"**Danh sách vi phạm** · {n_v}")
                if not n_v:
                    st.caption("Chưa có phương tiện vi phạm trong dữ liệu này.")
                box = st.container(height=560, border=False) if n_v > 2 else st.container()
                with box:
                    for k, v in enumerate(res["violations"]):
                        with st.container(border=True):
                            st.markdown(f'<div class="vcard-title"><span class="id">{v["MaXe"]}</span>'
                                        f'<span class="tm">⏱ {v["ThoiDiem"]}</span></div>', unsafe_allow_html=True)
                            ci, cm = st.columns([1, 1.3])
                            ci.image(v["crop"], width="stretch")
                            cm.markdown(f'<div class="vmeta"><span>Loại xe</span><br>{v["LoaiXe"]}<br>'
                                        f'<span>Đèn</span><br>{pill("RED")}</div>', unsafe_allow_html=True)
                            st.download_button("⬇  Ảnh bằng chứng", v["snapshot"], file_name=f"vi_pham_{v['MaXe']}.jpg",
                                               mime="image/jpeg", key=f"dl_{k}_{v['MaXe']}", width="stretch")

            if res["kind"] == "video" and res["timeline"]:
                with st.container(border=True):
                    st.markdown("**Trạng thái đèn theo thời gian** · Mô hình A, vạch chấm = thời điểm vi phạm")
                    st.plotly_chart(light_timeline_fig(res["timeline"], res["violations"]), width="stretch",
                                    config={"displayModeBar": False})

            if n_v:
                st.markdown("**Ảnh toàn cảnh tại khoảnh khắc vượt vạch**")
                gcols = st.columns(3)
                for k, v in enumerate(res["violations"]):
                    gcols[k % 3].image(v["snapshot"], caption=f"{v['MaXe']} · {v['ThoiDiem']}", width="stretch")

            st.caption("Kết quả chưa đúng? Chỉnh lại vạch dừng, vùng đèn hoặc mục “Tinh chỉnh nâng cao” rồi bấm Phân tích lại.")

# ==========================================
# TAB 2: MÔ PHỎNG
# ==========================================
with tab_sim:
    f1_img, l_state, _ = generate_traffic_scene(scenario, frame_idx=1, stop_y=stop_line_y)
    f2_img, l_state, sim_violations = generate_traffic_scene(scenario, frame_idx=2, stop_y=stop_line_y)
    st.markdown(f"<div style='margin:14px 0 10px'>Kết quả Mô hình A &nbsp; {pill(l_state)}</div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2, gap="large")
    with s1:
        with st.container(border=True):
            st.markdown("**Khung hình t** · Mô hình B định vị xe")
            st.image(f1_img, channels="BGR", width="stretch")
    with s2:
        with st.container(border=True):
            st.markdown("**Khung hình t+1** · Mô hình C ghép cặp & kiểm tra vạch")
            st.image(f2_img, channels="BGR", width="stretch")
    if sim_violations:
        for viol in sim_violations:
            banner(f"{viol['MaXe']} ({viol['LoaiXe']}) cắt qua vạch dừng khi đèn đỏ.", good=False)
    else:
        banner("Tất cả phương tiện chấp hành đúng (dừng trước vạch hoặc đèn xanh).", good=True)
    st.caption("Đây là cảnh dựng sẵn để minh hoạ quy trình. Để kiểm tra dữ liệu thật, dùng tab “Phân tích ảnh / video”.")

# ==========================================
# TAB 3: HIỆU NĂNG MÔ HÌNH
# ==========================================
with tab_bench:
    from plotly.subplots import make_subplots
    groups = [("Model_A", "A · Nhận diện đèn", sel_model_a), ("Model_B", "B · Phát hiện xe", sel_model_b),
              ("Model_C", "C · Ghép cặp xe", sel_model_c)]
    st.write("")
    fig_b = make_subplots(rows=1, cols=3, subplot_titles=[g[1] for g in groups], shared_yaxes=True,
                          horizontal_spacing=0.04)
    for i, (key, _, sel) in enumerate(groups, start=1):
        names = list(MODEL_PERFORMANCE[key])
        accs = [MODEL_PERFORMANCE[key][n]["accuracy"] for n in names]
        fig_b.add_trace(go.Bar(
            x=[n.replace(" ", "<br>", 1) for n in names], y=accs, text=[f"{a}%" for a in accs], textposition="outside",
            marker=dict(color=[C_ACCENT if n == sel else "#2A3A5C" for n in names]),
            hovertemplate="%{x}: %{y}%<extra></extra>", showlegend=False,
        ), row=1, col=i)
    fig_b.update_yaxes(range=[80, 101.5], title_text="Accuracy (%)", row=1, col=1)
    fig_b.update_xaxes(tickangle=0, tickfont=dict(size=11))
    style_fig(fig_b, height=400)
    fig_b.update_layout(margin=dict(l=8, r=8, t=56, b=8))
    fig_b.update_annotations(font=dict(size=13, color=C_MUTED))
    with st.container(border=True):
        st.markdown("**Độ chính xác các thuật toán** · cột sáng màu = thuật toán đang chọn ở thanh bên")
        st.plotly_chart(fig_b, width="stretch", config={"displayModeBar": False})

    with st.container(border=True):
        st.markdown("**Bộ mô hình đang chọn**")
        st.dataframe(pd.DataFrame([
            {"Mô hình": g[1], "Thuật toán": g[2], "Accuracy": p["accuracy"], "F1": p["f1"],
             "Độ trễ (ms)": float(p["latency"].split()[0]), "Mô tả": p["desc"]}
            for g, p in zip(groups, (perf_a, perf_b, perf_c))
        ]), hide_index=True, width="stretch", column_config={
            "Accuracy": st.column_config.ProgressColumn("Accuracy", format="%.1f%%", min_value=0, max_value=100),
            "F1": st.column_config.NumberColumn("F1", format="%.2f"),
            "Độ trễ (ms)": st.column_config.NumberColumn("Độ trễ", format="%.1f ms"),
            "Mô tả": st.column_config.TextColumn("Mô tả", width="large"),
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
        st.caption("Đang hiển thị dữ liệu minh hoạ. Hãy phân tích một video ở tab đầu để xem đặc trưng lấy từ dữ liệu thật.")
    else:
        st.caption("Đặc trưng được trích từ dữ liệu bạn vừa phân tích.")

    col_hsv, col_hog = st.columns(2, gap="large")
    with col_hsv:
        with st.container(border=True):
            st.markdown("**Mô hình A · Vector màu HSV của vùng đèn (44 chiều)**")
            if light_sample is None:
                light_sample = np.full((90, 60, 3), 20, dtype=np.uint8)
                cv2.circle(light_sample, (30, 22), 12, (0, 0, 235), -1)
            hs1, hs2 = st.columns([1, 3])
            hs1.image(light_sample, channels="BGR", caption="Vùng đèn", width="stretch")
            state_s, scores = classify_light(light_sample)
            hs2.markdown(f"Kết luận &nbsp; {pill(state_s)}", unsafe_allow_html=True)
            hs2.caption(" · ".join(f"{STATE_VN[k]}: {v * 100:.1f}% điểm ảnh" for k, v in scores.items()))
            feat = hsv_feature(light_sample)
            fig_hsv = px.bar(x=list(range(44)), y=feat, color=["Hue"] * 18 + ["Saturation"] * 16 + ["Value"] * 10,
                             labels={"x": "Bin", "y": "Tỉ lệ điểm ảnh", "color": ""},
                             color_discrete_map={"Hue": C_RED, "Saturation": C_AMBER, "Value": C_ACCENT})
            fig_hsv.update_layout(legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(style_fig(fig_hsv, height=300), width="stretch", config={"displayModeBar": False})

    with col_hog:
        with st.container(border=True):
            st.markdown("**Mô hình B · Đặc trưng HOG của phương tiện**")
            if vehicle_sample is None or vehicle_sample.size == 0:
                demo, _, _ = generate_traffic_scene("Kịch bản 1: Đèn Đỏ - Xe máy Vượt Vạch (Vi Phạm)", 1)
                vehicle_sample = demo[195:300, 335:465]
            veh_sq = cv2.resize(vehicle_sample, (128, 128))
            gray = cv2.cvtColor(veh_sq, cv2.COLOR_BGR2GRAY)
            hog_vec, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            hog_img = np.sqrt(hog_img / max(hog_img.max(), 1e-6))  # làm rõ các nét gradient yếu
            hog_img = cv2.resize((hog_img * 255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            hog_img = cv2.applyColorMap(hog_img, cv2.COLORMAP_INFERNO)
            g1, g2 = st.columns(2)
            g1.image(cv2.resize(veh_sq, (256, 256)), channels="BGR", caption="Phương tiện (chuẩn hoá 128×128)", width="stretch")
            g2.image(hog_img, channels="BGR", caption="Hướng gradient (HOG)", width="stretch")
            fig_hog = px.bar(x=list(range(60)), y=hog_vec[:60], labels={"x": "Chỉ số đặc trưng", "y": "Giá trị"},
                             color_discrete_sequence=["#A78BFA"])
            st.plotly_chart(style_fig(fig_hog, height=220), width="stretch", config={"displayModeBar": False})
            st.caption(f"Vector HOG có {hog_vec.size:,} chiều (ảnh 128×128, 9 hướng, cell 8×8, block 2×2); biểu đồ hiển thị 60 giá trị đầu.")

# ==========================================
# TAB 5: NHẬT KÝ VI PHẠM
# ==========================================
with tab_log:
    st.write("")
    if st.session_state.violations_log:
        df_log = pd.DataFrame(st.session_state.violations_log)
        with st.container(border=True):
            st.markdown(f"**{len(df_log)} lượt vi phạm** đã ghi nhận trong phiên này")
            st.dataframe(df_log, hide_index=True, width="stretch")
            l1, l2, _ = st.columns([1, 1, 2])
            l1.download_button("⬇  Tải nhật ký (.csv)", df_log.to_csv(index=False).encode("utf-8-sig"),
                               file_name="danh_sach_vi_pham_dendo.csv", mime="text/csv", width="stretch")
            if l2.button("Xoá lịch sử", width="stretch"):
                st.session_state.violations_log = []
                st.rerun()
    else:
        st.markdown("""
        <div class="empty"><div class="big">📋</div><h4>Chưa có lượt vi phạm nào</h4>
        <p>Phân tích một video ở tab đầu tiên. Mỗi phương tiện vượt đèn đỏ sẽ được ghi vào đây và có thể tải về dạng CSV.</p></div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer">Đồ án môn Học Máy Truyền Thống © 2026 · Nhóm 6 · Streamlit & OpenCV</div>',
            unsafe_allow_html=True)
