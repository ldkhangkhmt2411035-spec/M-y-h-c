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

try:  # cho phép bấm chuột lên ảnh để chọn đèn / vạch dừng
    from streamlit_image_coordinates import streamlit_image_coordinates as image_picker
except ImportError:
    image_picker = None

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
.stButton > button:disabled, .stButton > button[kind="primary"]:disabled {
    background: #16213A; color: #64748B; box-shadow: none; border: 1px dashed #2A3A5C; cursor: not-allowed;
}

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
.banner.warn { background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.5); color: #FDE68A; font-weight: 500; }
.banner.warn .dot { background: #F59E0B; box-shadow: 0 0 10px #F59E0B; }

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
        """Điểm tiếp đất gần camera nhất (giữa cạnh dưới của khung).

        Khi trừ nền làm dính khung xe với xe khác, phần dính thừa luôn nằm ở phía trên khung (xa camera hơn),
        nên cạnh dưới là điểm ổn định nhất của chính chiếc xe gần camera.
        - Xe chạy xuống (về phía camera): cạnh dưới = đầu xe.
        - Xe chạy lên (ra xa camera): cạnh dưới = đuôi xe ⇒ tính vi phạm khi cả thân xe đã qua vạch.
        """
        x, y, w, h = box
        return (x + w / 2, y + h)

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
        "RED": lit & ((H <= 14) | (H >= 165)),
        "YELLOW": lit & (H >= 15) & (H <= 35),
        "GREEN": lit & (H >= 40) & (H <= 95),
    }
    total = float(H.size)
    scores = {k: float(m.sum()) / total for k, m in masks.items()}
    best = max(scores, key=scores.get)
    h_, w_ = H.shape
    if scores[best] >= 0.004 and best != "GREEN" and h_ >= 1.8 * w_:
        # Khung ôm cả hộp đèn DỌC: đèn vàng ngoài đời có màu cam, sắc độ sát với đỏ ⇒ dùng VỊ TRÍ bóng sáng
        # (trên = đỏ, giữa = vàng) để phân biệt. Màu xanh thì không nhầm nên giữ theo màu.
        warm = masks["RED"] | masks["YELLOW"]
        ys = np.nonzero(warm)[0]
        if ys.size:
            pos = float(np.median(ys)) / h_
            return ("RED" if pos < 0.36 else "YELLOW" if pos < 0.64 else best), scores
    if scores[best] >= 0.004:
        return best, scores
    # Không rõ màu (bóng cháy sáng thành trắng): khung ôm cả hộp đèn ⇒ suy ra theo VỊ TRÍ bóng đang sáng.
    # Đèn dọc: trên = đỏ, giữa = vàng, dưới = xanh. Đèn ngang: trái = đỏ, giữa = vàng, phải = xanh.
    h, w = V.shape
    if max(h, w) >= 1.8 * min(h, w) and min(h, w) >= 4:
        bright = V >= 225
        axis_len = h if h >= w else w
        parts = [bright[i * axis_len // 3:(i + 1) * axis_len // 3] if h >= w else
                 bright[:, i * axis_len // 3:(i + 1) * axis_len // 3] for i in range(3)]
        counts = [int(p.sum()) for p in parts]
        k = int(np.argmax(counts))
        others = sorted(counts)[1]
        if counts[k] >= 0.01 * total and counts[k] >= 2 * max(others, 1):
            return ["RED", "YELLOW", "GREEN"][k], scores
    return "UNKNOWN", scores


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
            self.n_unknown = 0
        else:
            self.n_unknown = getattr(self, "n_unknown", 0) + 1
            if self.n_unknown > 3 * self.hist.maxlen:  # lâu không thấy đèn sáng ⇒ không còn biết đèn màu gì
                self.hist.clear()
                self.last = "UNKNOWN"
        if self.hist:
            self.last = Counter(self.hist).most_common(1)[0][0]
        return self.last


# ---------- Mô hình B: phát hiện xe (trừ nền + contour) ----------
def mask_to_boxes(fg, min_area, ignore_rect=None, max_area=None):
    """Lọc nhiễu hình thái học rồi lấy khung bao quanh từng vùng chuyển động.

    Bỏ các vùng quá nhỏ (nhiễu, lá cây) và quá lớn (cả đoàn xe dính nhau, camera rung, chuyển cảnh).
    """
    k = max(7, (fg.shape[1] // 80) | 1)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
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
        if w * h < min_area or cv2.contourArea(c) < 0.3 * min_area:
            continue
        if max_area is not None and w * h > max_area:
            continue
        boxes.append((x, y, w, h))
    return boxes, fg


class VideoVehicleDetector:
    """Trừ nền thích nghi MOG2 (Gaussian Mixture Model) cho camera cố định."""

    def __init__(self, sensitivity):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=sensitivity, detectShadows=True)

    def detect(self, frame, min_area, ignore_rect=None, max_area=None):
        """Trả về (danh sách khung xe, tỉ lệ điểm ảnh chuyển động trên toàn khung hình)."""
        fg = self.bg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
        # MOG2 đánh dấu "bóng đổ" = 127. Xe màu tối (ô tô đen, áo mưa sẫm) cũng hay bị coi là bóng đổ,
        # nên giữ lại cả vùng này; bóng thật dính liền với xe chỉ làm khung rộng ra một chút.
        _, fg = cv2.threshold(fg, 100, 255, cv2.THRESH_BINARY)
        ratio = float(np.count_nonzero(fg)) / fg.size
        boxes, _ = mask_to_boxes(fg, min_area, ignore_rect, max_area)
        return boxes, ratio


def detect_by_background(frame, background, min_area, sensitivity, ignore_rect=None, max_area=None):
    """Dùng cho ảnh tĩnh: so sánh với ảnh nền (đường trống)."""
    a = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(background, (5, 5), 0).astype(np.int16)
    diff = np.abs(a - b).max(axis=2).astype(np.uint8)
    _, fg = cv2.threshold(diff, max(15, sensitivity + 5), 255, cv2.THRESH_BINARY)
    ratio = float(np.count_nonzero(fg)) / fg.size
    boxes, _ = mask_to_boxes(fg, min_area, ignore_rect, max_area)
    return boxes, ratio


def snap_light_roi(frame, px, py, layout="auto"):
    """Người dùng bấm vào đèn → tìm đốm sáng màu (đỏ/vàng/xanh) gần điểm bấm nhất và ôm khung quanh nó.

    Trả về (x, y, rộng, cao) theo % khung hình.
    """
    H, W = frame.shape[:2]
    r = int(0.07 * W)
    x0, y0, x1, y1 = max(0, px - r), max(0, py - r), min(W, px + r), min(H, py + r)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = (S >= 90) & (V >= 140) & ((Hh <= 14) | (Hh >= 160) | ((Hh >= 15) & (Hh <= 95)))
    overexposed = (V >= 235) & (S < 80)  # tâm bóng đèn bị cháy sáng thành màu trắng
    best = None
    # xét riêng đốm có màu và đốm cháy trắng (để vùng trời trắng không dính vào bóng đèn)
    for mask in (colored, overexposed, colored | overexposed):
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for k in range(1, n):
            bw_, bh_ = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            # đèn là một đốm sáng nhỏ, gọn: bỏ qua mảng lớn như cây xanh, bầu trời, thân xe
            if stats[k, cv2.CC_STAT_AREA] < 6 or bw_ > 0.12 * W or bh_ > 0.14 * H:
                continue
            bx_, by_ = stats[k, cv2.CC_STAT_LEFT] + x0, stats[k, cv2.CC_STAT_TOP] + y0
            # khoảng cách từ điểm bấm tới khung của đốm sáng (= 0 nếu bấm trúng bên trong)
            d = np.hypot(max(bx_ - px, 0, px - (bx_ + bw_)), max(by_ - py, 0, py - (by_ + bh_)))
            # ưu tiên đốm gần điểm bấm nhất; bằng nhau thì lấy đốm nhỏ hơn (tâm bóng đèn, không phải quầng sáng)
            key_ = (round(d / max(2.0, 0.01 * W)), stats[k, cv2.CC_STAT_AREA])
            if best is None or key_ < best[2]:
                best = (d, (bx_, by_, int(bw_), int(bh_)), key_)
    if best is None or best[0] > 0.025 * W:
        return None  # chỗ bấm không có đốm đèn sáng nào
    lamp = best[1]
    bw, bh = lamp[2], lamp[3]
    lamp_color, _ = classify_light(frame[lamp[1]:lamp[1] + bh, lamp[0]:lamp[0] + bw])
    if layout == "lamp":  # chỉ ôm bóng đèn đang sáng
        d_ = max(bw, bh)
        hx0, hy0, hx1, hy1 = lamp[0] - 0.4 * d_, lamp[1] - 0.4 * d_, lamp[0] + bw + 0.4 * d_, lamp[1] + bh + 0.4 * d_
    else:
        hx0, hy0, hx1, hy1 = expand_to_housing(frame, lamp, lamp_color,
                                               {"auto": None, "vertical": True, "horizontal": False}[layout])
    pad_x, pad_y = 0.08 * (hx1 - hx0), 0.05 * (hy1 - hy0)
    rx0, ry0 = max(0.0, (hx0 - pad_x) / W * 100), max(0.0, (hy0 - pad_y) / H * 100)
    rx1, ry1 = min(100.0, (hx1 + pad_x) / W * 100), min(100.0, (hy1 + pad_y) / H * 100)
    return int(rx0), int(ry0), max(1, int(np.ceil(rx1 - int(rx0)))), max(1, int(np.ceil(ry1 - int(ry0))))


def expand_to_housing(frame, lamp, lamp_color, force_vertical=None):
    """Từ bóng đèn đang sáng, mở rộng khung ra cả hộp đèn 3 bóng.

    Hộp đèn có hình học cố định: 3 bóng cách đều nhau, dọc (đỏ trên – vàng giữa – xanh dưới) hoặc
    ngang (đỏ trái – vàng giữa – xanh phải). Tạo các khung ứng viên theo hình học đó rồi chọn khung có
    phần vỏ màu tối nhiều nhất. Trả về (x0, y0, x1, y1) theo pixel.
    """
    H, W = frame.shape[:2]
    lx, ly, lw, lh = lamp
    cx, cy = lx + lw / 2, ly + lh / 2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2]
    # đường kính bóng: cạnh ngắn của đốm (đốm có thể dính số đếm ngược); tâm cháy trắng nhỏ hơn bóng thật
    d = float(max(min(lw, lh) if max(lw, lh) > 1.5 * min(lw, lh) else max(lw, lh), 4))
    core = frame[ly:ly + lh, lx:lx + lw]
    if core.size and float(np.mean(cv2.cvtColor(core, cv2.COLOR_BGR2HSV)[..., 1])) < 80:
        d *= 1.35
    pitch = 1.3 * d
    nb = V[max(0, int(cy - 5 * d)):int(cy + 5 * d), max(0, int(cx - 5 * d)):int(cx + 5 * d)]
    # độ tối TƯƠNG ĐỐI so với cảnh xung quanh (trời mưa / sương mù thì vỏ đèn chỉ xám chứ không đen)
    ref = max(60.0, float(np.percentile(nb, 75))) if nb.size else 160.0
    dark = np.clip((ref - V.astype(np.float32)) / ref, 0.0, 1.0)
    lamp_slot = {"RED": [0], "YELLOW": [1], "GREEN": [2]}.get(lamp_color, [0, 1, 2])

    # Quầng màu quanh bóng (gồm cả số đếm ngược cùng màu đặt cạnh bóng) cho biết hộp đèn nằm ngang hay dọc
    S_, H_ = hsv[..., 1], hsv[..., 0]
    colored = ((S_ >= 90) & (V >= 140)).astype(np.uint8)
    r_ = int(3 * d)
    wx0, wy0 = max(0, int(cx) - r_), max(0, int(cy) - r_)
    win = colored[wy0:int(cy) + r_, wx0:int(cx) + r_]
    halo = None
    if win.size:
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(win, 8)
        k_ = lab_[min(win.shape[0] - 1, int(cy) - wy0), min(win.shape[1] - 1, int(cx) - wx0)]
        if k_ == 0 and n_ > 1:  # tâm cháy trắng: lấy quầng màu gần nhất
            dists = [np.hypot(st_[j, 0] + st_[j, 2] / 2 - (cx - wx0), st_[j, 1] + st_[j, 3] / 2 - (cy - wy0))
                     for j in range(1, n_)]
            k_ = 1 + int(np.argmin(dists)) if min(dists) < 1.5 * d else 0
        if k_ > 0:
            hx, hy, hw, hh = st_[k_, :4]
            halo = (hx + wx0, hy + wy0, hw, hh)
    layout_bonus = {True: 0.03, False: 0.0}  # đèn dọc phổ biến hơn ⇒ ưu tiên nhẹ
    if halo is not None and halo[2] >= 1.8 * halo[3]:
        layout_bonus = {True: 0.0, False: 0.12}
    elif halo is not None and halo[3] >= 1.8 * halo[2]:
        layout_bonus = {True: 0.12, False: 0.0}

    best = None
    for vertical in ((True, False) if force_vertical is None else (force_vertical,)):
        for slot in lamp_slot:
            # bóng đang sáng nằm ở vị trí `slot` (0, 1, 2) trong hộp
            start = -slot * pitch - 0.62 * pitch
            end = (2 - slot) * pitch + 0.62 * pitch
            if vertical:
                x0, x1, y0, y1 = cx - 0.62 * pitch, cx + 0.62 * pitch, cy + start, cy + end
            else:
                x0, x1, y0, y1 = cx + start, cx + end, cy - 0.62 * pitch, cy + 0.62 * pitch
            X0, Y0, X1, Y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
            if X1 - X0 < 3 or Y1 - Y0 < 3:
                continue
            region = dark[Y0:Y1, X0:X1].copy()
            # bỏ phần bóng đang sáng khỏi phép đo độ tối
            region[max(0, ly - Y0):max(0, ly + lh - Y0), max(0, lx - X0):max(0, lx + lw - X0)] = np.nan
            inner = float(np.nanmean(region)) if np.isfinite(region).any() else 0.0
            # dải nền chạy dọc hai bên cạnh dài của hộp: vỏ đèn phải tối hơn hẳn nền xung quanh nó
            s = max(2, int(0.5 * pitch))
            if vertical:
                sides = [dark[Y0:Y1, max(0, X0 - s):X0], dark[Y0:Y1, X1:min(W, X1 + s)]]
            else:
                sides = [dark[max(0, Y0 - s):Y0, X0:X1], dark[Y1:min(H, Y1 + s), X0:X1]]
            sides = [z for z in sides if z.size]
            outer = float(np.mean([z.mean() for z in sides])) if sides else 0.5
            score = inner - outer
            # phần hộp bị cắt bởi mép ảnh thì kém tin cậy
            inside = (X1 - X0) * (Y1 - Y0) / max(1.0, (x1 - x0) * (y1 - y0))
            score *= inside
            score += layout_bonus[vertical]
            if best is None or score > best[0]:
                best = (score, (X0, Y0, X1, Y1))
    if best is None:
        return int(cx - d), int(cy - d), int(cx + d), int(cy + d)
    X0, Y0, X1, Y1 = best[1]
    if halo is not None and halo[2] * halo[3] < 9 * d * d:  # gộp cả số đếm ngược vào khung
        X0, Y0 = min(X0, halo[0]), min(Y0, halo[1])
        X1, Y1 = max(X1, halo[0] + halo[2]), max(Y1, halo[1] + halo[3])
    return X0, Y0, X1, Y1


def merge_fragments(boxes, W, H, max_area=None):
    """Gộp các mảnh của cùng một xe (trừ nền hay tách xe thành nóc / thân / đuôi) thành một khung."""
    boxes = [list(b) for b in boxes]
    gx, gy = 0.01 * W, 0.02 * H
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                touching = ix > -gx and iy > -gy
                small, big = sorted((a[2] * a[3], b[2] * b[3]))
                # mảnh nhỏ nằm sát/chồng lên khung lớn (nóc, đuôi xe...). Hai xe cùng cỡ nối đuôi nhau thì KHÔNG gộp.
                stacked = ix >= 0.6 * min(a[2], b[2]) and small < 0.35 * big
                overlap = ix > 0 and iy > 0 and ix * iy >= 0.3 * small
                if touching and (stacked or overlap):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    if max_area is None or (x1 - x0) * (y1 - y0) <= max_area:
                        boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                        del boxes[j]
                        changed = True
                        break
            if changed:
                break
    return [tuple(int(v) for v in b) for b in boxes]


def box_iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1.0, a[2] * a[3] + b[2] * b[3] - inter)


def same_vehicle(a, b):
    """Hai khung có phải cùng một xe không (trùng nhau, hoặc tâm khung này nằm trong khung kia)."""
    if box_iou(a, b) > 0.1:
        return True
    for p, q in ((a, b), (b, a)):
        cx, cy = p[0] + p[2] / 2, p[1] + p[3] / 2
        if q[0] <= cx <= q[0] + q[2] and q[1] <= cy <= q[1] + q[3]:
            return True
    return False


# ---------- Mô hình C: ghép cặp xe giữa các khung hình ----------
class VehicleTracker:
    """Mô hình C – ghép cặp cùng một xe giữa khung t và t+1.

    Chi phí ghép = 1 − IoU giữa khung DỰ ĐOÁN (khung cũ dịch theo vận tốc) và khung mới; chỉ khi hai khung
    không chồng nhau mới xét khoảng cách tâm (trong phạm vi nửa thân xe). Kích thước phải tương đương.
    Giải bài toán ghép cặp tối ưu bằng thuật toán Hungarian. Cách này hạn chế "nhảy ID" sang xe khác.
    """

    def __init__(self, max_dist, max_missed=8, diag=1000.0):
        self.max_dist = max_dist
        self.diag = diag
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
        BIG = 1e9
        if ids and boxes:
            cost = np.full((len(ids), len(boxes)), BIG)
            for r, i in enumerate(ids):
                t = self.tracks[i]
                k = min(t["missed"], 3) + 1
                shift = t["v"] * k
                tb = t["box"]
                pb = (tb[0] + shift[0], tb[1] + shift[1], tb[2], tb[3])  # khung dự đoán
                size = max(tb[2], tb[3])
                for c, b in enumerate(boxes):
                    ratio = (b[2] * b[3]) / max(1.0, tb[2] * tb[3])
                    if not (0.33 <= ratio <= 3.0):
                        continue  # kích thước khác hẳn ⇒ xe khác
                    iou = box_iou(pb, b)
                    if iou >= 0.05:
                        cost[r, c] = 1.0 - iou
                    else:
                        dist = float(np.linalg.norm(t["c"] + shift - cents[c]))
                        if dist <= min(0.5 * size * k, self.max_dist):
                            cost[r, c] = 1.0 + dist / max(size, 1.0)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] >= BIG:
                    continue
                t = self.tracks[ids[r]]
                step_v = (cents[c] - t["c"]) / (1 + t["missed"])
                t["v"] = 0.6 * t["v"] + 0.4 * step_v if t["age"] > 1 else step_v
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
                self.tracks[self.next_id] = dict(box=b, c=cents[j], c0=cents[j], v=np.zeros(2), missed=0, age=1,
                                                 moved=False, seen_before=False, done=False, violation=False,
                                                 hist=deque(maxlen=60))
                self.next_id += 1
        return {tid: t for tid, t in self.tracks.items() if t["missed"] == 0}


def vehicle_type(box, frame_area, car_area_pct):
    """Ước lượng theo kích thước + tỉ lệ khung: xe máy thường hẹp và cao, ô tô rộng và to."""
    area_pct = box[2] * box[3] / frame_area * 100
    ratio = box[2] / max(box[3], 1)
    if area_pct >= car_area_pct or (ratio >= 0.95 and area_pct >= 0.4 * car_area_pct):
        return "Ô tô"
    return "Xe máy"


def draw_overlay(frame, line, light_rect, light_state, active, info_text, show_light=True):
    img = frame.copy()
    p1, p2 = line.pts()
    line_color = BGR_RED if light_state == "RED" else BGR_YELLOW
    cv2.line(img, p1, p2, line_color, 3)
    cv2.putText(img, "VACH DUNG", (p1[0] + 5, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
    if show_light:
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
    results = dict(violations=[], timeline=[], annotated=[], n_frames=0, crossed_ok=0, skipped=0, moving_down=0, moving_up=0,
                   light_sample=None, light_samples={}, vehicle_sample=None, vehicle_ids=set())
    line = light_rect = detector = tracker = None
    lights = LightTracker(cfg["light_mode"])
    single_image = background is not None and total == 1
    warmup = 0 if background is not None else max(5, int(fps_eff * 0.8))
    margin = 3.0
    cross_m = None  # ngưỡng trễ (hysteresis) quanh vạch, tính theo chiều cao khung hình

    for i, (t_sec, frame) in enumerate(frames):
        if line is None:
            H, W = frame.shape[:2]
            frame_area = float(H * W)
            line = StopLine(cfg["line"], frame.shape, cfg["direction"])
            light_rect = pct_rect_to_px(cfg["light_roi"], frame.shape)
            min_area = cfg["min_area_pct"] / 100 * frame_area
            max_area = cfg.get("max_area_pct", 20) / 100 * frame_area
            diag = float(np.hypot(W, H))
            max_dist = (0.35 if background is not None else 0.12 * max(1, cfg["frame_step"]) ** 0.5) * diag
            tracker = VehicleTracker(max_dist=min(max_dist, 0.4 * diag), max_missed=max(8, int(1.5 * fps_eff)), diag=diag)
            if background is None:
                detector = VideoVehicleDetector(cfg["sensitivity"])

        x0, y0, x1, y1 = light_rect
        roi = frame[y0:y1, x0:x1]
        state = lights.update(roi)
        results["timeline"].append((round(t_sec, 2), state))
        if roi.size and state not in results["light_samples"]:
            results["light_samples"][state] = roi.copy()
        if results["light_sample"] is None and state in ("RED", "GREEN", "YELLOW") and roi.size:
            results["light_sample"] = roi.copy()

        if background is None:
            boxes, motion = detector.detect(frame, min_area, ignore_rect=light_rect, max_area=max_area)
        else:
            boxes, motion = detect_by_background(frame, background, min_area, cfg["sensitivity"],
                                                 ignore_rect=light_rect, max_area=max_area)
        # Quá nửa khung hình "chuyển động" ⇒ camera rung / chuyển cảnh / chữ phủ lên video: bỏ khung này.
        shaky = i >= warmup and background is None and motion > 0.45
        if shaky:
            results["skipped"] += 1
        if i < warmup or shaky:
            boxes = []
        boxes = merge_fragments(boxes, W, H, max_area)
        active = tracker.update(boxes)
        for tr in active.values():
            tr["t_last"] = t_sec
        for tid, tr in active.items():
            if tr["age"] == 1 and "first_db" not in tr:
                x_, y_, w_, h_ = tr["box"]
                head = line.ref_point(tr["box"])
                tail = (x_ + w_ / 2, y_)
                tr["first_db"] = line.dist_before(head)
                tr["box0"], tr["t0"] = tr["box"], t_sec
                tr["first_tail_db"] = line.dist_before(tail)
                # mảnh mới sinh ra đè lên một xe đã được xét ⇒ thừa hưởng kết quả, không tính lại
                for other_id, other in active.items():
                    if other_id != tid and other["age"] > 1 and other["done"] and same_vehicle(tr["box"], other["box"]):
                        tr["done"], tr["violation"] = True, other["violation"]
                        break

        new_viol = []
        replace_evidence = []  # (chỉ số bản ghi, id xe) — thay ảnh bằng chứng bằng khung xe đầy đủ hơn
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
            if cross_m is None:
                cross_m = max(margin, 0.015 * H)
            tr["max_db"] = max(tr.get("max_db", db), db)
            tr["hist"].append((t_sec, db, float(tr["c"][0]), tr["box"][2], ref))
            if db > cross_m:
                tr["seen_before"] = True
            # Quỹ đạo liền mạch: lần gần nhất xe còn ở TRƯỚC vạch phải cách đây ≤ 3 s và cùng làn
            # (tâm xe lệch ngang không quá ~ một thân xe). Chặn trường hợp ID nhảy từ xe này sang xe khác.
            continuous = False
            for (th, dbh, cxh, wh, _) in reversed(tr["hist"]):
                if dbh > cross_m:
                    continuous = (t_sec - th <= 3.0
                                  and abs(float(tr["c"][0]) - cxh) <= max(0.8 * max(wh, tr["box"][2]), 0.06 * W))
                    break
            # Xe đã từng ở trước vạch, HOẶC lúc mới xuất hiện thân xe đang nằm vắt qua vạch
            # (xe có sẵn từ đầu cảnh / đi vào từ mép khung hình ngay sát vạch)
            straddled = tr.get("first_tail_db", -1) > margin and tr.get("first_db", 0) < margin
            # phải thực sự chạy qua vạch một đoạn, không phải khung xe "rung" qua lại quanh vạch khi xe đứng yên
            progressed = tr["max_db"] - db > 0.05 * H
            bx_, by_, bw_, bh_ = tr["box"]
            # mảnh vụn (bóng, kính, mép xe) thường dẹt hoặc rất nhỏ ⇒ không đủ tư cách là "một xe"
            vehicle_like = (bw_ * bh_ >= 1.5 * min_area and 0.3 <= bw_ / max(bh_, 1) <= 2.4
                            and bh_ >= 0.035 * H)
            if db < -cross_m and not tr["done"] and tr["age"] >= 3 and vehicle_like and progressed and (
                    (tr["seen_before"] and continuous) or (straddled and tr["age"] <= 4 * fps_eff)):
                tr["done"] = True
                tr["t_cross"] = t_sec
                recent = results.setdefault("_recent", [])
                def _inside(p, q):  # tâm khung p nằm trong khung q
                    cx__, cy__ = p[0] + p[2] / 2, p[1] + p[3] / 2
                    return q[0] <= cx__ <= q[0] + q[2] and q[1] <= cy__ <= q[1] + q[3]
                # trùng với một xe VỪA qua vạch (≤ 4 s) mà khung mới nằm trong / đè lên khung xe đó
                dup_of = next((o for oid, o in active.items()
                               if oid != tid and o["done"] and t_sec - o.get("t_cross", -99) <= 4.0
                               and (box_iou(tr["box"], o["box"]) > 0.3 or _inside(tr["box"], o["box"]))), None)
                dup = dup_of is not None
                cx_, cy_ = tr["c"]
                for t0, o in recent:  # o = xe đã qua vạch trước đó (giữ vị trí cuối cùng nhìn thấy)
                    dt = t_sec - o.get("t_last", t0)
                    # xe mới "mọc ra" đúng chỗ xe cũ vừa mất dấu ⇒ vẫn là chiếc xe đó (bị mất dấu rồi nhận lại)
                    born = tr.get("t0", t_sec)
                    if (o.get("t_last", t0) - 0.5 <= born <= o.get("t_last", t0) + 6.0
                            and box_iou(tr.get("box0", tr["box"]), o["box"]) > 0.3):
                        dup, dup_of = True, o
                        break
                    if t_sec - t0 > 4.0:
                        continue
                    reach = 0.8 * max(o["box"][2], o["box"][3]) + 0.12 * H * min(max(dt, 0), 2.0) + 0.04 * H
                    if (box_iou(tr["box"], o["box"]) > 0.3 or _inside(tr["box"], o["box"])
                            or np.hypot(cx_ - o["c"][0], cy_ - o["c"][1]) <= reach):
                        dup, dup_of = True, o
                        break
                if dup and state == "RED" and dup_of is not None and dup_of.get("viol_idx") is not None:
                    # bản ghi trước có thể là mảnh vụn của xe ⇒ nếu khung mới to hơn hẳn thì thay ảnh bằng chứng
                    if tr["box"][2] * tr["box"][3] > 1.5 * dup_of["viol_area"]:
                        replace_evidence.append((dup_of["viol_idx"], tid))
                        dup_of["viol_area"] = tr["box"][2] * tr["box"][3]
                recent.append((t_sec, tr))
                if state == "RED":
                    tr["violation"] = True
                    if not dup:  # mảnh khác của cùng một xe vừa được ghi ⇒ không ghi thêm
                        new_viol.append(tid)
                        tr["viol_idx"] = len(results["violations"]) + len(new_viol) - 1
                        tr["viol_area"] = tr["box"][2] * tr["box"][3]
                    elif dup_of is not None:
                        tr["viol_idx"], tr["viol_area"] = dup_of.get("viol_idx"), dup_of.get("viol_area", 0)
                elif not dup:
                    results["crossed_ok"] += 1

        # Thống kê hướng di chuyển thực tế của xe (để phát hiện chọn nhầm hướng)
        for tr in active.values():
            if not tr["moved"] and line.within(tr["c"], tol=0.1):
                dyy = tr["c"][1] - tr["c0"][1]
                if abs(dyy) > 0.08 * H:
                    tr["moved"] = True
                    results["moving_down" if dyy > 0 else "moving_up"] += 1

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
        elif shaky:
            info = f"t={t_sec:5.1f}s | Camera rung / chuyen canh - bo qua khung"
        out = draw_overlay(frame, line, light_rect, state, active, info)

        for tid in new_viol:
            b = active[tid]["box"]
            ref = line.ref_point(b)
            # ảnh bằng chứng: vẽ thêm đường đi của xe (điểm tiếp đất qua các khung) cắt qua vạch dừng
            ev = out.copy()
            pts = [(int(p[4][0]), int(p[4][1])) for p in active[tid].get("hist", [])]
            if len(pts) >= 2:
                cv2.polylines(ev, [np.array(pts, np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(ev, pts[0], 5, (0, 255, 255), -1)
                cv2.arrowedLine(ev, pts[max(0, len(pts) - 4)], pts[-1], (0, 255, 255), 2, tipLength=0.5)
            results["violations"].append({
                "MaXe": f"ID-{tid:03d}",
                "LoaiXe": vehicle_type(b, frame_area, cfg["car_area_pct"]) + " (ước lượng)",
                "ThoiDiem": f"{t_sec:.1f} s",
                "TrangThaiDen": STATE_VN[state],
                "ViTri": f"Xe đã qua vạch tại ({int(ref[0])}, {int(ref[1])})",
                "snapshot": encode_jpg(ev),
                "crop": encode_jpg(crop_box(frame, b)),
            })
        for idx, tid in replace_evidence:
            if 0 <= idx < len(results["violations"]) and tid in active:
                b = active[tid]["box"]
                results["violations"][idx].update(
                    MaXe=f"ID-{tid:03d}", LoaiXe=vehicle_type(b, frame_area, cfg["car_area_pct"]) + " (ước lượng)",
                    snapshot=encode_jpg(out), crop=encode_jpg(crop_box(frame, b)))

        if writer is not None:
            writer(out)
        else:
            results["annotated"].append(out)
        results["n_frames"] += 1
        if progress is not None and total:
            progress.progress(min(1.0, (i + 1) / total), text=f"Đang xử lý khung {i + 1}/{total}")

    results.pop("_recent", None)
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


# ---------- Tiền xử lý video: tách cảnh, bỏ viền đen, đoán hướng xe, phát hiện camera rung/zoom ----------
def _scene_signature(frame):
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def content_box(frames):
    """Vùng hình camera thật (x0, y0, x1, y1): bỏ viền đen và chữ tiêu đề nằm trên viền đen của video dọc."""
    g = np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]), axis=0)
    bright = g > 28

    def longest_run(frac):
        good = list(frac > 0.55) + [False]
        best, best_len, cur, start = (0, len(frac)), 0, 0, 0
        for i, v in enumerate(good):
            if v:
                start = i if cur == 0 else start
                cur += 1
            else:
                if cur > best_len:
                    best_len, best = cur, (start, start + cur)
                cur = 0
        return best if best_len > 0.25 * len(frac) else (0, len(frac))

    y0, y1 = longest_run(bright.mean(axis=1))
    x0, x1 = longest_run(bright[y0:y1].mean(axis=0))
    # làm tròn chẵn để mã hoá video
    return x0, y0, x0 + (x1 - x0) // 2 * 2, y0 + (y1 - y0) // 2 * 2


def motion_profile(frames):
    """Optical flow Farneback: (hướng xe 'down'/'up'/None, độ tin cậy, camera có đang di chuyển/zoom không)."""
    votes, cam = [], []
    prev = None
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (320, max(32, int(320 * f.shape[0] / f.shape[1])))), cv2.COLOR_BGR2GRAY)
        if prev is not None and prev.shape == g.shape:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            cam.append(float(np.median(mag)))  # camera đứng yên ⇒ phần lớn điểm ảnh không chuyển động
            low = flow[g.shape[0] // 3:]
            m = np.hypot(low[..., 0], low[..., 1]) > 1.0
            if m.sum() > 50:
                votes.append(float(np.median(low[..., 1][m])))
        prev = g
    moving_cam = bool(cam) and float(np.median(cam)) > 0.5
    v = np.array(votes)
    down, up = int((v > 0.3).sum()), int((v < -0.3).sum())
    if down + up == 0:
        return None, 0.0, moving_cam
    return ("down" if down >= up else "up"), max(down, up) / (down + up), moving_cam


@st.cache_data(show_spinner=False, max_entries=6)
def scan_video_scenes(video_path, file_size, sample_fps=5.0, max_seconds=300):
    """Đọc lướt video, chia thành các cảnh quay (video tổng hợp thường ghép nhiều camera)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    step = max(1, int(round(fps / sample_fps)))
    samples, idx = [], 0
    while idx / fps < max_seconds:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            scale = min(1.0, 480 / f.shape[1])
            samples.append((idx / fps, cv2.resize(f, None, fx=scale, fy=scale) if scale < 1 else f))
        idx += 1
    cap.release()
    duration = idx / fps
    if not samples:
        return dict(fps=fps, duration=0.0, scenes=[])
    H0, W0 = samples[0][1].shape[:2]

    # 1) Vùng hình chung (bỏ viền đen) để so sánh các khung với nhau
    gx0, gy0, gx1, gy1 = content_box([f for _, f in samples[:: max(1, len(samples) // 20)]])
    sigs = [_scene_signature(f[gy0:gy1, gx0:gx1]) for _, f in samples]
    diffs, corrs = [], []
    for a, b in zip(sigs, sigs[1:]):
        corrs.append(cv2.compareHist(a[0], b[0], cv2.HISTCMP_CORREL))
        diffs.append(float(np.mean(np.abs(a[1] - b[1]))))
    thr = max(14.0, 4 * float(np.median(diffs))) if diffs else 14.0
    cuts = [0.0]
    for k, (d, c) in enumerate(zip(diffs, corrs)):
        t = samples[k + 1][0]
        if (d > thr or c < 0.5) and t - cuts[-1] >= 0.8:
            cuts.append(t)
    bounds = cuts + [duration]

    scenes = []
    for i in range(len(cuts)):
        s, e = bounds[i], bounds[i + 1]
        fr = [f for t, f in samples if s + 0.25 <= t < e - 0.1] or [f for t, f in samples if s <= t < e]
        if not fr:
            continue
        x0, y0, x1, y1 = content_box(fr[:: max(1, len(fr) // 15)])
        crop = [f[y0:y1, x0:x1] for f in fr[: 40]]
        direction, conf, moving = motion_profile(crop)
        mid = fr[len(fr) // 2][y0:y1, x0:x1]
        thumb = cv2.resize(mid, (240, max(1, int(240 * mid.shape[0] / mid.shape[1]))))
        # toạ độ vùng hình theo tỉ lệ (để áp lên khung gốc độ phân giải đầy đủ)
        box = (x0 / W0, y0 / H0, x1 / W0, y1 / H0)
        scenes.append(dict(start=s, end=e, box=box, direction=direction, conf=conf, moving=moving, thumb=thumb))
    return dict(fps=fps, duration=duration, scenes=scenes)


def crop_rel(frame, box):
    if box is None:
        return frame
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
    x1 = x0 + max(2, (x1 - x0) // 2 * 2)
    y1 = y0 + max(2, (y1 - y0) // 2 * 2)
    return frame[y0:y1, x0:x1]


def read_frame_at(video_path, t_sec=0.0, box=None):
    cap = cv2.VideoCapture(video_path)
    if t_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ok, frame = cap.read()
    if not ok and t_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    return resize_frame(crop_rel(frame, box)) if ok else None


def read_first_frame(video_path):
    return read_frame_at(video_path, 0.0)


def analyze_video(video_path, cfg, progress, start=0.0, end=None, box=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, cfg["frame_step"])
    start_idx = int(round(start * fps))
    stop_idx = start_idx + int(fps * cfg["max_seconds"])
    if end is not None:
        stop_idx = min(stop_idx, int(round(end * fps)))
    if n_total > 0:
        stop_idx = min(stop_idx, n_total)
    n_proc = max(1, (stop_idx - start_idx) // step)
    fps_eff = fps / step

    if start_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    ok, first = cap.read()
    if not ok:
        cap.release()
        return None
    first = resize_frame(crop_rel(first, box))
    size = (first.shape[1], first.shape[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    def gen():
        idx = start_idx
        while idx < stop_idx:
            ok, f = cap.read()
            if not ok:
                break
            if (idx - start_idx) % step == 0:
                yield idx / fps, resize_frame(crop_rel(f, box), size)
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
    kind = good if isinstance(good, str) else ("good" if good else "bad")
    st.markdown(f'<div class="banner {kind}"><span class="dot"></span><div>{text}</div></div>',
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
            with st.spinner("Đang đọc lướt video: tách cảnh quay, bỏ viền đen, ước lượng hướng xe..."):
                scan = scan_video_scenes(video_path, os.path.getsize(video_path))
            scenes = scan["scenes"]
            if not scenes:
                banner("Không đọc được video này. Hãy chuyển sang định dạng .mp4 (H.264) rồi thử lại.", good=False)
            else:
                base_key = source_key
                scene_idx = 0
                if len(scenes) > 1:
                    st.markdown(f"**Video gồm {len(scenes)} cảnh quay khác nhau** · mỗi cảnh là một camera / góc quay "
                                "riêng nên cần chọn đèn và vạch dừng riêng. Chọn cảnh cần phân tích:")
                    per_row = 6
                    for r0 in range(0, len(scenes), per_row):
                        cols = st.columns(per_row)
                        for j, sc in enumerate(scenes[r0:r0 + per_row]):
                            k = r0 + j
                            flag = " · ⚠ camera chuyển động" if sc["moving"] else ""
                            cols[j].image(sc["thumb"], channels="BGR", width="stretch",
                                          caption=f"Cảnh {k + 1} · {sc['start']:.1f}–{sc['end']:.1f}s{flag}")
                    labels = [f"Cảnh {k + 1}" for k in range(len(scenes))]
                    pick = st.segmented_control("Cảnh cần phân tích", labels, default=labels[0],
                                                key=f"scene_pick_{base_key}", label_visibility="collapsed")
                    scene_idx = labels.index(pick) if pick in labels else 0
                    source_key = f"{base_key}#canh{scene_idx + 1}"
                scene = scenes[scene_idx]
                st.session_state.scene_info = dict(start=scene["start"], end=scene["end"], box=scene["box"],
                                                   label=f"Cảnh {scene_idx + 1}" if len(scenes) > 1 else "")
                cut = scene["box"]
                full = abs(cut[0]) < 0.01 and abs(cut[1]) < 0.01 and cut[2] > 0.99 and cut[3] > 0.99
                st.session_state.scene_info["box"] = None if full else cut
                preview = read_frame_at(video_path, scene["start"] + min(0.5, (scene["end"] - scene["start"]) / 3),
                                        st.session_state.scene_info["box"])
                if not full:
                    st.caption("Đã tự cắt bỏ viền đen / chữ tiêu đề quanh hình camera.")
                if scene["moving"]:
                    banner("<b>Camera trong cảnh này đang zoom hoặc lia.</b> Phương pháp trừ nền chỉ đúng với camera đứng yên, "
                           "nên kết quả cảnh này sẽ không đáng tin (xe bị cắt thành nhiều mảnh, báo vi phạm nhầm). "
                           "Nên chọn cảnh khác.", good="warn")
                # Hướng xe: tự điền theo ước lượng optical flow cho mỗi cảnh mới (người dùng vẫn đổi được)
                if scene["direction"] and st.session_state.get("dir_applied_for") != source_key:
                    st.session_state.direction = "Từ trên xuống ↓" if scene["direction"] == "down" else "Từ dưới lên ↑"
                    st.session_state.dir_applied_for = source_key
                st.session_state.dir_guess = (scene["direction"], scene["conf"])
                if preview is None:
                    banner("Không đọc được khung hình của cảnh này.", good=False)
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
                # video mẫu: đèn ở góc phải, vạch dừng ở 65% chiều cao — cấu hình sẵn để demo ngay
                st.session_state.pending_cfg = dict(rx=87, ry=3, rw=7, rh=30, lx1=5, ly1=65, lx2=95, ly2=65,
                                                    light_mode="auto")
                st.session_state.light_confirmed_for = "vid:video_thu.mp4:sample"
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
        step(2, "Hiệu chỉnh trên hình", "Bấm trực tiếp lên ảnh để chọn đèn và vẽ vạch dừng")

        # Giá trị mặc định + áp dụng thay đổi từ lần bấm chuột trước (phải làm trước khi tạo thanh trượt)
        for k_, v_ in dict(lx1=5, ly1=65, lx2=95, ly2=65, rx=85, ry=5, rw=10, rh=25).items():
            st.session_state.setdefault(k_, v_)
        st.session_state.setdefault("light_mode", "auto")
        st.session_state.setdefault("line_first_pt", None)
        st.session_state.cur_source = source_key

        def _confirm_light():  # người dùng tự kéo thanh trượt vùng đèn ⇒ coi như đã chọn đèn
            st.session_state.light_confirmed_for = st.session_state.get("cur_source")
        pending = st.session_state.pop("pending_cfg", None)
        if pending:
            for k_, v_ in pending.items():
                st.session_state[k_] = v_

        col_prev, col_cfg = st.columns([1.5, 1], gap="large")
        with col_cfg:
            with st.container(border=True):
                tool = st.segmented_control(
                    "Công cụ bấm trên ảnh", ["🚦 Chọn đèn", "📏 Vẽ vạch dừng"], default="🚦 Chọn đèn", key="tool",
                ) or "🚦 Chọn đèn"
                if tool == "🚦 Chọn đèn":
                    layout_opts = {"auto": "Tự động", "vertical": "Dọc", "horizontal": "Ngang", "lamp": "Chỉ bóng"}
                    st.segmented_control("Kiểu hộp đèn", list(layout_opts), format_func=layout_opts.get,
                                         default="auto", key="light_layout",
                                         help="Khung tự ôm cả hộp đèn 3 bóng. Nếu đoán sai kiểu đèn (dọc/ngang), chọn lại "
                                              "ở đây — khung tự tính lại, không cần bấm lại vào đèn.")
                    st.markdown(
                        "Bấm vào **đèn tròn điều khiển làn xe cần giám sát**. Hệ thống tự ôm khung quanh đèn.  \n"
                        f"<span style='color:{C_AMBER}'>Không chọn đèn mũi tên rẽ phải/trái</span> nếu làn đó có đèn riêng.",
                        unsafe_allow_html=True)
                else:
                    first_pt = st.session_state.line_first_pt
                    st.markdown(
                        ("Bấm **điểm thứ hai** (đầu còn lại của vạch dừng)." if first_pt else
                         "Bấm **2 điểm** ở hai đầu vạch dừng.")
                        + "  \nChỉ kéo vạch qua những làn chịu đèn đã chọn; bỏ làn rẽ có đèn mũi tên riêng.")
                direction_lbl = st.radio("Hướng xe chạy (của làn chịu đèn đã chọn)", ["Từ trên xuống ↓", "Từ dưới lên ↑"],
                                         horizontal=True, key="direction",
                                         help="Thấy đầu xe ⇒ xe chạy về phía camera (thường là từ trên xuống). "
                                              "Thấy đuôi xe, lưng người lái ⇒ xe chạy ra xa camera (từ dưới lên). "
                                              "Mặt đèn quay về phía camera thường điều khiển các xe đang đi ra xa camera.")
                direction = "down" if direction_lbl.startswith("Từ trên") else "up"
                guess = st.session_state.get("dir_guess") if input_type == "🎥 Video" else None
                if guess and guess[0]:
                    g_lbl = "từ trên xuống ↓" if guess[0] == "down" else "từ dưới lên ↑"
                    st.caption(f"Hệ thống đoán (optical flow): xe chủ yếu chạy **{g_lbl}** · độ tin cậy {guess[1] * 100:.0f}%")
                light_options = {
                    "auto": "Tự động nhận diện màu đèn",
                    "red": "Không thấy đèn trong hình · coi như luôn đỏ",
                    "green": "Coi như luôn xanh",
                }
                light_mode = st.selectbox("Trạng thái đèn", list(light_options), format_func=light_options.get,
                                          key="light_mode")

            with st.expander("Chỉnh chi tiết bằng thanh trượt"):
                st.caption("Vạch dừng (% khung hình)")
                a1, a2 = st.columns(2)
                lx1 = a1.slider("Đầu 1 · ngang", 0, 100, key="lx1")
                ly1 = a1.slider("Đầu 1 · dọc", 0, 100, key="ly1")
                lx2 = a2.slider("Đầu 2 · ngang", 0, 100, key="lx2")
                ly2 = a2.slider("Đầu 2 · dọc", 0, 100, key="ly2")
                st.caption("Vùng đèn (% khung hình)")
                b1, b2 = st.columns(2)
                rx = b1.slider("Vị trí ngang", 0, 99, key="rx", on_change=_confirm_light)
                ry = b1.slider("Vị trí dọc", 0, 99, key="ry", on_change=_confirm_light)
                rw = b2.slider("Chiều rộng", 1, 50, key="rw", on_change=_confirm_light)
                rh = b2.slider("Chiều cao", 1, 60, key="rh", on_change=_confirm_light)

            with st.expander("Tinh chỉnh nâng cao"):
                min_area_pct = st.slider("Diện tích xe nhỏ nhất (% khung hình)", 0.05, 5.0, 0.4, 0.05,
                                         help="Tăng lên nếu lá cây, người đi bộ hoặc nhiễu bị nhận nhầm là xe.")
                max_area_pct = st.slider("Diện tích xe lớn nhất (% khung hình)", 3, 60, 20,
                                         help="Bỏ các vùng quá lớn: cả đoàn xe dính nhau, camera rung, chữ phủ lên video.")
                car_area_pct = st.slider("Ngưỡng coi là ô tô (% khung hình)", 0.5, 15.0, 2.5, 0.5)
                sensitivity = st.slider("Độ nhạy trừ nền (nhỏ = nhạy hơn)", 8, 80, 32)
                frame_step = st.slider("Xử lý 1 trên mỗi N khung (lớn = nhanh hơn)", 1, 6, 2)
                max_seconds = st.slider("Chỉ phân tích tối đa N giây đầu video", 5, 180, 60)

        cfg = dict(line=(lx1, ly1, lx2, ly2), direction=direction, light_roi=(rx, ry, rw, rh),
                   light_mode=light_mode, min_area_pct=min_area_pct, max_area_pct=max_area_pct,
                   car_area_pct=car_area_pct, sensitivity=sensitivity, frame_step=frame_step,
                   max_seconds=max_seconds)

        light_ok = light_mode != "auto" or st.session_state.get("light_confirmed_for") == source_key

        with col_prev:
            Hp, Wp = preview.shape[:2]
            line_prev = StopLine(cfg["line"], preview.shape, direction)
            rect_prev = pct_rect_to_px(cfg["light_roi"], preview.shape)
            x0, y0, x1, y1 = rect_prev
            light_roi_img = preview[y0:y1, x0:x1]
            if not light_ok:
                prev_state = "UNKNOWN"
            elif light_mode == "auto":
                prev_state, _ = classify_light(light_roi_img)
            else:
                prev_state = "RED" if light_mode == "red" else "GREEN"
            hint = "Xem truoc" if light_ok else "BAM VAO DEN GIAO THONG DE CHON DEN"
            prev_img = draw_overlay(preview, line_prev, rect_prev, prev_state, {}, hint,
                                    show_light=light_ok and light_mode == "auto")
            p1, p2 = line_prev.pts()
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            dy = 45 if direction == "down" else -45
            cv2.arrowedLine(prev_img, (mid[0], mid[1] - dy), (mid[0], mid[1] + dy // 3), (248, 189, 56), 3, tipLength=0.35)
            fp = st.session_state.line_first_pt
            if tool != "🚦 Chọn đèn" and fp:
                cv2.circle(prev_img, (int(fp[0] / 100 * Wp), int(fp[1] / 100 * Hp)), 7, (0, 215, 255), -1)
                cv2.circle(prev_img, (int(fp[0] / 100 * Wp), int(fp[1] / 100 * Hp)), 11, (255, 255, 255), 2)

            with st.container(border=True):
                click = None
                if image_picker is not None:
                    click = image_picker(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB), width="stretch",
                                         key=f"picker_{source_key}", cursor="crosshair",
                                         image_format="JPEG", jpeg_quality=88)
                else:
                    st.image(prev_img, channels="BGR", width="stretch")
                    st.caption("Thiếu gói streamlit-image-coordinates nên chưa bấm được lên ảnh; hãy dùng thanh trượt.")

                if click and click.get("unix_time") != st.session_state.get("last_click"):
                    st.session_state.last_click = click.get("unix_time")
                    cx = int(click["x"] * Wp / max(click["width"], 1))
                    cy = int(click["y"] * Hp / max(click["height"], 1))
                    if tool == "🚦 Chọn đèn":
                        st.session_state.last_lamp_click = (source_key, cx, cy)
                        snapped = snap_light_roi(preview, cx, cy, st.session_state.get("light_layout") or "auto")
                        st.session_state.layout_applied = (source_key, st.session_state.get("light_layout") or "auto")
                        if snapped is None:
                            st.session_state.light_miss = source_key
                        else:
                            nrx, nry, nrw, nrh = snapped
                            st.session_state.light_miss = None
                            st.session_state.pending_cfg = dict(rx=min(nrx, 99), ry=min(nry, 99), rw=min(max(nrw, 1), 50),
                                                                rh=min(max(nrh, 1), 60), light_mode="auto")
                            st.session_state.light_confirmed_for = source_key
                    else:
                        pt = (round(cx / Wp * 100), round(cy / Hp * 100))
                        if st.session_state.line_first_pt is None:
                            st.session_state.line_first_pt = pt
                        else:
                            a_ = st.session_state.line_first_pt
                            left, right = sorted([a_, pt])
                            st.session_state.pending_cfg = dict(lx1=left[0], ly1=left[1], lx2=right[0], ly2=right[1])
                            st.session_state.line_first_pt = None
                    st.rerun()

                if st.session_state.get("light_miss") == source_key and tool == "🚦 Chọn đèn":
                    banner("<b>Chỗ vừa bấm không có đèn đang sáng</b>, nên chưa đặt khung đèn. Hãy bấm đúng vào bóng đèn "
                           "đang sáng (đỏ, vàng hoặc xanh). Nếu muốn vẽ vạch dừng, chọn công cụ 📏 trước.", good=False)
                if not light_ok:
                    banner("<b>Chưa chọn đèn.</b> Chọn công cụ 🚦 rồi bấm lên ảnh, vào đúng <b>đèn tròn</b> điều khiển "
                           "làn xe cần giám sát (không phải đèn mũi tên rẽ). Nếu video không quay được đèn, chọn "
                           "“coi như luôn đỏ” ở mục Trạng thái đèn.", good="warn")
                cur_layout = st.session_state.get("light_layout") or "auto"
                lc = st.session_state.get("last_lamp_click")
                if (lc and lc[0] == source_key and light_ok and light_mode == "auto"
                        and st.session_state.get("layout_applied") != (source_key, cur_layout)):
                    st.session_state.layout_applied = (source_key, cur_layout)
                    snapped = snap_light_roi(preview, lc[1], lc[2], cur_layout)
                    if snapped is not None:
                        nrx, nry, nrw, nrh = snapped
                        st.session_state.pending_cfg = dict(rx=min(nrx, 99), ry=min(nry, 99), rw=min(max(nrw, 1), 50),
                                                            rh=min(max(nrh, 1), 60))
                        st.rerun()
                z1, z2 = st.columns([1, 4], vertical_alignment="center")
                if light_ok and light_mode == "auto" and light_roi_img.size:
                    zoom = cv2.resize(light_roi_img, (96, max(1, int(96 * light_roi_img.shape[0] / max(light_roi_img.shape[1], 1)))),
                                      interpolation=cv2.INTER_NEAREST)
                    z1.image(zoom, channels="BGR", width="stretch")
                if light_ok and light_mode != "auto":
                    z2.markdown(f"Không dùng camera để đọc đèn &nbsp; {pill(prev_state)}", unsafe_allow_html=True)
                elif light_ok:
                    z2.markdown(f"Bên trong khung đèn &nbsp; {pill(prev_state)}<br>"
                                f"<span style='color:{C_MUTED};font-size:.85rem'>Khung đầu tiên của video · "
                                f"mũi tên xanh = hướng xe chạy</span>", unsafe_allow_html=True)
                if light_ok and prev_state == "UNKNOWN" and light_mode == "auto":
                    st.caption("Chưa thấy đèn sáng trong khung. Bấm lại vào đúng đèn, hoặc chọn “coi như luôn đỏ” "
                               "nếu video không quay được đèn.")

        step(3, "Phân tích", "")
        can_run = light_ok and not (input_type == "🖼️ Ảnh" and background is None)
        if input_type == "🖼️ Ảnh" and background is None:
            banner("Với 1–2 ảnh, hãy tải thêm ảnh nền đường trống, hoặc tải từ 3 ảnh liên tiếp trở lên.", good=False)
        if not light_ok:
            st.caption("Nút phân tích sẽ mở khi bạn đã chọn đèn ở Bước 2.")
        run = st.button("🔍  Phân tích vi phạm", type="primary", disabled=not can_run, width="stretch")
        if run:
            progress = st.progress(0.0, text="Đang chuẩn bị...")
            try:
                if input_type == "🎥 Video":
                    sc_ = st.session_state.get("scene_info") or {}
                    res = analyze_video(video_path, cfg, progress, start=sc_.get("start", 0.0),
                                        end=sc_.get("end"), box=sc_.get("box"))
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
                res["light_mode"] = cfg["light_mode"]
                res["direction"] = cfg["direction"]
                st.session_state.analysis = res
                now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                src_name = source_key.split(":")[1] if source_key else ""
                for v in res["violations"]:
                    st.session_state.violations_log.append({
                        "Mã xe": v["MaXe"], "Loại xe": v["LoaiXe"], "Thời điểm": v["ThoiDiem"],
                        "Đèn": v["TrangThaiDen"], "Vị trí": v["ViTri"],
                        "Nguồn": src_name + (f" · {(st.session_state.get('scene_info') or {}).get('label')}"
                                             if input_type == "🎥 Video" and (st.session_state.get('scene_info') or {}).get('label') else ""),
                        "Phân tích lúc": now,
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

            # Cảnh báo khi kết quả có thể sai do cấu hình
            seen_states = {s for _, s in res["timeline"]}
            if res.get("light_mode") == "auto" and res["timeline"] and "RED" not in seen_states:
                seen_txt = ", ".join(STATE_VN[s].lower() for s in seen_states)
                banner(f"<b>Suốt đoạn đã xét, đèn chưa lần nào được nhận là ĐỎ</b> (chỉ thấy: {seen_txt}). "
                       "Nếu thực tế đèn đỏ, có thể khung đèn đang đặt sai chỗ, ví dụ trúng đèn mũi tên rẽ phải. "
                       "Hãy bấm lại vào đúng đèn tròn ở Bước 2 rồi phân tích lại.", good="warn")
            n_dn, n_up = res.get("moving_down", 0), res.get("moving_up", 0)
            chosen_dir = res.get("direction", "down")
            n_same, n_opp = (n_dn, n_up) if chosen_dir == "down" else (n_up, n_dn)
            if res["kind"] == "video" and n_opp >= 3 and n_opp > 1.5 * n_same:
                other = "Từ dưới lên ↑" if chosen_dir == "down" else "Từ trên xuống ↓"
                banner(f"<b>Có thể bạn chọn ngược hướng xe chạy.</b> Trong vùng vạch dừng, {n_opp} xe đi theo hướng "
                       f"<b>{other}</b>, chỉ {n_same} xe đi theo hướng đã chọn. Mẹo: nếu thấy đuôi xe / lưng người lái "
                       "thì xe đang đi ra xa camera (từ dưới lên).", good="warn")
            if res["kind"] == "video" and res["n_frames"] and res.get("skipped", 0) / res["n_frames"] > 0.15:
                banner(f"{res['skipped']} / {res['n_frames']} khung hình bị bỏ qua vì gần như cả khung đều chuyển động. "
                       "Video có thể bị rung, bị quay lại từ màn hình điện thoại, hoặc có chữ/hiệu ứng chạy đè lên. "
                       "Kết quả sẽ chính xác hơn với video gốc từ camera đặt cố định.", good="warn")

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
                    samples = res.get("light_samples") or {}
                    if samples:
                        st.caption("Vùng đèn hệ thống đã nhìn thấy ứng với từng trạng thái (để kiểm tra khung đèn đặt đúng chưa):")
                        scols = st.columns(8)
                        for j, (s_, img_) in enumerate(samples.items()):
                            with scols[j]:
                                st.image(cv2.resize(img_, (80, max(1, int(80 * img_.shape[0] / max(img_.shape[1], 1)))),
                                                    interpolation=cv2.INTER_NEAREST), channels="BGR", width="stretch")
                                st.markdown(pill(s_, ""), unsafe_allow_html=True)

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
