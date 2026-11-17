import base64
import json
import os
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from skimage.feature import hog

import hashlib
import importlib

import he_thong as _lo_he_thong

# Luôn nạp lại he_thong.py ở mỗi lần chạy, để khi repo cập nhật lõi mới thì web dùng ngay bản mới
_lo_he_thong = importlib.reload(_lo_he_thong)
with open(_lo_he_thong.__file__, "rb") as _f:
    PHIEN_BAN_LOI = hashlib.md5(_f.read()).hexdigest()[:8]
from he_thong import (BoTheoDoi, HeThongCanhBao, TEN_NHOM_DEN, dac_trung_den_anh, hog_lo)

try:
    from he_thong import anh_nen_trung_vi
except ImportError:
    def anh_nen_trung_vi(ds_anh):
        return None

try:  # he_thong.py bản mới có hàm này; nếu repo còn he_thong.py cũ thì web vẫn chạy (chỉ không vẽ khung bóng đèn)
    from he_thong import khung_bong_sang
except ImportError:
    def khung_bong_sang(img, khung, gian=0.25):
        return khung

try:  # cho phép bấm chuột lên ảnh để chọn đèn / vạch dừng
    from streamlit_image_coordinates import streamlit_image_coordinates as image_picker
except ImportError:
    image_picker = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "mo_hinh")


DEN_MAU_B64 = "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"


def _khoa_ten(ten):
    """Sắp tên file theo số tự nhiên: anh_2.jpg đứng trước anh_10.jpg."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", ten)]


def _tim(ten):
    """Tìm file trong mo_hinh/ trước, nếu không có thì tìm cạnh app.py (repo để file ở thư mục gốc)."""
    for d in (MODEL_DIR, APP_DIR):
        p = os.path.join(d, ten)
        if os.path.exists(p):
            return p
    return os.path.join(MODEL_DIR, ten)


MODEL_PATH = _tim("he_thong_canh_bao.joblib")     # notebook, mục E: lưu mô hình
RESULT_PATH = _tim("ket_qua_tong_hop.json")       # notebook: mọi số liệu đánh giá
SAMPLE_VIDEO = os.path.join(APP_DIR, "video_thu.mp4")
NOTEBOOK_SAMPLE = _tim("video_mau_test.mp4")      # video ghép từ frame test (notebook mục F1)

# ==========================================
# CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="Cảnh báo vượt đèn đỏ",
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
# TIỆN ÍCH ẢNH / VIDEO
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


def bright_lamp_color(roi_bgr):
    """CHỈ dùng cho công cụ bấm chọn đèn (tìm bóng đèn đang sáng để ôm khung), KHÔNG phải mô hình A.

    Đếm tỉ lệ điểm ảnh sáng & đậm màu thuộc dải Đỏ / Vàng / Xanh trong vùng đèn."""
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
    lamp_color, _ = bright_lamp_color(frame[lamp[1]:lamp[1] + bh, lamp[0]:lamp[0] + bw])
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


def encode_jpg(img_bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def crop_box(frame, box, pad=0.25):
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    H, W = frame.shape[:2]
    return frame[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)].copy()


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


# ==========================================
# NẠP MÔ HÌNH ĐÃ HUẤN LUYỆN TỪ NOTEBOOK
# ==========================================
@st.cache_resource(show_spinner="Đang nạp mô hình A, B, C từ notebook...")
def load_system(path, mtime, phien_ban_loi):
    # phien_ban_loi nằm trong khóa cache: he_thong.py đổi thì mô hình được nạp lại bằng lõi mới
    return HeThongCanhBao(path)


@st.cache_data(show_spinner=False)
def load_results(path, mtime):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


if not os.path.exists(MODEL_PATH):
    st.error("Không tìm thấy **mo_hinh/he_thong_canh_bao.joblib**. Chép file này (notebook lưu ở mục E, nằm trong "
             "`ket_qua_mo_hinh.zip`) vào thư mục `mo_hinh/` cạnh app.py rồi tải lại trang.")
    st.stop()
HT = load_system(MODEL_PATH, os.path.getmtime(MODEL_PATH), PHIEN_BAN_LOI)
KQ = load_results(RESULT_PATH, os.path.getmtime(RESULT_PATH)) if os.path.exists(RESULT_PATH) else {}


def _kq(path, default=None):
    """Lấy một giá trị trong ket_qua_tong_hop.json theo đường dẫn 'A/test_macro_f1'."""
    cur = KQ
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ten_C():
    return _kq("C_mo_hinh_ghep_xe/mo_hinh", "k-NN (k = 25)")


def _auc_C():
    for r in _kq("C_mo_hinh_ghep_xe/bang_ngoai_fold", []) or []:
        if r.get("Mô hình C") == _ten_C():
            return r.get("AUC cặp thật (ngoài fold)")
    return None


def _b_tim_test():
    for r in _kq("B_tim_xe/test", []) or []:
        if r.get("Phiên bản B") == HT.phien_ban_B:
            return r
    return None


# Mô tả ba mô hình đang chạy, lấy đúng từ file joblib + ket_qua_tong_hop.json
MODELS = {
    "A": dict(ten="Nhận diện màu đèn", thuat_toan=_kq("A/mo_hinh", "k-NN"), cau_hinh=_kq("A/cau_hinh", "k=9, distance, manhattan"),
              dac_trung="44 đặc trưng màu HSV của vùng đèn (chuẩn hóa StandardScaler)",
              chi_so=[("Macro F1 (test)", _kq("A/test_macro_f1")), ("Recall đèn đỏ (test)", _kq("A/recall_den_do")),
                      ("Accuracy (test)", _kq("A/test_accuracy"))]),
    "B": dict(ten="Phát hiện xe", thuat_toan=_kq("B/mo_hinh", "SVM-RBF"), cau_hinh=_kq("B/cau_hinh", "C=10, γ=scale"),
              dac_trung="HOG 64×64 (1.764 chiều) → PCA 150 thành phần, quét cửa sổ trượt quanh vạch dừng",
              chi_so=[("F1 xe/nền (test)", _kq("B/test_f1")),
                      ("F1 tìm xe trên frame (test)", (_b_tim_test() or {}).get("F1"))]),
    "C": dict(ten="Theo dõi phương tiện", thuat_toan=_ten_C(), cau_hinh=f"ngưỡng {HT.nguong_C:g} + Hungarian ghép một-một",
              dac_trung="21 đặc trưng của cặp khung xe ở hai frame liền nhau",
              chi_so=[("AUC cặp thật (ngoài fold)", _auc_C())]),
}


DIR_UP = "Từ dưới lên ↑ · ra xa camera (như dữ liệu huấn luyện)"
DIR_DOWN = "Từ trên xuống ↓ · về phía camera"


# ==========================================
# PIPELINE A → B → C → BƯỚC KẾT LUẬN (MÔ HÌNH THẬT CỦA NOTEBOOK)
# ==========================================
def to640(frame):
    """Mô hình được huấn luyện trên ảnh 640×640 của dataset ⇒ mọi khung hình đều đưa về 640×640 trước khi vào A, B, C."""
    return cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)


def light_box_rel(light_roi_pct):
    """(x, y, rộng, cao) % ⇒ (x1, y1, x2, y2) tỉ lệ 0..1, đúng dạng khung_den của notebook."""
    x, y, w, h = light_roi_pct
    return (x / 100, y / 100, min(1.0, (x + w) / 100), min(1.0, (y + h) / 100))


def predict_light(frame, light_roi_pct):
    """Mô hình A trên một khung hình: (trạng thái RED/YELLOW/GREEN, {lớp: xác suất})."""
    img = to640(frame)
    box = light_box_rel(light_roi_pct)
    if hasattr(HT, "doan_den_chi_tiet"):
        lop, xs, ct = HT.doan_den_chi_tiet(img, box)
    else:
        (lop, xs), ct = HT.doan_den(img, box), {}
    st.session_state["light_detail"] = ct
    return {"red_light": "RED", "yellow_light": "YELLOW", "green_light": "GREEN"}.get(lop, "UNKNOWN"), xs


def scale_box(k, sx, sy):
    return int(k[0] * sx), int(k[1] * sy), int(k[2] * sx), int(k[3] * sy)


def draw_result(frame, cfg, r, info_text, tracks_bottom=None):
    """Vẽ kết quả của một frame lên ảnh hiển thị (toạ độ mô hình 640×640 được đổi về kích thước ảnh)."""
    img = frame.copy()
    H, W = img.shape[:2]
    sx, sy = W / 640, H / 640
    ly = int(cfg["y_vach"] / 100 * H)
    state = r["den"] if r else "UNKNOWN"
    line_color = BGR_RED if state == "RED" else BGR_YELLOW
    # dải B quét cửa sổ trượt (vạch − 20% … vạch + 15% chiều cao)
    lo, hi = int(max(0, cfg["y_vach"] / 100 + HT.roi[0]) * H), int(min(1, cfg["y_vach"] / 100 + HT.roi[1]) * H)
    band = img.copy()
    cv2.rectangle(band, (0, lo), (W, hi), (56, 189, 248), -1)
    img = cv2.addWeighted(band, 0.06, img, 0.94, 0)
    cv2.line(img, (0, ly), (W, ly), line_color, 3)
    cv2.putText(img, "VACH DUNG", (6, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
    if cfg["light_mode"] == "auto":
        x0, y0, x1, y1 = pct_rect_to_px(cfg["light_roi"], img.shape)
        cv2.rectangle(img, (x0, y0), (x1, y1), STATE_COLOR[state], 2)
        # bóng đèn đang sáng mà mô hình A thực sự đọc (khung trắng mảnh)
        bx = khung_bong_sang(frame, light_box_rel(cfg["light_roi"]))
        Hf, Wf = frame.shape[:2]
        cv2.rectangle(img, (int(bx[0] * Wf), int(bx[1] * Hf)), (int(bx[2] * Wf), int(bx[3] * Hf)), (255, 255, 255), 1)
        lbl = f"A: {STATE_ASCII[state]}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ty = y0 - 6 if y0 - 6 >= th + 32 else y1 + th + 6
        cv2.putText(img, lbl, (max(2, min(x0, W - tw - 6)), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATE_COLOR[state], 2)
    if r:
        for k in r["K"]:                                    # khung B tìm (mảnh)
            x0, y0, x1, y1 = scale_box(k, sx, sy)
            cv2.rectangle(img, (x0, y0), (x1, y1), (80, 210, 60), 1)
        canh_ids = {c["id"] for c in r["canh_bao"]}
        for g in r["giu"]:                                  # xe C ghép giữa hai frame (đậm), có ID
            x0, y0, x1, y1 = scale_box(g["khung_b"], sx, sy)
            bad = g["id"] in canh_ids or g["id"] in (tracks_bottom or {}).get("_viol", set())
            color = BGR_RED if bad else BGR_GREEN
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 4 if bad else 2)
            label = f"ID {g['id']}" + (" VUOT DEN DO" if bad else "")
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(img, label, (max(2, min(x0, W - tw - 4)), max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            pts = (tracks_bottom or {}).get(g["id"], [])
            if len(pts) >= 2:                               # quỹ đạo đáy khung (đầu ra ID + quỹ đạo của C)
                arr = np.array([(int(p[0] * sx), int(p[1] * sy)) for _, p in pts[-12:]], np.int32)
                cv2.polylines(img, [arr], False, (0, 255, 255), 2, cv2.LINE_AA)
    (iw, _), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(img, (0, 0), (min(W, iw + 16), 28), (20, 20, 20), -1)
    cv2.putText(img, info_text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return img


def analyze_sequence(frames, cfg, total, progress=None, writer=None):
    """Chạy hệ thống trên chuỗi khung hình (giống chay_video của notebook).

    frames: iterator (thời_gian_giây, khung_BGR đã cùng kích thước hiển thị).
    Mỗi frame: A đọc đèn → B quét cửa sổ trượt quanh vạch → C ghép xe với frame trước (Hungarian)
    → xe C ghép đi qua vạch khi A báo đỏ ⇒ cảnh báo (mỗi ID một lần).
    """
    bt = BoTheoDoi(HT, cfg["y_vach"] / 100,
                   light_box_rel(cfg["light_roi"]) if cfg["light_mode"] == "auto" else None,
                   che_do_den=cfg["light_mode"], xe_di_len=cfg["direction"] == "up",
                   nguong_xe=cfg["nguong_xe"], nguong_C=cfg["nguong_C"], lam_muot=cfg["lam_muot"],
                   he_so_co=cfg.get("he_so_co", 1.0), nen=cfg.get("nen"))
    res = dict(violations=[], timeline=[], annotated=[], n_frames=0, crossed_ok=0, light_samples={},
               light_sample=None, vehicle_sample=None, n_boxes=0, n_windows=0, t_model=0.0, p_light=[])
    viol_ids = set()
    t_start = time.time()
    for i, (t_sec, frame) in enumerate(frames):
        t0 = time.time()
        r = bt.buoc(to640(frame), t_sec)
        res["t_model"] += time.time() - t0
        res["timeline"].append((round(t_sec, 2), r["den"]))
        res["p_light"].append((round(t_sec, 2), r["xac_suat_den"]))
        res["crossed_ok"] += r["vuot_hop_le"]
        res["n_boxes"] += len(r["K"])
        res["n_windows"] = r["so_cua_so"]
        H, W = frame.shape[:2]
        sx, sy = W / 640, H / 640
        if cfg["light_mode"] == "auto":
            x0, y0, x1, y1 = pct_rect_to_px(cfg["light_roi"], frame.shape)
            roi = frame[y0:y1, x0:x1]
            if roi.size and r["den"] not in res["light_samples"]:
                res["light_samples"][r["den"]] = roi.copy()
            if res["light_sample"] is None and r["den"] in ("RED", "GREEN", "YELLOW") and roi.size:
                res["light_sample"] = roi.copy()
        for k in r["K"]:
            area = (k[2] - k[0]) * (k[3] - k[1])
            if res["vehicle_sample"] is None or area > res["vehicle_sample"][1]:
                x0, y0, x1, y1 = scale_box(k, sx, sy)
                crop = frame[max(0, y0):y1, max(0, x0):x1]
                if crop.size:
                    res["vehicle_sample"] = (crop.copy(), area)
        for c in r["canh_bao"]:
            viol_ids.add(c["id"])
        tracks = dict(bt.quy_dao)
        tracks["_viol"] = viol_ids
        info = (f"t={t_sec:5.1f}s | A: den {STATE_ASCII[r['den']]} | B: {len(r['K'])} xe | "
                f"C: {len(r['giu'])} cap | Canh bao: {len(res['violations']) + len(r['canh_bao'])}")
        out = draw_result(frame, cfg, r, info, tracks)
        for c in r["canh_bao"]:
            x0, y0, x1, y1 = scale_box(c["khung_b"], sx, sy)
            ev = out.copy()
            pts = bt.quy_dao.get(c["id"], [])
            if len(pts) >= 2:
                arr = np.array([(int(p[0] * sx), int(p[1] * sy)) for _, p in pts], np.int32)
                cv2.polylines(ev, [arr], False, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.arrowedLine(ev, tuple(arr[-2]), tuple(arr[-1]), (0, 255, 255), 2, tipLength=0.5)
            p_do = r["xac_suat_den"].get("red_light")
            res["violations"].append({
                "MaXe": f"ID-{c['id']:03d}",
                "ThoiDiem": f"{t_sec:.1f} s",
                "TrangThaiDen": STATE_VN[r["den"]],
                "XacSuatDo": None if p_do is None else float(p_do),
                "XacSuatCungXe": float(c["p_C"]),
                "DiemXe": float(c["p_b"]),
                "ViTri": f"Đáy khung qua vạch tại ({int((c['khung_b'][0] + c['khung_b'][2]) / 2 * sx)}, {int(c['khung_b'][3] * sy)})",
                "snapshot": encode_jpg(ev),
                "crop": encode_jpg(crop_box(frame, (x0, y0, x1 - x0, y1 - y0))),
            })
        if writer is not None:
            writer(out)
        else:
            res["annotated"].append(out)
        res["n_frames"] += 1
        if progress is not None and total:
            el = time.time() - t_start
            eta = el / (i + 1) * (total - i - 1)
            progress.progress(min(1.0, (i + 1) / total),
                              text=f"Đang xử lý frame {i + 1}/{total} · còn khoảng {eta:.0f} giây "
                                   f"(B quét {r['so_cua_so']:,} cửa sổ / frame bằng SVM-RBF)")
    res["n_vehicles"] = bt.next_id - 1
    return res


def analyze_video(video_path, cfg, progress, start=0.0, end=None, box=None):
    """Lấy frame cách nhau `buoc_giay` giây (giống tham số buoc_giay của chay_video trong notebook)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end_t = end if end is not None else (n_total / fps if n_total else start + cfg["buoc_giay"] * cfg["max_frames"])
    wanted = sorted({int(round(t * fps)) for t in np.arange(start, end_t + 1e-9, cfg["buoc_giay"])})[:cfg["max_frames"]]
    if not wanted:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    ok, first = cap.read()
    if not ok:
        cap.release()
        return None
    first = resize_frame(crop_rel(first, box))
    size = (first.shape[1], first.shape[0])
    want = set(wanted)
    if cfg.get("tru_nen", True):
        # ảnh nền của camera = trung vị 25 khung rải đều trên đoạn được xét (xe chạy qua bị loại)
        mau = []
        # lấy rải trên cả video (không chỉ đoạn được xét) để xe đứng chờ đèn lâu ít bị lẫn vào nền
        lo_f, hi_f = (0, n_total - 1) if n_total > len(wanted) else (wanted[0], wanted[-1])
        for fi in np.linspace(lo_f, hi_f, num=min(40, max(3, n_total or len(wanted)))).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok_, f_ = cap.read()
            if ok_:
                mau.append(to640(resize_frame(crop_rel(f_, box), size)))
        cfg = dict(cfg, nen=anh_nen_trung_vi(mau))

    def gen():
        cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
        idx, got = wanted[0], 0
        while got < len(wanted):
            ok_, f = cap.read()
            if not ok_:
                break
            if idx in want:
                got += 1
                yield idx / fps, resize_frame(crop_rel(f, box), size)
            idx += 1

    out_fps = max(1.0, min(4.0, 1.0 / cfg["buoc_giay"]))
    out_path = os.path.join(tempfile.mkdtemp(), "ket_qua.mp4")
    write, close, browser_ok = open_video_writer(out_path, out_fps, size)
    try:
        res = analyze_frames_wrap(gen(), cfg, len(wanted), progress, write)
    finally:
        close()
        cap.release()
    with open(out_path, "rb") as fh:
        res["video_bytes"] = fh.read()
    res["video_browser_ok"] = browser_ok
    res["duration"] = (wanted[min(res["n_frames"], len(wanted)) - 1] - wanted[0]) / fps if res["n_frames"] else 0.0
    return res


def analyze_frames_wrap(frames, cfg, total, progress, writer=None):
    return analyze_sequence(frames, cfg, total, progress=progress, writer=writer)


def decode_upload_image(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="tl-icon"><span class="r"></span><span class="y"></span><span class="g"></span></div>
        <div><div class="t">Cảnh báo vượt đèn đỏ</div><div class="s">Đồ án Học Máy Truyền Thống</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Mô hình đang chạy (từ notebook)</div>', unsafe_allow_html=True)
    rows_html = "".join(
        f'<div class="row"><span>{k} · {m["ten"]}</span><b>{m["thuat_toan"]}</b></div>'
        f'<div class="row" style="padding-top:0"><span style="color:{C_DIM};font-size:.78rem">{m["cau_hinh"]}</span>'
        f'<b style="color:{C_GREEN};font-size:.8rem">{fmt(m["chi_so"][0][1])}</b></div>'
        for k, m in MODELS.items())
    st.markdown(f'<div class="sb-sum">{rows_html}</div>', unsafe_allow_html=True)
    st.caption("Số xanh lá: macro F1 test (A), F1 xe/nền test (B), AUC ngoài fold (C). "
               "Chỉ các mô hình được chọn ở notebook mới được lưu vào file joblib, nên web chạy đúng bộ này.")

    st.caption(f"Phiên bản lõi: {PHIEN_BAN_LOI} · đọc màu đèn: "
               f"{'mô hình A + vị trí bóng sáng' if hasattr(HT, 'doan_den_chi_tiet') else 'chỉ mô hình A (lõi cũ)'}")
    st.markdown('<div class="sb-sec">Ngưỡng của hệ thống</div>', unsafe_allow_html=True)
    nguong_xe = st.slider("Ngưỡng B (xác suất là xe)", 0.5, 0.99, float(HT.nguong_xe), 0.01,
                          help=f"Một vùng ảnh được coi là xe khi B chấm xác suất ≥ ngưỡng này. Cao hơn: ít khung nhầm "
                               f"nhưng dễ sót xe; thấp hơn: ngược lại. Mặc định {HT.nguong_xe:g} (chọn trên 60 khung val). "
                               f"Ngưỡng này chỉ đổi khung xe được vẽ; C1 luôn xét mọi khung có điểm ≥ {HT.nguong_vao_C:g}.")
    nguong_C = st.slider("Ngưỡng C (xác suất cùng một xe)", 0.1, 0.9, float(HT.nguong_C), 0.05,
                         help=f"Hai khung ở hai ảnh liên tiếp chỉ được ghép là cùng một xe khi C1 chấm xác suất ≥ ngưỡng này. "
                              f"Cao hơn: ghép chắc chắn, ít cảnh báo nhầm nhưng dễ sót; thấp hơn: ngược lại. "
                              f"Mặc định {HT.nguong_C:g} (chọn theo F1 vượt vạch ngoài fold trên train).")
    lam_muot = st.slider("Làm mượt màu đèn: số frame bỏ phiếu", 1, 5, 3,
                         help="Màu đèn ở mỗi frame là màu xuất hiện nhiều nhất trong N frame gần nhất, để một frame đoán "
                              "sai không làm đổi màu. N = 1: không làm mượt. N lớn: ổn định hơn nhưng đổi màu chậm hơn "
                              "khi đèn chuyển. Mặc định 3 như notebook.")
    with st.expander("Giải thích các ngưỡng"):
        st.markdown(
            f"""
**Ngưỡng B = {HT.nguong_xe:g} (xác suất là xe).** Mô hình B (SVM-RBF) chấm mỗi cửa sổ quét một xác suất "là xe".
Chỉ cửa sổ có xác suất từ ngưỡng này trở lên mới thành khung xe trên màn hình.
- Tăng ngưỡng: ít khung nhầm (vẽ cả cây, vạch sơn…), nhưng dễ bỏ sót xe.
- Giảm ngưỡng: bắt được nhiều xe hơn, nhưng nhiều khung nhầm hơn.
- {HT.nguong_xe:g} là ngưỡng cho F1 tìm xe cao nhất trên 60 khung val (test: F1 0,447; Precision 0,538; Recall 0,383).
- Mô hình C1 vẫn nhận mọi khung có điểm ≥ {HT.nguong_vao_C:g} để ghép, nên ngưỡng B chủ yếu đổi phần hiển thị.

**Ngưỡng C = {HT.nguong_C:g} (xác suất cùng một xe).** Với mỗi cặp khung ở hai ảnh liên tiếp, mô hình C1 (k-NN) chấm xác
suất hai khung là cùng một xe. Cặp có xác suất ≥ ngưỡng mới được ghép; thuật toán Hungary đảm bảo mỗi xe chỉ ghép với một xe.
- Tăng ngưỡng: chỉ ghép cặp rất chắc chắn, ít cảnh báo nhầm, nhưng bỏ sót nhiều lần vượt vạch hơn.
- Giảm ngưỡng: ghép được nhiều hơn, nhưng dễ ghép nhầm hai xe khác nhau và báo nhầm.
- {HT.nguong_C:g} là ngưỡng cho F1 vượt vạch cao nhất khi dự đoán ngoài fold trên train (không dùng test để chọn).

**Mô hình A đọc bóng nào.** Bạn chọn cả hộp đèn; web tự tìm bóng đang sáng trong hộp (khung trắng mảnh trên hình) và chỉ
đưa vùng bóng đó vào A, vì ảnh đèn xanh, đèn đỏ trong dữ liệu huấn luyện chủ yếu cắt sát bóng sáng.

**Cỡ xe so với camera huấn luyện.** B quét cửa sổ có cỡ cố định theo 4 camera của dataset. Camera gần hoặc zoom thì xe to
hơn cửa sổ, B chỉ ôm được một phần xe; khi đó tăng hệ số (1,5–2) và bấm “Thử B trên khung này” để kiểm tra.

**Làm mượt màu đèn = 3 frame.** Mô hình A đoán màu đèn ở từng frame. Màu dùng để kết luận là màu xuất hiện nhiều nhất
trong 3 frame gần nhất, để một frame đoán sai (đèn lóa, bị che) không biến đèn xanh thành đỏ hay ngược lại.

**Khi nào cảnh báo:** cùng một xe (C1 ghép) có đáy khung ở ảnh trước nằm trước vạch dừng, ở ảnh sau đã qua vạch, và màu
đèn (sau khi làm mượt) là đỏ. Mỗi xe chỉ bị cảnh báo một lần.
""")
    if (nguong_xe, nguong_C, lam_muot) != (HT.nguong_xe, HT.nguong_C, 3):
        st.caption("⚠ Đang khác cấu hình notebook: số liệu đánh giá ở tab Hiệu năng không còn áp dụng.")


# ==========================================
# HEADER
# ==========================================
st.markdown(f"""
<div class="hero">
    <div class="hero-left">
        <div class="tl-icon"><span class="r"></span><span class="y"></span><span class="g"></span></div>
        <div>
            <h1>Phát hiện & cảnh báo xe vượt đèn đỏ</h1>
            <p>Chạy đúng ba mô hình máy học truyền thống đã huấn luyện trong notebook, không dùng học sâu</p>
        </div>
    </div>
    <div class="chips">
        <span class="chip">A · <b>{MODELS['A']['thuat_toan']}</b> trên HSV</span>
        <span class="chip">B · <b>{MODELS['B']['thuat_toan']}</b> trên HOG+PCA</span>
        <span class="chip">C · <b>{MODELS['C']['thuat_toan']}</b> + Hungarian</span>
    </div>
</div>
""", unsafe_allow_html=True)


def _stage(k, color, m):
    name, val = m["chi_so"][0]
    return (f'<div class="stage" style="--c:{color}"><div class="k">Mô hình {k} · {m["ten"]}</div>'
            f'<div class="v">{m["thuat_toan"]}</div><div class="s">{name} <b>{fmt(val)}</b></div></div>')


st.markdown(f"""
<div class="strip">
    {_stage("A", C_ACCENT, MODELS["A"])}
    {_stage("B", "#A78BFA", MODELS["B"])}
    {_stage("C", "#F472B6", MODELS["C"])}
    <div class="stage" style="--c:{C_RED}"><div class="k">Nhật ký vi phạm</div>
        <div class="v">{len(st.session_state.violations_log)} lượt</div><div class="s">Lưu trong phiên làm việc này</div></div>
</div>
""", unsafe_allow_html=True)

tab_real, tab_bench, tab_feat, tab_log = st.tabs([
    "Phân tích ảnh / video", "Hiệu năng mô hình", "Đặc trưng A & B", "Nhật ký vi phạm",
])

# ==========================================
# TAB 1: PHÂN TÍCH ẢNH / VIDEO
# ==========================================
with tab_real:
    step(1, "Dữ liệu đầu vào", "Video hoặc chuỗi ảnh từ camera đặt cố định tại ngã tư")
    input_type = st.segmented_control("Loại dữ liệu", ["🎥 Video", "🖼️ Ảnh"], default="🎥 Video",
                                      label_visibility="collapsed") or "🎥 Video"
    preview, source_key, video_path, images = None, None, None, []
    samples_avail = [p for p in (NOTEBOOK_SAMPLE, SAMPLE_VIDEO) if os.path.exists(p)]

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
        elif st.session_state.use_sample and st.session_state.get("sample_path") in samples_avail:
            video_path = st.session_state.sample_path
            source_key = f"vid:{os.path.basename(video_path)}:sample"
            c_s1, c_s2 = st.columns([4, 1])
            c_s1.caption(f"Đang dùng **video mẫu** ({os.path.basename(video_path)}). Tải video của bạn lên để thay thế.")
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
                    banner("<b>Camera trong cảnh này đang zoom hoặc lia.</b> Vạch dừng khai báo một lần cho cả đoạn nên "
                           "sẽ lệch khỏi vạch thật khi camera chuyển động. Nên chọn cảnh khác.", good="warn")
                if scene["direction"] and st.session_state.get("dir_applied_for") != source_key:
                    st.session_state.direction = DIR_UP if scene["direction"] == "up" else DIR_DOWN
                    st.session_state.dir_applied_for = source_key
                st.session_state.dir_guess = (scene["direction"], scene["conf"])
                if preview is None:
                    banner("Không đọc được khung hình của cảnh này.", good=False)
    else:
        ups = st.file_uploader("Ảnh cần kiểm tra: 1 ảnh, hoặc nhiều ảnh liên tiếp cùng góc camera",
                               type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True,
                               help="Từ 2 ảnh trở lên, mô hình C ghép xe giữa từng cặp ảnh liền nhau để biết xe có đi qua "
                                    "vạch không (giống hàm chay_anh_thuc_te của notebook). 1 ảnh: chỉ chạy A và B. "
                                    "Ảnh được xếp theo tên file, nên đặt tên theo thứ tự thời gian: 01.jpg, 02.jpg, 03.jpg… "
                                    "hoặc anh_1.jpg, anh_2.jpg… (web so số theo giá trị, nên anh_2 đứng trước anh_10).")
        st.caption("Thứ tự ảnh = thứ tự tên file. Đặt tên theo thời gian chụp, ví dụ 01.jpg, 02.jpg, 03.jpg "
                   "(ảnh chụp sớm hơn đặt số nhỏ hơn). Mỗi cặp ảnh liền nhau nên cách nhau khoảng 0,3 giây.")
        if ups:
            ups = sorted(ups, key=lambda f: _khoa_ten(f.name))
            decoded = [decode_upload_image(u) for u in ups]
            if any(d is None for d in decoded):
                banner("Có ảnh không đọc được. Hãy dùng ảnh JPG hoặc PNG.", good=False)
            else:
                first = resize_frame(decoded[0])
                size = (first.shape[1], first.shape[0])
                images = [first] + [resize_frame(d, size) for d in decoded[1:]]
                preview = images[0]
                source_key = "img:" + "|".join(f"{u.name}:{u.size}" for u in ups)

    if preview is None:
        st.markdown("""
        <div class="empty">
            <div class="big">🚦</div>
            <h4>Chưa có dữ liệu</h4>
            <p>Tải lên video quay ngã tư từ camera cố định. Mô hình A đọc màu đèn, mô hình B tìm xe quanh vạch dừng,
            mô hình C ghép cùng một xe giữa các frame, rồi bước kết luận đánh dấu xe đi qua vạch khi đèn đỏ.</p>
        </div>""", unsafe_allow_html=True)
        if input_type == "🎥 Video" and samples_avail:
            st.write("")
            cols_s = st.columns([1] + [1.2] * len(samples_avail) + [1])
            for j, p in enumerate(samples_avail):
                is_nb = p == NOTEBOOK_SAMPLE
                lbl = "▶  Video test của notebook" if is_nb else "▶  Thử với video mẫu"
                if cols_s[j + 1].button(lbl, type="primary" if j == 0 else "secondary", width="stretch", key=f"sample_{j}"):
                    st.session_state.use_sample = True
                    st.session_state.sample_path = p
                    key_ = f"vid:{os.path.basename(p)}:sample"
                    if is_nb:   # vạch dừng và khung đèn khai báo như notebook mục F1 (y ≈ 0,42; đèn ở góc phải)
                        st.session_state.pending_cfg = dict(rx=70, ry=15, rw=2, rh=5, ly=42, light_mode="auto",
                                                            buoc_giay=0.5, direction=DIR_UP)
                    else:
                        st.session_state.pending_cfg = dict(rx=87, ry=3, rw=7, rh=30, ly=65, light_mode="auto")
                    st.session_state.sample_confirmed = key_      # cấu hình sẵn đúng cho mọi cảnh của video mẫu
                    st.session_state.dir_applied_for = key_
                    st.rerun()
        st.write("")
        st.markdown(f"""
        <div class="flow">
            <div><b>① Mô hình A · {MODELS['A']['thuat_toan']}</b>44 đặc trưng màu HSV của vùng đèn → đỏ / vàng / xanh.</div>
            <div><b>② Mô hình B · {MODELS['B']['thuat_toan']}</b>Cửa sổ trượt quanh vạch dừng, HOG → PCA 150 → xe hay nền.</div>
            <div><b>③ Mô hình C · {MODELS['C']['thuat_toan']}</b>21 đặc trưng cặp khung → cùng một xe? Hungarian ghép một-một, ra ID.</div>
            <div><b>④ Bước kết luận</b>Xe C ghép có đáy khung đi qua vạch dừng khi A báo đỏ ⇒ cảnh báo.</div>
        </div>""", unsafe_allow_html=True)

    else:
        step(2, "Hiệu chỉnh trên hình", "Bấm trực tiếp lên ảnh để chọn đèn và đặt vạch dừng")

        for k_, v_ in dict(ly=65, rx=85, ry=5, rw=10, rh=25).items():
            st.session_state.setdefault(k_, v_)
        st.session_state.setdefault("light_mode", "auto")
        st.session_state.setdefault("buoc_giay", 0.3)
        st.session_state.setdefault("direction", DIR_UP)
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
                    "Công cụ bấm trên ảnh", ["🚦 Chọn đèn", "📏 Đặt vạch dừng"], default="🚦 Chọn đèn", key="tool",
                ) or "🚦 Chọn đèn"
                if tool == "🚦 Chọn đèn":
                    layout_opts = {"auto": "Tự động", "vertical": "Dọc", "horizontal": "Ngang", "lamp": "Chỉ bóng"}
                    st.segmented_control("Kiểu hộp đèn", list(layout_opts), format_func=layout_opts.get,
                                         default="auto", key="light_layout",
                                         help="Khung tự ôm cả hộp đèn. Nếu đoán sai kiểu đèn (dọc/ngang), chọn lại ở đây.")
                    st.markdown(
                        "Bấm vào **đèn tròn điều khiển làn xe cần giám sát**. Hệ thống tự ôm khung quanh đèn, "
                        "mô hình A đọc màu trong khung đó.  \n"
                        f"<span style='color:{C_AMBER}'>Không chọn đèn mũi tên rẽ</span> nếu làn đó có đèn riêng.",
                        unsafe_allow_html=True)
                else:
                    st.markdown("Bấm **một điểm trên vạch dừng**. Vạch được coi là đường ngang (như notebook: `y_vach`). "
                                "Dải xanh nhạt là vùng B quét tìm xe: từ 20% phía trên tới 15% phía dưới vạch.")
                direction_lbl = st.radio("Hướng xe chạy qua vạch", [DIR_UP, DIR_DOWN], horizontal=False, key="direction",
                                         help="Mô hình C và bước kết luận được học với xe đi từ dưới lên (ra xa camera, "
                                              "XE_DI_LEN = True). Hướng ngược lại vẫn chạy được nhưng C chưa được học cho hướng đó.")
                direction = "up" if direction_lbl == DIR_UP else "down"
                guess = st.session_state.get("dir_guess") if input_type == "🎥 Video" else None
                if guess and guess[0]:
                    g_lbl = "từ trên xuống ↓" if guess[0] == "down" else "từ dưới lên ↑"
                    st.caption(f"Ước lượng bằng optical flow: xe chủ yếu chạy **{g_lbl}** · độ tin cậy {guess[1] * 100:.0f}%")
                if direction == "down":
                    st.caption("⚠ Dữ liệu huấn luyện chỉ có xe đi từ dưới lên, kết quả hướng này kém tin cậy hơn.")
                light_options = {
                    "auto": "Mô hình A đọc màu đèn",
                    "red": "Không thấy đèn trong hình · coi như luôn đỏ",
                    "green": "Coi như luôn xanh",
                }
                light_mode = st.selectbox("Trạng thái đèn", list(light_options), format_func=light_options.get,
                                          key="light_mode")

            he_so_co = st.select_slider(
                "Cỡ xe so với camera huấn luyện", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0], value=1.0, key="he_so_co",
                help="Mô hình B quét cửa sổ có cỡ cố định theo camera trong dataset. Nếu xe trong video của bạn to hơn "
                     "(camera gần, zoom) thì tăng lên, nhỏ hơn thì giảm. Bấm “Thử B trên khung này” để xem khung xe có ôm "
                     "vừa xe không. 1,0 = đúng như notebook.")
            tru_nen = st.checkbox(
                "Tìm thêm xe ở vùng có chuyển động (camera đứng yên)", value=True, key="tru_nen",
                help="Web lấy ảnh nền của camera (trung vị nhiều khung), tìm các vùng khác nền quanh vạch dừng, rồi cho "
                     "mô hình B chấm từng vùng là xe hay nền. Giúp B không bỏ sót xe to hoặc góc camera khác dataset. "
                     "Đây là bổ sung của web; số liệu trong báo cáo đo bằng notebook không có bước này.")
            with st.expander("Chỉnh chi tiết bằng thanh trượt"):
                ly = st.slider("Vạch dừng · vị trí dọc (% chiều cao)", 0, 100, key="ly")
                st.caption("Vùng đèn (% khung hình)")
                b1, b2 = st.columns(2)
                rx = b1.slider("Vị trí ngang", 0, 99, key="rx", on_change=_confirm_light)
                ry = b1.slider("Vị trí dọc", 0, 99, key="ry", on_change=_confirm_light)
                rw = b2.slider("Chiều rộng", 1, 50, key="rw", on_change=_confirm_light)
                rh = b2.slider("Chiều cao", 1, 60, key="rh", on_change=_confirm_light)

            with st.expander("Lấy mẫu frame (tốc độ)"):
                buoc_giay = st.select_slider("Khoảng cách giữa hai frame được xét (giây)",
                                             [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0], key="buoc_giay",
                                             help="Như buoc_giay của notebook (mặc định 0,3 s): xe nên dịch khoảng một chiều "
                                                  "cao xe trở xuống giữa hai frame để C ghép được.")
                max_frames = st.slider("Số frame tối đa", 5, 150, 40,
                                       help="B quét hàng nghìn cửa sổ bằng SVM-RBF: khoảng 3 đến 6 giây mỗi frame trên CPU.")
                st.caption(f"Ước tính: {max_frames} frame × ~3–6 s ≈ {max_frames * 3 // 60}–{max(1, max_frames * 6 // 60)} phút. "
                           f"Đoạn video được xét: {max_frames * buoc_giay:.0f} giây.")

        cfg = dict(y_vach=float(ly), direction=direction, light_roi=(rx, ry, rw, rh), light_mode=light_mode,
                   nguong_xe=nguong_xe, nguong_C=nguong_C, lam_muot=lam_muot, buoc_giay=float(buoc_giay),
                   he_so_co=float(he_so_co), tru_nen=bool(tru_nen),
                   max_frames=int(max_frames))

        if (st.session_state.get("sample_confirmed") and source_key
                and source_key.split("#")[0] == st.session_state.sample_confirmed):
            st.session_state.light_confirmed_for = source_key
            st.session_state.dir_applied_for = source_key
        light_ok = light_mode != "auto" or st.session_state.get("light_confirmed_for") == source_key

        with col_prev:
            Hp, Wp = preview.shape[:2]
            x0, y0, x1, y1 = pct_rect_to_px(cfg["light_roi"], preview.shape)
            light_roi_img = preview[y0:y1, x0:x1]
            probs = {}
            if not light_ok:
                prev_state = "UNKNOWN"
            elif light_mode == "auto":
                prev_state, probs = predict_light(preview, cfg["light_roi"])
            else:
                prev_state = "RED" if light_mode == "red" else "GREEN"
            fake = dict(den=prev_state, K=np.zeros((0, 4)), giu=[], canh_bao=[])
            b_prev = st.session_state.get("b_preview")
            if b_prev and b_prev[0] == (source_key, cfg["y_vach"], nguong_xe, cfg["he_so_co"]):
                fake["K"] = b_prev[1]
            hint = "Xem truoc" if light_ok else "BAM VAO DEN GIAO THONG DE CHON DEN"
            cfg_prev = dict(cfg, light_mode=light_mode if light_ok else "none")
            prev_img = draw_result(preview, cfg_prev, fake, hint)
            ly_px = int(cfg["y_vach"] / 100 * Hp)
            dy = -45 if direction == "up" else 45
            cv2.arrowedLine(prev_img, (Wp // 2, ly_px - dy), (Wp // 2, ly_px + dy // 3), (248, 189, 56), 3, tipLength=0.35)

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
                        st.session_state.pending_cfg = dict(ly=int(round(cy / Hp * 100)))
                    st.rerun()

                if st.session_state.get("light_miss") == source_key and tool == "🚦 Chọn đèn":
                    banner("<b>Chỗ vừa bấm không có đèn đang sáng</b>, nên chưa đặt khung đèn. Hãy bấm đúng vào bóng đèn "
                           "đang sáng. Nếu muốn đặt vạch dừng, chọn công cụ 📏 trước.", good=False)
                if not light_ok:
                    banner("<b>Chưa chọn đèn.</b> Chọn công cụ 🚦 rồi bấm vào đúng <b>đèn tròn</b> điều khiển làn xe cần "
                           "giám sát. Nếu video không quay được đèn, chọn “coi như luôn đỏ” ở mục Trạng thái đèn.", good="warn")
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
                z1, z2, z3 = st.columns([1, 3, 1.6], vertical_alignment="center")
                if light_ok and light_mode == "auto" and light_roi_img.size:
                    zoom = cv2.resize(light_roi_img, (96, max(1, int(96 * light_roi_img.shape[0] / max(light_roi_img.shape[1], 1)))),
                                      interpolation=cv2.INTER_NEAREST)
                    z1.image(zoom, channels="BGR", width="stretch")
                if light_ok and light_mode != "auto":
                    z2.markdown(f"Không dùng camera để đọc đèn &nbsp; {pill(prev_state)}", unsafe_allow_html=True)
                elif light_ok:
                    ptxt = " · ".join(f"{STATE_VN[{'red_light': 'RED', 'yellow_light': 'YELLOW', 'green_light': 'GREEN'}[k]]} "
                                      f"{v:.2f}" for k, v in probs.items())
                    ct_ = st.session_state.get("light_detail") or {}
                    _vn = {"red_light": "đỏ", "yellow_light": "vàng", "green_light": "xanh"}
                    ctxt = ""
                    if ct_.get("A"):
                        a_lop = max(ct_["A"], key=ct_["A"].get)
                        ctxt = (f"<br><span style='color:{C_MUTED};font-size:.85rem'>Riêng mô hình A: {_vn[a_lop]} "
                                f"{ct_['A'][a_lop]:.2f} · vị trí bóng sáng: "
                                f"{_vn[ct_['vi_tri']] if ct_.get('vi_tri') else 'không dùng (hộp đèn không rõ 3 bóng)'}</span>")
                    z2.markdown(f"Màu đèn &nbsp; {pill(prev_state)}<br>"
                                f"<span style='color:{C_MUTED};font-size:.85rem'>Xác suất: {ptxt}</span>{ctxt}",
                                unsafe_allow_html=True)
                if z3.button("Thử B trên khung này", width="stretch",
                             help="Chạy mô hình B (SVM-RBF) trên khung xem trước, mất vài giây."):
                    with st.spinner("Mô hình B đang quét cửa sổ trượt..."):
                        fr_ = HT.quet_frame(to640(preview), cfg["y_vach"] / 100, cfg["he_so_co"])
                        Kb_, _, _, _ = HT.khung_xe(fr_, nguong_xe)
                    st.session_state.b_preview = ((source_key, cfg["y_vach"], nguong_xe, cfg["he_so_co"]), Kb_)
                    st.rerun()
                if b_prev and b_prev[0] == (source_key, cfg["y_vach"], nguong_xe, cfg["he_so_co"]):
                    st.caption(f"Mô hình B tìm thấy {len(b_prev[1])} khung xe trong dải quét (khung xanh mảnh).")

        step(3, "Phân tích", "")
        if direction == "up" and cfg["y_vach"] > 85:
            banner("Vạch dừng đang sát mép dưới khung hình. Hệ thống chỉ báo vượt vạch khi thấy đáy xe <b>trước</b> vạch ở "
                   "khung này và <b>sau</b> vạch ở khung kế tiếp; vạch sát mép thì xe chưa kịp hiện ra trước vạch nên không "
                   "bao giờ được tính là vượt. Hãy đặt vạch đúng chỗ vạch sơn thật, chừa ít nhất khoảng 15% chiều cao bên dưới.",
                   good="warn")
        elif direction == "down" and cfg["y_vach"] < 15:
            banner("Vạch dừng đang sát mép trên khung hình: xe đi từ trên xuống chưa kịp hiện ra trước vạch nên không thể "
                   "tính là vượt. Hãy đặt vạch đúng chỗ vạch sơn thật, chừa ít nhất khoảng 15% chiều cao bên trên.",
                   good="warn")
        st.caption("Lưu ý: ba mô hình chỉ học từ 4 camera của dataset (ảnh 640×640). Trên camera khác, kết quả kém đi rõ rệt: "
                   "notebook đo trên 60 ảnh ngoài 4 camera, B chỉ đạt F1 tìm xe 0,078; với đoạn video chưa thấy, A còn "
                   "macro F1 0,700. Chỉnh “Cỡ xe so với camera huấn luyện” và chọn đúng hộp đèn sẽ giúp phần nào, "
                   "nhưng muốn chạy tốt trên camera mới cần thêm dữ liệu của camera đó để huấn luyện lại.")
        can_run = light_ok and not (input_type == "🖼️ Ảnh" and not images)
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
                    if cfg.get("tru_nen", True) and len(images) >= 3:
                        cfg = dict(cfg, nen=anh_nen_trung_vi([to640(im) for im in images]))
                    res = analyze_frames_wrap(frames, cfg, len(images), progress)
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
                lbl_ = (st.session_state.get("scene_info") or {}).get("label") if input_type == "🎥 Video" else ""
                for v in res["violations"]:
                    st.session_state.violations_log.append({
                        "Mã xe (ID của C)": v["MaXe"], "Thời điểm": v["ThoiDiem"], "Đèn (A)": v["TrangThaiDen"],
                        "P(đỏ) của A": None if v["XacSuatDo"] is None else round(v["XacSuatDo"], 3),
                        "P(cùng xe) của C": round(v["XacSuatCungXe"], 3), "P(xe) của B": round(v["DiemXe"], 3),
                        "Vị trí": v["ViTri"], "Nguồn": src_name + (f" · {lbl_}" if lbl_ else ""), "Phân tích lúc": now,
                    })
                st.rerun()

        # ---------- Kết quả ----------
        res = st.session_state.analysis
        if res is not None and res.get("source") == source_key:
            step(4, "Kết quả", "")
            n_v = len(res["violations"])
            st.markdown(f"""
            <div class="kpis">
                <div class="kpi"><div class="k">Xe được C theo dõi (ID)</div><div class="v">{res['n_vehicles']}</div></div>
                <div class="kpi {'bad' if n_v else 'good'}"><div class="k">Cảnh báo vượt đèn đỏ</div><div class="v">{n_v}</div></div>
                <div class="kpi good"><div class="k">Qua vạch khi không đỏ</div><div class="v">{res['crossed_ok']}</div></div>
                <div class="kpi"><div class="k">Frame đã xét</div><div class="v">{res['n_frames']}</div></div>
            </div>""", unsafe_allow_html=True)
            if res["kind"] == "image" and res["n_frames"] == 1:
                banner("Chỉ có 1 ảnh nên mới chạy được A và B. Cần ít nhất 2 ảnh liên tiếp để C ghép xe và biết xe có "
                       "đi qua vạch hay không.", good="warn")
            elif n_v:
                banner(f"Phát hiện {n_v} phương tiện đi qua vạch dừng khi mô hình A báo đèn đỏ.", good=False)
            else:
                banner("Không có phương tiện nào được C ghép đi qua vạch dừng khi đèn đỏ.", good=True)
            st.caption(f"Thời gian chạy mô hình: {res['t_model']:.0f} s cho {res['n_frames']} frame "
                       f"({res['t_model'] / max(res['n_frames'], 1):.1f} s/frame, B quét {res['n_windows']:,} cửa sổ mỗi frame).")

            seen_states = {s for _, s in res["timeline"]}
            if res.get("light_mode") == "auto" and res["timeline"] and "RED" not in seen_states:
                seen_txt = ", ".join(STATE_VN[s].lower() for s in seen_states)
                banner(f"<b>Suốt đoạn đã xét, mô hình A chưa lần nào báo ĐỎ</b> (chỉ thấy: {seen_txt}). "
                       "Nếu thực tế đèn đỏ, có thể khung đèn đặt sai chỗ. Hãy bấm lại vào đúng đèn tròn ở Bước 2.", good="warn")
            if res["n_boxes"] == 0 and res["n_frames"]:
                banner("<b>Mô hình B không tìm thấy xe nào trong dải quét.</b> Kiểm tra vạch dừng đã đặt đúng chưa "
                       "(B chỉ quét quanh vạch), hoặc giảm ngưỡng B ở thanh bên.", good="warn")

            col_main, col_list = st.columns([1.75, 1], gap="large")
            with col_main:
                with st.container(border=True):
                    if res["kind"] == "video":
                        st.markdown("**Video kết quả** · khung mảnh = B tìm xe, khung đậm = C ghép (có ID), "
                                    "khung đỏ = vượt đèn đỏ, vệt vàng = quỹ đạo")
                        if res["video_browser_ok"]:
                            st.video(res["video_bytes"], format="video/mp4")
                        else:
                            st.caption("Máy chủ thiếu bộ mã hoá H.264 nên không phát trực tiếp được. Hãy tải video về để xem.")
                        st.download_button("⬇  Tải video kết quả (.mp4)", res["video_bytes"],
                                           file_name="ket_qua_phan_tich.mp4", mime="video/mp4", width="stretch")
                    else:
                        st.markdown("**Ảnh kết quả** · khung mảnh = B tìm xe, khung đậm = C ghép, khung đỏ = vượt đèn đỏ")
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
                            pdo = "" if v["XacSuatDo"] is None else f" · P = {v['XacSuatDo']:.2f}"
                            cm.markdown(f'<div class="vmeta"><span>Đèn (A)</span><br>{pill("RED")}{pdo}<br>'
                                        f'<span>Cùng một xe (C)</span><br>P = {v["XacSuatCungXe"]:.2f}<br>'
                                        f'<span>Là xe (B)</span><br>P = {v["DiemXe"]:.2f}</div>', unsafe_allow_html=True)
                            st.download_button("⬇  Ảnh bằng chứng", v["snapshot"], file_name=f"vi_pham_{v['MaXe']}.jpg",
                                               mime="image/jpeg", key=f"dl_{k}_{v['MaXe']}", width="stretch")

            if res["timeline"] and len(res["timeline"]) > 1:
                with st.container(border=True):
                    st.markdown("**Trạng thái đèn theo thời gian** · Mô hình A (sau bỏ phiếu), vạch chấm = thời điểm cảnh báo")
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

            st.caption("Kết quả chưa đúng? Chỉnh lại vạch dừng, vùng đèn, khoảng cách giữa hai frame hoặc ngưỡng ở thanh bên "
                       "rồi bấm Phân tích lại.")

# ==========================================
# TAB 2: HIỆU NĂNG MÔ HÌNH (số liệu thật từ notebook)
# ==========================================
with tab_bench:
    st.write("")
    if not KQ:
        st.info("Không tìm thấy **mo_hinh/ket_qua_tong_hop.json** (notebook lưu cùng file joblib). "
                "Chép file này vào thư mục mo_hinh để xem số liệu đánh giá.")
    else:
        st.caption("Mọi con số dưới đây đọc từ ket_qua_tong_hop.json do notebook ghi ra. Mô hình được chọn trên tập val "
                   "(A, B) hoặc theo AUC ngoài fold trên train (C); test chỉ chấm một lần. Lớp đa số là mốc đối chứng, "
                   "MLP chỉ để tham khảo (DUNG_MLP = False).")

        def cand_chart(rows, name_key, val_key, chosen, title, ymin=0.0):
            rows = [r for r in rows if r.get(val_key) is not None]
            names = [r[name_key] for r in rows]
            vals = [r[val_key] for r in rows]
            colors = [C_ACCENT if n == chosen else ("#475569" if "đa số" in n or n.startswith("MLP") else "#2A3A5C")
                      for n in names]
            fig = go.Figure(go.Bar(x=[n.replace(" (", "<br>(", 1) for n in names], y=vals,
                                   text=[f"{v:.3f}" for v in vals], textposition="outside", marker=dict(color=colors),
                                   hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
            fig.update_yaxes(range=[ymin, 1.05])
            fig.update_xaxes(tickfont=dict(size=11))
            return style_fig(fig, height=330, title=title)

        g1, g2, g3 = st.columns(3, gap="medium")
        with g1:
            with st.container(border=True):
                st.plotly_chart(cand_chart(_kq("A/bang_val", []), "Mô hình", "F1 val (TB)", MODELS["A"]["thuat_toan"],
                                           "A · macro F1 trên val"), width="stretch", config={"displayModeBar": False})
        with g2:
            with st.container(border=True):
                st.plotly_chart(cand_chart(_kq("B/bang_val", []), "Mô hình", "F1 val (TB)", MODELS["B"]["thuat_toan"],
                                           "B · F1 xe/nền trên val"), width="stretch", config={"displayModeBar": False})
        with g3:
            with st.container(border=True):
                st.plotly_chart(cand_chart(_kq("C_mo_hinh_ghep_xe/bang_ngoai_fold", []), "Mô hình C",
                                           "AUC cặp thật (ngoài fold)", _ten_C(), "C · AUC ngoài fold (train)", ymin=0.5),
                                width="stretch", config={"displayModeBar": False})
        st.caption("Cột sáng màu = mô hình được chọn và đang chạy trên web.")

        with st.container(border=True):
            st.markdown("**Kết quả test của các mô hình đang chạy**")
            bt_ = _b_tim_test() or {}
            st.dataframe(pd.DataFrame([
                {"Khối": "A · màu đèn", "Mô hình": f"{MODELS['A']['thuat_toan']} ({MODELS['A']['cau_hinh']})",
                 "Kết quả test": f"macro F1 {fmt(_kq('A/test_macro_f1'))} · accuracy {fmt(_kq('A/test_accuracy'))} · "
                                 f"recall đèn đỏ {fmt(_kq('A/recall_den_do'))}", "Số mẫu test": _kq("A/so_mau/test")},
                {"Khối": "B · xe / nền (vùng cắt sẵn)", "Mô hình": f"{MODELS['B']['thuat_toan']} ({MODELS['B']['cau_hinh']})",
                 "Kết quả test": f"F1 {fmt(_kq('B/test_f1'))} · precision {fmt(_kq('B/test_precision'))} · "
                                 f"recall {fmt(_kq('B/test_recall'))}", "Số mẫu test": _kq("B/so_mau/test")},
                {"Khối": "B · tìm xe trên frame", "Mô hình": f"{HT.phien_ban_B}, ngưỡng {HT.nguong_xe:g}",
                 "Kết quả test": f"F1 {fmt(bt_.get('F1'))} · recall {fmt(bt_.get('Recall'))} · precision {fmt(bt_.get('Precision'))} "
                                 f"(IoU ≥ 0,3, ghép một-một)", "Số mẫu test": _kq("B_tim_xe/so_frame_test")},
                {"Khối": "C · theo dõi phương tiện", "Mô hình": f"{_ten_C()}, ngưỡng {HT.nguong_C:g}",
                 "Kết quả test": f"AUC cặp thật ngoài fold {fmt(_auc_C())}",
                 "Số mẫu test": _kq("C_mo_hinh_ghep_xe/so_cap_ung_vien_train")},
            ]), hide_index=True, width="stretch")
            st.caption("Số mẫu: vùng đèn (A), vùng ảnh (B), frame (B tìm xe), cặp ứng viên train (C).")

        sys_rows = _kq("C_he_thong_A_B_C/ket_qua", []) or []
        if sys_rows:
            with st.container(border=True):
                st.markdown("**Cả hệ thống A + B + C** · phát hiện xe vượt vạch và cảnh báo vượt đèn đỏ theo nhãn dataset")
                df_sys = pd.DataFrame(sys_rows)
                cols_ = [c for c in ["Hệ thống", "Tập", "Số cặp frame", "Lần đi vào giao lộ (nhãn)", "Vượt vạch: đúng",
                                     "Vượt vạch: nhầm", "Vượt vạch: sót", "F1 vượt vạch", "Vượt đèn đỏ (nhãn)",
                                     "Cảnh báo đúng", "Cảnh báo nhầm", "Bỏ sót vượt đỏ"] if c in df_sys]
                st.dataframe(df_sys[cols_], hide_index=True, width="stretch",
                             column_config={"F1 vượt vạch": st.column_config.NumberColumn(format="%.3f")})
                st.caption("Val và test chỉ có 1 lần vượt đèn đỏ có nhãn nên số liệu cảnh báo rất ít mẫu; "
                           "F1 vượt vạch là chỉ số chính để so sánh có và không có mô hình C.")

        speed = _kq("toc_do/bang", []) or []
        if speed:
            with st.container(border=True):
                st.markdown("**Tốc độ đo trong notebook** (CPU 2 nhân, mỗi frame)")
                df_sp = pd.DataFrame(speed).rename(columns={"index": "Bước"})
                fig_sp = px.bar(df_sp[df_sp["Bước"] != "Toàn pipeline / frame"], x="TB (ms)", y="Bước", orientation="h",
                                color_discrete_sequence=["#A78BFA"], text="TB (ms)")
                fig_sp.update_traces(texttemplate="%{x:.0f} ms", textposition="outside")
                fig_sp.update_yaxes(autorange="reversed", title="")
                st.plotly_chart(style_fig(fig_sp, height=280), width="stretch", config={"displayModeBar": False})
                tot = df_sp[df_sp["Bước"] == "Toàn pipeline / frame"]
                if len(tot):
                    st.caption(f"Toàn pipeline ≈ {tot['TB (ms)'].iloc[0] / 1000:.1f} giây / frame "
                               f"({_kq('toc_do/fps', 0):.2f} FPS). Phần lớn thời gian là SVM-RBF của B chấm hàng nghìn cửa sổ.")

# ==========================================
# TAB 3: ĐẶC TRƯNG CỦA MÔ HÌNH A VÀ B
# ==========================================
with tab_feat:
    res = st.session_state.analysis
    light_sample = res.get("light_sample") if res else None
    vehicle_sample = res["vehicle_sample"][0] if res and res.get("vehicle_sample") else None
    st.write("")
    if res is None:
        st.caption("Đang hiển thị dữ liệu minh hoạ. Hãy phân tích một video ở tab đầu để xem đặc trưng lấy từ dữ liệu thật.")
    else:
        st.caption("Đặc trưng được trích từ dữ liệu bạn vừa phân tích, đúng như mô hình nhìn thấy.")

    col_hsv, col_hog = st.columns(2, gap="large")
    with col_hsv:
        with st.container(border=True):
            st.markdown("**Mô hình A · 44 đặc trưng màu của vùng đèn**")
            if light_sample is None:
                # vùng đèn đỏ thật lấy từ dataset (ảnh vẽ tay màu thuần không giống dữ liệu huấn luyện)
                light_sample = cv2.imdecode(np.frombuffer(base64.b64decode(DEN_MAU_B64), np.uint8), cv2.IMREAD_COLOR)
            feat = dac_trung_den_anh(light_sample, (0.0, 0.0, 1.0, 1.0))
            lop, xs = HT.doan_den(light_sample, (0.0, 0.0, 1.0, 1.0))
            st_ = {"red_light": "RED", "yellow_light": "YELLOW", "green_light": "GREEN"}.get(lop, "UNKNOWN")
            hs1, hs2 = st.columns([1, 3])
            hs1.image(light_sample, channels="BGR", caption="Vùng đèn", width="stretch")
            hs2.markdown(f"{MODELS['A']['thuat_toan']} kết luận &nbsp; {pill(st_)}", unsafe_allow_html=True)
            hs2.caption(" · ".join(f"P({STATE_VN[{'red_light': 'RED', 'yellow_light': 'YELLOW', 'green_light': 'GREEN'}[k]].lower()}) "
                                   f"= {v:.2f}" for k, v in xs.items()))
            hs2.caption("Vùng đèn được nới 10% mỗi phía, thu về 32×32 rồi tính đặc trưng (giống bước tiền xử lý).")
            fig_hsv = px.bar(x=list(range(44)), y=feat, color=TEN_NHOM_DEN,
                             labels={"x": "Chỉ số đặc trưng", "y": "Giá trị (trước chuẩn hóa)", "color": ""},
                             color_discrete_map={"H": C_RED, "S": C_AMBER, "V": C_ACCENT, "Tỉ lệ sáng": C_GREEN,
                                                 "Thống kê": "#A78BFA", "Hình dạng": "#F472B6"})
            fig_hsv.update_layout(legend=dict(orientation="h", y=1.15, x=0))
            st.plotly_chart(style_fig(fig_hsv, height=320), width="stretch", config={"displayModeBar": False})
            st.caption("18 bin Hue + 8 bin Saturation + 8 bin Value + tỉ lệ điểm sáng đỏ/vàng/xanh + "
                       "trung bình và độ lệch H, S, V + tỉ lệ cao/rộng của khung.")

    with col_hog:
        with st.container(border=True):
            st.markdown("**Mô hình B · HOG → PCA 150 của vùng ảnh**")
            if vehicle_sample is None or vehicle_sample.size == 0:
                vehicle_sample = np.full((90, 60, 3), 70, dtype=np.uint8)
                cv2.rectangle(vehicle_sample, (12, 20), (48, 80), (200, 200, 210), -1)
                cv2.rectangle(vehicle_sample, (16, 26), (44, 44), (60, 60, 70), -1)
            veh = cv2.resize(vehicle_sample, (64, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(veh, cv2.COLOR_BGR2GRAY)
            hog_vec = hog_lo(gray[None])[0]
            _, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             block_norm="L2-Hys", visualize=True)
            hog_img = np.sqrt(hog_img / max(hog_img.max(), 1e-6))
            hog_img = cv2.applyColorMap(cv2.resize((hog_img * 255).astype(np.uint8), (256, 256),
                                                   interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)
            z = HT.hog_sang_pca(hog_vec[None])
            p_xe = float(HT.diem_xe(z)[0])
            g1, g2 = st.columns(2)
            g1.image(cv2.resize(veh, (256, 256), interpolation=cv2.INTER_NEAREST), channels="BGR",
                     caption="Vùng ảnh chuẩn hoá 64×64", width="stretch")
            g2.image(hog_img, channels="BGR", caption="Hướng gradient (HOG)", width="stretch")
            st.markdown(f"{MODELS['B']['thuat_toan']}: P(xe) = **{p_xe:.3f}** "
                        f"({'≥' if p_xe >= nguong_xe else '<'} ngưỡng {nguong_xe:g})")
            fig_hog = px.bar(x=list(range(z.shape[1])), y=z[0], labels={"x": "Thành phần PCA", "y": "Giá trị"},
                             color_discrete_sequence=["#A78BFA"])
            st.plotly_chart(style_fig(fig_hog, height=220), width="stretch", config={"displayModeBar": False})
            st.caption(f"HOG {hog_vec.size:,} chiều (9 hướng, ô 8×8, khối 2×2, L2-Hys) → chuẩn hóa → PCA "
                       f"{z.shape[1]} thành phần → SVM-RBF. 150 thành phần giữ khoảng 69,5% phương sai.")

# ==========================================
# TAB 4: NHẬT KÝ VI PHẠM
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

st.markdown('<div class="footer">Đồ án môn Học Máy Truyền Thống © 2026 · Streamlit, OpenCV & scikit-learn</div>',
            unsafe_allow_html=True)
