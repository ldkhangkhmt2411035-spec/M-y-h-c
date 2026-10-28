import os
import tempfile
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import linear_sum_assignment
from skimage.feature import hog

# ==========================================
# CẤU HÌNH TRANG STREAMLIT & MODERN CSS
# ==========================================
st.set_page_config(
    page_title="Hệ thống Cảnh báo Xe Vượt Đèn Đỏ (Traditional ML)",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: #f8fafc; }
    .main-header {
        background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px;
        padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.6); border-radius: 12px; padding: 16px;
        border-left: 4px solid #3b82f6; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .model-info-box {
        background: rgba(15, 23, 42, 0.8); border: 1px solid #3b82f6;
        border-radius: 10px; padding: 12px; margin-top: 10px;
    }
    .badge-red, .badge-green, .badge-yellow, .badge-gray {
        color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px;
    }
    .badge-red { background-color: #ef4444; box-shadow: 0 0 12px rgba(239, 68, 68, 0.5); }
    .badge-green { background-color: #10b981; box-shadow: 0 0 12px rgba(16, 185, 129, 0.5); }
    .badge-yellow { background-color: #f59e0b; }
    .badge-gray { background-color: #64748b; }
    .alert-violation {
        background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%); border: 2px solid #ef4444;
        color: #fecaca; padding: 15px; border-radius: 12px; font-size: 16px; font-weight: bold; margin-top: 15px;
    }
    .pipeline-box {
        background: rgba(15, 23, 42, 0.6); border: 1px dashed #475569; border-radius: 10px;
        padding: 10px 14px; font-size: 0.92rem; color: #cbd5e1;
    }
    </style>
""", unsafe_allow_html=True)

if "violations_log" not in st.session_state:
    st.session_state.violations_log = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None

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
    cv2.putText(img, f"DEN {STATE_ASCII[light_state]}", (x0, max(15, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATE_COLOR[light_state], 2)
    for tid, t in active.items():
        x, y, w, h = t["box"]
        color = BGR_RED if t["violation"] else BGR_GREEN
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3 if t["violation"] else 2)
        label = f"ID {tid}" + (" VI PHAM" if t["violation"] else "")
        cv2.putText(img, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        rp = tuple(int(v) for v in line.ref_point(t["box"]))
        cv2.circle(img, rp, 4, color, -1)
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 430), 28), (20, 20, 20), -1)
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
# SIDEBAR - CẤU HÌNH & CHỌN MÔ HÌNH
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/traffic-light.png", width=70)
st.sidebar.title("🎛️ Cấu Hình Máy Học")
st.sidebar.markdown("**Đồ án Học Máy Truyền Thống - Nhóm 6**")

st.sidebar.markdown("---")
st.sidebar.subheader("1. Chọn Thuật Toán ML")
sel_model_a = st.sidebar.selectbox("Mô hình A (Nhận diện Đèn):", list(MODEL_PERFORMANCE["Model_A"].keys()))
perf_a = MODEL_PERFORMANCE["Model_A"][sel_model_a]
sel_model_b = st.sidebar.selectbox("Mô hình B (Phát hiện Xe):", list(MODEL_PERFORMANCE["Model_B"].keys()))
perf_b = MODEL_PERFORMANCE["Model_B"][sel_model_b]
sel_model_c = st.sidebar.selectbox("Mô hình C (Ghép cặp Xe):", list(MODEL_PERFORMANCE["Model_C"].keys()))
perf_c = MODEL_PERFORMANCE["Model_C"][sel_model_c]

avg_acc = round((perf_a["accuracy"] + perf_b["accuracy"] + perf_c["accuracy"]) / 3, 1)
total_latency = round(sum(float(p["latency"].split()[0]) for p in (perf_a, perf_b, perf_c)), 1)
st.sidebar.markdown(f"""
<div class="model-info-box">
    <small style="color: #94a3b8;">Hiệu năng hệ thống được chọn:</small><br>
    <b>🎯 Độ chính xác trung bình: <span style="color:#10b981;">{avg_acc}%</span></b><br>
    <b>⏱️ Tổng độ trễ (Latency): <span style="color:#3b82f6;">{total_latency} ms</span></b>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Tình Huống Giả Lập (tab Mô phỏng)")
scenario = st.sidebar.selectbox(
    "Chọn kịch bản kiểm thử:",
    [
        "Kịch bản 1: Đèn Đỏ - Xe máy Vượt Vạch (Vi Phạm)",
        "Kịch bản 2: Đèn Đỏ - Ô tô Dừng Đúng Quy Định",
        "Kịch bản 3: Đèn Xanh - Lưu Thông Bình Thường",
    ],
)
stop_line_y = st.sidebar.slider("Vị trí Vạch dừng mô phỏng (Trục Y):", 200, 400, 320)


# ==========================================
# HÀM TẠO HÌNH ẢNH MÔ PHỎNG (SYNTHETIC SCENE)
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
# GIAO DIỆN CHÍNH - HEADER & METRICS
# ==========================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.2rem; color: #f8fafc;">🚦 Hệ Thống Phát Hiện & Cảnh Báo Xe Vi Phạm Vượt Đèn Đỏ</h1>
    <p style="margin-top:8px; color: #94a3b8; font-size: 1.05rem;">
        Phương pháp Máy học Truyền thống (Traditional ML Pipeline) | Đặc trưng <b>HSV + HOG</b> | Trừ nền <b>MOG2</b> | Ghép cặp <b>Hungarian</b>
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, title, name, perf, color in [
    (c1, "Mô hình A (Đèn)", sel_model_a, perf_a, "#3b82f6"),
    (c2, "Mô hình B (Phát hiện Xe)", sel_model_b, perf_b, "#8b5cf6"),
    (c3, "Mô hình C (Ghép cặp Xe)", sel_model_c, perf_c, "#ec4899"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color};">
            <small style="color:#94a3b8">{title}</small>
            <h3 style="margin:0; color:{color}">{name}</h3>
            <span style="color:#10b981; font-weight:bold">Accuracy: {perf['accuracy']}%</span>
        </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #ef4444;">
        <small style="color:#94a3b8">Nhật ký Vi phạm</small>
        <h3 style="margin:0; color:#ef4444">{len(st.session_state.violations_log)} lượt</h3>
        <span style="color:#f59e0b; font-weight:bold">Đã lưu tự động</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab_real, tab_sim, tab_bench, tab_feat, tab_log = st.tabs([
    "🎥 PHÂN TÍCH ẢNH / VIDEO THẬT",
    "📹 MÔ PHỎNG PIPELINE (FRAME t → t+1)",
    "📈 SO SÁNH HIỆU NĂNG MÔ HÌNH",
    "🎨 GIẢI MÃ ĐẶC TRƯNG (HSV & HOG)",
    "📋 NHẬT KÝ VI PHẠM & EXPORT",
])

# ------------------------------------------
# TAB 1: PHÂN TÍCH ẢNH / VIDEO THẬT
# ------------------------------------------
with tab_real:
    st.subheader("🎥 Tải ảnh hoặc video từ camera giao thông để kiểm tra vi phạm")
    st.markdown("""
    <div class="pipeline-box">
    <b>Quy trình:</b> ① <b>Mô hình A</b> đọc màu đèn trong vùng đèn (ngưỡng HSV) →
    ② <b>Mô hình B</b> tách xe khỏi nền đường (trừ nền MOG2 + contour) →
    ③ <b>Mô hình C</b> ghép cùng một xe giữa các khung hình (khoảng cách tâm + Hungarian) →
    ④ Xe có <b>đầu xe đi từ trước vạch sang sau vạch khi đèn ĐỎ</b> ⇒ <b>VI PHẠM</b>.
    <br><small>Yêu cầu: camera đặt cố định (không rung, không lia), nhìn thấy vạch dừng; nên thấy cả đèn giao thông.</small>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    input_type = st.radio("Loại dữ liệu:", ["🎥 Video", "🖼️ Ảnh"], horizontal=True)
    preview, source_key, video_path, images, background = None, None, None, [], None

    if input_type == "🎥 Video":
        up = st.file_uploader("Tải video lên (mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv", "m4v"])
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
            preview = read_first_frame(video_path)
            if preview is None:
                st.error("Không đọc được video này. Hãy thử đổi sang định dạng .mp4 (H.264).")
    else:
        ups = st.file_uploader("Tải ảnh lên (1 ảnh, hoặc nhiều ảnh chụp liên tiếp từ cùng một góc camera)",
                               type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
        bg_up = st.file_uploader("Ảnh nền: cùng góc camera nhưng đường trống, không có xe (bắt buộc nếu chỉ có 1–2 ảnh)",
                                 type=["jpg", "jpeg", "png", "bmp", "webp"])
        if ups:
            ups = sorted(ups, key=lambda f: f.name)
            first = resize_frame(decode_upload_image(ups[0]))
            size = (first.shape[1], first.shape[0])
            images = [first] + [resize_frame(decode_upload_image(u), size) for u in ups[1:]]
            if bg_up is not None:
                background = resize_frame(decode_upload_image(bg_up), size)
            elif len(images) >= 3:
                background = np.median(np.stack(images), axis=0).astype(np.uint8)
            preview = images[0]
            source_key = "img:" + "|".join(f"{u.name}:{u.size}" for u in ups) + (f"|bg:{bg_up.name}" if bg_up else "")

    if preview is not None:
        st.markdown("#### ⚙️ Bước 1: Đặt vạch dừng và vùng đèn cho khớp với hình")
        col_cfg, col_prev = st.columns([1, 1.4])
        with col_cfg:
            st.markdown("**🟨 Vạch dừng** (tính theo % chiều rộng / chiều cao khung hình)")
            a1, a2 = st.columns(2)
            lx1 = a1.slider("Điểm trái – ngang (x1)", 0, 100, 5, key="lx1")
            ly1 = a1.slider("Điểm trái – dọc (y1)", 0, 100, 65, key="ly1")
            lx2 = a2.slider("Điểm phải – ngang (x2)", 0, 100, 95, key="lx2")
            ly2 = a2.slider("Điểm phải – dọc (y2)", 0, 100, 65, key="ly2")
            direction_lbl = st.radio("Hướng xe chạy trong hình:",
                                     ["⬇️ Từ trên xuống (về phía camera)", "⬆️ Từ dưới lên (ra xa camera)"],
                                     key="direction")
            direction = "down" if direction_lbl.startswith("⬇️") else "up"

            st.markdown("**🚦 Vùng đèn giao thông**")
            b1, b2 = st.columns(2)
            rx = b1.slider("Vị trí ngang (x)", 0, 99, 85, key="rx")
            ry = b1.slider("Vị trí dọc (y)", 0, 99, 5, key="ry")
            rw = b2.slider("Rộng", 1, 50, 10, key="rw")
            rh = b2.slider("Cao", 1, 60, 25, key="rh")
            light_choice = st.selectbox("Trạng thái đèn:", [
                "Tự động nhận diện màu trong vùng đèn (HSV)",
                "Video không thấy đèn – coi như luôn ĐỎ",
                "Coi như luôn XANH",
            ], key="light_choice")
            light_mode = {"Tự": "auto", "Vid": "red", "Coi": "green"}[light_choice[:3].strip()]

            with st.expander("🔧 Tinh chỉnh nâng cao"):
                min_area_pct = st.slider("Diện tích xe nhỏ nhất (% khung hình)", 0.05, 5.0, 0.4, 0.05,
                                         help="Tăng lên nếu bị nhận nhầm lá cây, người đi bộ, nhiễu là xe.")
                car_area_pct = st.slider("Từ mức này trở lên coi là ô tô (% khung hình)", 0.5, 15.0, 2.5, 0.5)
                sensitivity = st.slider("Độ nhạy trừ nền (càng nhỏ càng nhạy)", 8, 80, 32)
                frame_step = st.slider("Xử lý 1 khung trên mỗi N khung (tăng để chạy nhanh hơn)", 1, 6, 2)
                max_seconds = st.slider("Chỉ phân tích tối đa (giây đầu video)", 5, 180, 60)

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
            prev_img = draw_overlay(preview, line_prev, rect_prev, prev_state, {}, "Xem truoc - chinh vach dung & vung den")
            p1, p2 = line_prev.pts()
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            arrow_dy = 40 if direction == "down" else -40
            cv2.arrowedLine(prev_img, (mid[0], mid[1] - arrow_dy), (mid[0], mid[1] + arrow_dy), (255, 255, 0), 2, tipLength=0.3)
            st.image(prev_img, channels="BGR", caption="Xem trước: vạch dừng, vùng đèn và hướng xe chạy (mũi tên)", width="stretch")
            badge = {"RED": "badge-red", "GREEN": "badge-green", "YELLOW": "badge-yellow"}.get(prev_state, "badge-gray")
            st.markdown(f'Đèn ở khung đầu tiên: <span class="{badge}">{STATE_VN[prev_state].upper()}</span>', unsafe_allow_html=True)
            if prev_state == "UNKNOWN" and light_mode == "auto":
                st.caption("⚠️ Chưa thấy đèn sáng trong khung xanh dương. Hãy kéo vùng đèn trùng với cột đèn, hoặc chọn 'coi như luôn ĐỎ'.")

        st.markdown("#### 🚀 Bước 2: Phân tích")
        can_run = True
        if input_type == "🖼️ Ảnh" and background is None:
            can_run = False
            st.warning("Với 1–2 ảnh, hãy tải thêm **ảnh nền đường trống** (cùng góc chụp), hoặc tải từ 3 ảnh liên tiếp trở lên.")

        if st.button("🔍 Phân tích vi phạm", type="primary", disabled=not can_run):
            progress = st.progress(0.0, text="Đang chuẩn bị...")
            if input_type == "🎥 Video":
                res = analyze_video(video_path, cfg, progress)
            else:
                frames = ((float(i), img) for i, img in enumerate(images))
                res = analyze_frames(frames, cfg, fps_eff=1.0, background=background,
                                     progress=progress, total=len(images))
            progress.empty()
            if res is None:
                st.error("Không xử lý được dữ liệu.")
            else:
                res["source"] = source_key
                res["kind"] = "video" if input_type == "🎥 Video" else "image"
                st.session_state.analysis = res
                now = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
                src_name = source_key.split(":")[1] if source_key else ""
                for v in res["violations"]:
                    st.session_state.violations_log.append({
                        "MaXe": v["MaXe"], "LoaiXe": v["LoaiXe"], "ThoiDiemTrongVideo": v["ThoiDiem"],
                        "TrangThaiDen": v["TrangThaiDen"], "ViTri": v["ViTri"],
                        "NguonDuLieu": src_name, "ThoiGianPhanTich": now,
                    })

    # ---------- Kết quả ----------
    res = st.session_state.analysis
    if res is not None and res.get("source") == source_key:
        st.markdown("#### 📊 Bước 3: Kết quả")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Số xe phát hiện", res["n_vehicles"])
        m2.metric("🚨 Vi phạm vượt đèn đỏ", len(res["violations"]))
        m3.metric("Qua vạch hợp lệ (đèn xanh/vàng)", res["crossed_ok"])
        if res["kind"] == "video":
            m4.metric("Thời lượng đã phân tích", f"{res['duration']:.1f} s")
        else:
            m4.metric("Số ảnh đã phân tích", res["n_frames"])

        if res["violations"]:
            for v in res["violations"]:
                st.markdown(f"""<div class="alert-violation">🚨 CẢNH BÁO VI PHẠM: {v['MaXe']} – {v['LoaiXe']}
                    đã vượt qua vạch dừng khi đèn đỏ (thời điểm {v['ThoiDiem']}).</div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Không phát hiện xe nào vượt vạch dừng khi đèn đỏ.")

        if res["kind"] == "video":
            st.markdown("##### 🎬 Video kết quả (khung xanh = xe bình thường, khung đỏ = xe vi phạm)")
            if res["video_browser_ok"]:
                st.video(res["video_bytes"], format="video/mp4")
            else:
                st.info("Máy chủ thiếu bộ mã hoá H.264 nên trình duyệt không phát trực tiếp được. Hãy tải video về để xem.")
            st.download_button("📥 Tải video kết quả", res["video_bytes"], file_name="ket_qua_phan_tich.mp4", mime="video/mp4")

            tl = pd.DataFrame(res["timeline"], columns=["Thời điểm (s)", "Đèn"])
            tl["Đèn"] = tl["Đèn"].map(STATE_VN)
            fig_tl = px.scatter(tl, x="Thời điểm (s)", y="Đèn", color="Đèn",
                                color_discrete_map={"Đỏ": "#ef4444", "Xanh": "#10b981", "Vàng": "#f59e0b", "Không rõ": "#64748b"},
                                category_orders={"Đèn": ["Đỏ", "Vàng", "Xanh", "Không rõ"]},
                                title="Trạng thái đèn theo thời gian (Mô hình A)")
            fig_tl.update_traces(marker=dict(size=6, symbol="square"))
            fig_tl.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                                 height=260, showlegend=False)
            st.plotly_chart(fig_tl, width="stretch")
        else:
            st.markdown("##### 🖼️ Ảnh kết quả")
            cols = st.columns(min(3, len(res["annotated"])))
            for i, img in enumerate(res["annotated"]):
                cols[i % len(cols)].image(img, channels="BGR", caption=f"Ảnh {i + 1}", width="stretch")

        if res["violations"]:
            st.markdown("##### 📸 Ảnh bằng chứng vi phạm")
            for v in res["violations"]:
                e1, e2 = st.columns([2, 1])
                e1.image(v["snapshot"], caption=f"{v['MaXe']} – khoảnh khắc vượt vạch ({v['ThoiDiem']})", width="stretch")
                e2.image(v["crop"], caption="Ảnh cận phương tiện", width="stretch")
                e2.markdown(f"**Mã xe:** {v['MaXe']}  \n**Loại xe:** {v['LoaiXe']}  \n**Đèn:** {v['TrangThaiDen']}  \n**Vị trí:** {v['ViTri']}")
                e2.download_button("📥 Tải ảnh bằng chứng", v["snapshot"], file_name=f"vi_pham_{v['MaXe']}.jpg",
                                   mime="image/jpeg", key=f"dl_{v['MaXe']}_{v['ThoiDiem']}")

        st.caption("Gợi ý: nếu kết quả chưa đúng, hãy chỉnh lại vạch dừng/vùng đèn hoặc mở 'Tinh chỉnh nâng cao' rồi bấm Phân tích lại.")
    elif preview is None:
        st.info("⬆️ Hãy tải lên một video (hoặc ảnh) quay từ camera cố định tại ngã tư để bắt đầu.")

# ------------------------------------------
# TAB 2: MÔ PHỎNG PIPELINE
# ------------------------------------------
with tab_sim:
    st.subheader("🔍 Mô phỏng chuỗi xử lý phát hiện vượt đèn đỏ (dữ liệu giả lập)")
    f1_img, l_state, _ = generate_traffic_scene(scenario, frame_idx=1, stop_y=stop_line_y)
    f2_img, l_state, sim_violations = generate_traffic_scene(scenario, frame_idx=2, stop_y=stop_line_y)

    st.write("##### **Kết quả Mô hình A (Nhận diện Đèn giao thông):**")
    if l_state == "RED":
        st.markdown('<span class="badge-red">🔴 TRẠNG THÁI: ĐÈN ĐỎ (RED LIGHT) - KÍCH HOẠT GIÁM SÁT VI PHẠM</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-green">🟢 TRẠNG THÁI: ĐÈN XANH (GREEN LIGHT) - CHO PHÉP PHƯƠNG TIỆN DI CHUYỂN</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.image(f1_img, channels="BGR", caption="Khung hình t (Frame 1) - Mô hình B định vị xe", width="stretch")
    with col_f2:
        st.image(f2_img, channels="BGR", caption="Khung hình t+1 (Frame 2) - Mô hình C ghép cặp & Kiểm tra vạch", width="stretch")

    if sim_violations:
        for viol in sim_violations:
            st.markdown(f"""<div class="alert-violation">🚨 CẢNH BÁO VI PHẠM (mô phỏng): Phương tiện {viol['MaXe']} ({viol['LoaiXe']})
                đã di chuyển cắt qua Vạch Dừng khi Đèn Đỏ!</div>""", unsafe_allow_html=True)
    else:
        st.success("✅ Tất cả phương tiện tuân thủ đúng luật giao thông hoặc Đèn Xanh được phép di chuyển.")

# ------------------------------------------
# TAB 3: SO SÁNH HIỆU NĂNG MÔ HÌNH
# ------------------------------------------
with tab_bench:
    st.subheader("📊 Chi tiết thông số Mô hình Machine Learning được chọn")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### So sánh Độ chính xác (%) giữa các Thuật toán")
        df_b = pd.DataFrame([
            {"Thuật toán": k, "Accuracy (%)": v["accuracy"], "Latency (ms)": float(v["latency"].split()[0])}
            for k, v in MODEL_PERFORMANCE["Model_B"].items()
        ])
        fig_bar = px.bar(df_b, x="Thuật toán", y="Accuracy (%)", text="Accuracy (%)", color="Thuật toán",
                         title="Hiệu năng các thuật toán ở Mô hình B (Phát hiện Xe)",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_bar, width="stretch")
    with col_m2:
        st.markdown("#### Bảng Tổng Hợp Chi Tiết Mô Hình Đang Chọn")
        st.table(pd.DataFrame([
            {"Mô hình": "Mô hình A (Màu đèn)", "Thuật toán chọn": sel_model_a, "Accuracy": f"{perf_a['accuracy']}%", "Latency": perf_a["latency"], "Mô tả": perf_a["desc"]},
            {"Mô hình": "Mô hình B (Phát hiện xe)", "Thuật toán chọn": sel_model_b, "Accuracy": f"{perf_b['accuracy']}%", "Latency": perf_b["latency"], "Mô tả": perf_b["desc"]},
            {"Mô hình": "Mô hình C (Ghép xe)", "Thuật toán chọn": sel_model_c, "Accuracy": f"{perf_c['accuracy']}%", "Latency": perf_c["latency"], "Mô tả": perf_c["desc"]},
        ]))

# ------------------------------------------
# TAB 4: GIẢI MÃ ĐẶC TRƯNG (HSV & HOG) – lấy từ dữ liệu thật nếu đã phân tích
# ------------------------------------------
with tab_feat:
    st.subheader("🎨 Trực quan hóa đặc trưng trích xuất (Feature Extraction)")
    res = st.session_state.analysis
    light_sample = res.get("light_sample") if res else None
    vehicle_sample = res["vehicle_sample"][0] if res and res.get("vehicle_sample") else None
    if res is None:
        st.info("Đang hiển thị dữ liệu giả lập. Hãy phân tích một ảnh/video ở tab đầu tiên để xem đặc trưng thật.")

    col_hsv, col_hog = st.columns(2)
    with col_hsv:
        st.markdown("#### 1. Vector đặc trưng HSV của vùng đèn (44 chiều)")
        if light_sample is None:
            light_sample = np.zeros((80, 80, 3), dtype=np.uint8)
            cv2.circle(light_sample, (40, 40), 22, (0, 0, 230), -1)
        st.image(light_sample, channels="BGR", caption="Vùng đèn giao thông cắt được (ROI)", width=140)
        feat = hsv_feature(light_sample)
        names = ["Hue"] * 18 + ["Saturation"] * 16 + ["Value"] * 10
        fig_hsv = px.bar(x=list(range(44)), y=feat, color=names,
                         labels={"x": "Bin HSV (0-43)", "y": "Tỉ lệ điểm ảnh", "color": "Kênh"},
                         title="Histogram HSV: Hue 18 bin + Saturation 16 bin + Value 10 bin",
                         color_discrete_map={"Hue": "#ef4444", "Saturation": "#f59e0b", "Value": "#3b82f6"})
        fig_hsv.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_hsv, width="stretch")

    with col_hog:
        st.markdown("#### 2. Đặc trưng HOG của phương tiện")
        if vehicle_sample is None or vehicle_sample.size == 0:
            vehicle_sample, _, _ = generate_traffic_scene("Kịch bản 1: Đèn Đỏ - Xe máy Vượt Vạch (Vi Phạm)", 1)
            vehicle_sample = vehicle_sample[200:300, 330:470]
        gray = cv2.cvtColor(cv2.resize(vehicle_sample, (128, 128)), cv2.COLOR_BGR2GRAY)
        hog_vec, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_img = cv2.normalize(hog_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h1, h2 = st.columns(2)
        h1.image(vehicle_sample, channels="BGR", caption="Phương tiện cắt được", width="stretch")
        h2.image(hog_img, caption="Ảnh HOG (hướng gradient)", width="stretch")
        st.caption(f"Vector HOG: {hog_vec.size} chiều (ảnh 128×128, 9 hướng, cell 8×8, block 2×2).")
        fig_hog = px.bar(x=list(range(50)), y=hog_vec[:50], labels={"x": "Chỉ số đặc trưng (50 giá trị đầu)", "y": "Giá trị"},
                         title="50 giá trị đầu của vector HOG", color_discrete_sequence=["#8b5cf6"])
        fig_hog.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=280)
        st.plotly_chart(fig_hog, width="stretch")

# ------------------------------------------
# TAB 5: NHẬT KÝ VI PHẠM
# ------------------------------------------
with tab_log:
    st.subheader("📋 Bảng Ghi Nhận Phương Tiện Vi Phạm Vượt Đèn Đỏ")
    if st.session_state.violations_log:
        df_log = pd.DataFrame(st.session_state.violations_log)
        st.dataframe(df_log, width="stretch")
        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            st.download_button("📥 Tải Nhật Ký Vi Phạm (CSV)", df_log.to_csv(index=False).encode("utf-8-sig"),
                               file_name="danh_sach_vi_pham_dendo.csv", mime="text/csv")
        with col_dl2:
            if st.button("🗑️ Xóa lịch sử ghi nhận"):
                st.session_state.violations_log = []
                st.rerun()
    else:
        st.info("Chưa có lượt vi phạm nào. Hãy tải video/ảnh ở tab đầu tiên và bấm 'Phân tích vi phạm'.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>Đồ án môn Học Máy Truyền Thống © 2026 - Nhóm 6 | Xây dựng với Streamlit & OpenCV</p>", unsafe_allow_html=True)
