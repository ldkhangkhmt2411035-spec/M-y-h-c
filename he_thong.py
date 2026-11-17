"""Lõi suy luận của hệ thống cảnh báo vượt đèn đỏ – tái hiện đúng notebook "Máy học".

Dùng file mô hình mà notebook lưu ở mục E: ``mo_hinh/he_thong_canh_bao.joblib``
    mo_hinh_A          k-NN (k=9, distance, manhattan) – màu đèn từ 44 đặc trưng HSV (đã chuẩn hóa bằng chuan_hoa_A)
    mo_hinh_B_tim_xe   SVM-RBF (C=10) – xe / nền trên HOG (64×64, 1.764 chiều) → PCA 150, quét cửa sổ trượt theo dải y quanh vạch
    mo_hinh_C          k-NN (k=25) trong pipeline StandardScaler – xác suất hai khung ở hai frame liền nhau là cùng một xe (21 đặc trưng)
                       + Hungarian ghép một-một (thuật toán, không phải mô hình học)
Bước kết luận (luật, không huấn luyện): xe được C ghép có đáy khung đi qua vạch dừng theo hướng xe chạy, trong lúc A báo đèn đỏ.

Mọi ảnh đưa vào mô hình được đưa về 640×640 như dataset. Toạ độ trong module này là pixel của ảnh 640×640.
Module không phụ thuộc Streamlit để có thể kiểm thử riêng.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

S = 640                     # cạnh ảnh đưa vào mô hình (giống dataset)
PATCH = 64                  # cỡ vùng ảnh để tính HOG
CROP_DEN, PAD_DEN = 32, 0.10

# Hậu xử lý B: notebook mục B4 chọn trên VAL "ngưỡng 0.8, NMS 0.2, bỏ phiếu" (ket_qua_tong_hop.json → B_hau_xu_ly).
# Hồi quy đáy khung (Ridge) cũng được chọn nhưng KHÔNG được lưu trong file joblib nên ở đây không dùng.
NMS_IOU, VOTE_IOU, BO_PHIEU = 0.2, 0.5, True
BUOC_TRUOT = 0.5
LAM_MUOT_A = 3              # D1: màu đèn = đa số của 3 frame gần nhất

TEN_DEN_VN = {"red_light": "RED", "yellow_light": "YELLOW", "green_light": "GREEN"}


# ---------------------------------------------------------------- tiện ích hình học
def iou_ma_tran(A, B):
    A = np.asarray(A, float).reshape(-1, 4); B = np.asarray(B, float).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    xx1 = np.maximum(A[:, None, 0], B[None, :, 0]); yy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    xx2 = np.minimum(A[:, None, 2], B[None, :, 2]); yy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    dt = lambda K: (K[:, 2] - K[:, 0]) * (K[:, 3] - K[:, 1])
    return inter / (dt(A)[:, None] + dt(B)[None, :] - inter + 1e-9)


def nms(khung, diem, iou_max=0.3):
    thu_tu, giu = np.argsort(diem)[::-1], []
    while len(thu_tu):
        i = thu_tu[0]; giu.append(i)
        thu_tu = thu_tu[1:][iou_ma_tran(khung[[i]], khung[thu_tu[1:]])[0] < iou_max]
    return np.array(giu, int)


def sang_xywh(K):
    """(x1, y1, x2, y2) px → (xc, đáy, rộng, cao) chuẩn hóa 0..1."""
    K = np.asarray(K, float).reshape(-1, 4) / S
    return np.c_[(K[:, 0] + K[:, 2]) / 2, K[:, 3], K[:, 2] - K[:, 0], K[:, 3] - K[:, 1]]


def vuot_vach(ya, yb, ly, xe_di_len=True):
    """+1: đi qua vạch theo hướng xe chạy; -1: ngược lại; 0: không đổi phía (ya, yb: đáy khung frame trước/sau)."""
    if ya > ly >= yb:
        return 1 if xe_di_len else -1
    if ya <= ly < yb:
        return -1 if xe_di_len else 1
    return 0


# ---------------------------------------------------------------- HOG (giống hệt hog_lo của notebook)
def hog_lo(gray):
    """HOG cả lô ảnh xám (N, 64, 64): 9 hướng, ô 8×8, khối 2×2, L2-Hys → 1.764 chiều (khớp skimage.feature.hog)."""
    g = gray.astype(np.float32); N, H, W = g.shape
    gr = np.zeros_like(g); gc = np.zeros_like(g)
    gr[:, 1:-1, :] = g[:, 2:, :] - g[:, :-2, :]; gc[:, :, 1:-1] = g[:, :, 2:] - g[:, :, :-2]
    mag = np.hypot(gc, gr)
    bin_ = np.minimum((np.rad2deg(np.arctan2(gr, gc)) % 180) // 20, 8).astype(np.int64)
    ch, cw = H // 8, W // 8
    o_cell = (np.arange(H)[:, None] // 8) * cw + (np.arange(W)[None, :] // 8)
    idx = (np.arange(N)[:, None, None] * (ch * cw) + o_cell[None]) * 9 + bin_
    oh = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=N * ch * cw * 9).reshape(N, ch, cw, 9) / 64.0
    blk = np.concatenate([np.concatenate([oh[:, :-1, :-1, None], oh[:, :-1, 1:, None]], 3)[:, :, :, None],
                          np.concatenate([oh[:, 1:, :-1, None], oh[:, 1:, 1:, None]], 3)[:, :, :, None]], 3)
    eps = 1e-5
    v = blk / np.sqrt((blk ** 2).sum((-3, -2, -1), keepdims=True) + eps ** 2); v = np.minimum(v, 0.2)
    v = v / np.sqrt((v ** 2).sum((-3, -2, -1), keepdims=True) + eps ** 2)
    return v.reshape(N, -1)


# ---------------------------------------------------------------- đặc trưng đèn (giống dac_trung_den_anh)
TEN_NHOM_DEN = (["H"] * 18 + ["S"] * 8 + ["V"] * 8 + ["Tỉ lệ sáng"] * 3 + ["Thống kê"] * 6 + ["Hình dạng"])


def dac_trung_den_anh(img, khung):
    """44 đặc trưng màu của vùng đèn. khung = (x1, y1, x2, y2) theo tỉ lệ 0..1 của ảnh img."""
    H, W = img.shape[:2]; x1, y1, x2, y2 = khung
    bw, bh = (x2 - x1) * (1 + 2 * PAD_DEN), (y2 - y1) * (1 + 2 * PAD_DEN)
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    r0, r1 = int(max((yc - bh / 2) * H, 0)), int(min((yc + bh / 2) * H, H))
    c0, c1 = int(max((xc - bw / 2) * W, 0)), int(min((xc + bw / 2) * W, W))
    r1, c1 = max(r1, r0 + 1), max(c1, c0 + 1)
    crop = cv2.resize(img[r0:r1, c0:c1], (CROP_DEN, CROP_DEN), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel(); n = h.size
    sang = (s > 80) & (v > 120)
    return np.concatenate([np.histogram(h, 18, (0, 180))[0] / n, np.histogram(s, 8, (0, 256))[0] / n,
                           np.histogram(v, 8, (0, 256))[0] / n,
                           [(sang & ((h <= 10) | (h >= 160))).mean(), (sang & (h > 10) & (h <= 35)).mean(),
                            (sang & (h > 35) & (h <= 100)).mean()],
                           np.array([h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std()]) / 255.0,
                           [((y2 - y1) * H) / max((x2 - x1) * W, 1e-6)]])


def khung_bong_sang(img, khung, gian=0.25):
    """Tìm bóng đèn đang sáng trong khung hộp đèn (x1, y1, x2, y2 tỉ lệ 0..1); trả về khung ôm sát bóng đó.

    Không tìm thấy đốm sáng rõ thì trả lại khung cũ.
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = khung
    c0, r0 = int(max(0, x1 * W)), int(max(0, y1 * H))
    c1, r1 = int(min(W, max(x2 * W, c0 + 1))), int(min(H, max(y2 * H, r0 + 1)))
    vung = img[r0:r1, c0:c1]
    if vung.size == 0 or min(vung.shape[:2]) < 4:
        return khung
    hsv = cv2.cvtColor(vung, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    co_mau = s >= 90
    mask = None
    if co_mau.sum() >= 4:                               # ưu tiên đốm có màu (bóng đèn), không lấy nền trời trắng
        nguong_v = max(110, int(0.7 * v[co_mau].max()))
        mask = (co_mau & (v >= nguong_v)).astype(np.uint8)
    if mask is None or mask.sum() < 4:
        # bóng đèn cháy sáng thành trắng: chỉ nhận đốm trắng nằm giữa hộp đèn (tránh nền trời ở mép)
        hh, ww = v.shape
        trong = np.zeros_like(v, dtype=bool)
        trong[int(0.1 * hh):int(0.9 * hh) + 1, int(0.15 * ww):int(0.85 * ww) + 1] = True
        mask = ((v >= 235) & trong).astype(np.uint8)
        if mask.sum() < 4:
            return khung
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, nhan, tk, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return khung
    i = 1 + int(np.argmax(tk[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = tk[i, 0], tk[i, 1], tk[i, 2], tk[i, 3]
    if w * h < 4:
        return khung
    dx, dy = gian * w, gian * h
    bx0, by0 = max(0, x - dx), max(0, y - dy)
    bx1, by1 = min(vung.shape[1], x + w + dx), min(vung.shape[0], y + h + dy)
    return ((c0 + bx0) / W, (r0 + by0) / H, (c0 + bx1) / W, (r0 + by1) / H)


def mau_theo_vi_tri(img, khung):
    """Màu suy từ vị trí bóng đang sáng trong hộp đèn 3 bóng: dọc trên → đỏ, giữa → vàng, dưới → xanh;
    ngang trái → đỏ, giữa → vàng, phải → xanh. Không chắc (hộp gần vuông, không thấy bóng, bóng nằm sát
    ranh giới hai ô) thì trả về None."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = khung
    hp, wp = (y2 - y1) * H, (x2 - x1) * W
    b = khung_bong_sang(img, khung)
    if tuple(round(float(v), 6) for v in b) == tuple(round(float(v), 6) for v in khung):
        return None
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if hp >= 1.8 * wp:
        t = (cy - y1) / max(y2 - y1, 1e-9)
    elif wp >= 1.8 * hp:
        t = (cx - x1) / max(x2 - x1, 1e-9)
    else:
        return None
    if abs(t - 1 / 3) < 0.05 or abs(t - 2 / 3) < 0.05:
        return None
    lop = "red_light" if t < 1 / 3 else ("yellow_light" if t < 2 / 3 else "green_light")
    # màu của bóng phải hợp với vị trí, nếu không thì khung người dùng chọn có thể không phải cả hộp đèn → bỏ qua
    r0, r1 = int(b[1] * H), max(int(b[3] * H), int(b[1] * H) + 1)
    c0, c1 = int(b[0] * W), max(int(b[2] * W), int(b[0] * W) + 1)
    hsv = cv2.cvtColor(img[r0:r1, c0:c1], cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] >= 90) & (hsv[..., 2] >= 110)
    if m.sum() >= 3:
        hue = float(np.median(hsv[..., 0][m]))
        hop = {"red_light": hue <= 20 or hue >= 150, "yellow_light": 8 <= hue <= 40,
               "green_light": 35 <= hue <= 100}[lop]
        if not hop:
            return None
    elif max(hp, wp) < 2.2 * min(hp, wp):          # bóng cháy trắng, không có màu: chỉ tin vị trí khi chắc là hộp 3 bóng
        return None
    return lop


# ---------------------------------------------------------------- kết quả B trên một frame
@dataclass
class KetQuaFrame:
    img: np.ndarray                 # ảnh 640×640 BGR
    y_vach: float                   # vạch dừng, tỉ lệ chiều cao 0..1
    K: np.ndarray                   # mọi cửa sổ (x1, y1, x2, y2) px
    d: np.ndarray                   # xác suất "xe" của SVM-RBF cho từng cửa sổ
    _cache: dict = field(default_factory=dict)


class HeThongCanhBao:
    """Nạp file joblib của notebook và chạy A → B → C → bước kết luận."""

    def __init__(self, duong_dan_joblib):
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # cảnh báo khác phiên bản scikit-learn khi unpickle
            g = joblib.load(duong_dan_joblib)
        self.goi = g
        self.mo_hinh_A, self.chuan_hoa_A, self.lop_den = g["mo_hinh_A"], g["chuan_hoa_A"], list(g["lop_den"])
        self.mo_hinh_B, self.phien_ban_B = g["mo_hinh_B_tim_xe"], g["phien_ban_B"]
        self.nguong_xe, self.nguong_vao_C = float(g["nguong_xe"]), float(g["nguong_vao_C"])
        self.kich_thuoc = [tuple(k) for k in g["kich_thuoc_cua_so"]]
        self.roi = tuple(g["vung_quet_quanh_vach"])
        self.hog_pca = {k: np.asarray(v) for k, v in g["tham_so_hog_pca"].items()}
        self.mo_hinh_C, self.dac_trung_C, self.nguong_C = g["mo_hinh_C"], list(g["dac_trung_C"]), float(g["nguong_C"])
        self.xe_di_len_khi_hoc = bool(g.get("xe_di_len", True))
        # Hậu xử lý B: lấy từ file nếu notebook có lưu (xem README), nếu không dùng giá trị notebook đã chọn
        self.nms_iou = float(g.get("nms_iou", NMS_IOU))
        self.vote_iou = float(g.get("vote_iou", VOTE_IOU))
        self.bo_phieu = bool(g.get("bo_phieu", BO_PHIEU))
        self.tinh_chinh = g.get("tinh_chinh_B")          # Ridge hồi quy đáy khung (None nếu không lưu)
        self.ten_dac_trung_A = list(getattr(self.chuan_hoa_A, "feature_names_in_", [])) or None

    # ---------------- A
    def doan_den(self, img, khung_den, tim_bong=True):
        """khung_den: một khung (x1, y1, x2, y2) tỉ lệ 0..1 hoặc danh sách khung. Trả về (lớp, {lớp: xác suất}).

        tim_bong=True: trong khung hộp đèn người dùng chọn, tìm bóng đang sáng và chỉ đưa vùng bóng đó vào A
        (dữ liệu huấn luyện đèn xanh, đèn đỏ chủ yếu cắt sát bóng sáng; đưa cả hộp đèn tối vào thì A dễ đoán sai).
        """
        lop, xs, _ = self.doan_den_chi_tiet(img, khung_den, tim_bong=tim_bong)
        return lop, xs

    def doan_den_chi_tiet(self, img, khung_den, tim_bong=True, ket_hop_vi_tri=True, trong_so_vi_tri=0.6):
        """Như doan_den, kèm chi tiết: xác suất riêng của A và màu suy từ vị trí bóng sáng.

        ket_hop_vi_tri=True (chỉ dùng trên web, notebook không có bước này): khi hộp đèn là loại 3 bóng và thấy rõ
        bóng nào đang sáng, xác suất cuối = 0,4 × xác suất của A + 0,6 × màu theo vị trí bóng. Bóng màu cam/vàng
        nằm sát ranh giới đỏ–vàng (H ≈ 10–15) hoặc màu xanh khác dữ liệu huấn luyện sẽ được vị trí bóng sửa lại.
        """
        ds = [khung_den] if np.isscalar(khung_den[0]) else list(khung_den)
        dung = [khung_bong_sang(img, k) for k in ds] if tim_bong else ds
        X = np.array([dac_trung_den_anh(img, k) for k in dung])
        if self.ten_dac_trung_A:
            import pandas as pd
            X = pd.DataFrame(X, columns=self.ten_dac_trung_A)
        pA = self.mo_hinh_A.predict_proba(self.chuan_hoa_A.transform(X)).mean(0)
        lop = list(self.mo_hinh_A.classes_)
        vi_tri = [mau_theo_vi_tri(img, k) for k in ds] if ket_hop_vi_tri else []
        vi_tri = [v for v in vi_tri if v is not None and v in lop]
        p = pA.copy()
        if vi_tri:
            mot = np.zeros(len(lop))
            for v in vi_tri:
                mot[lop.index(v)] += 1.0 / len(vi_tri)
            p = (1 - trong_so_vi_tri) * pA + trong_so_vi_tri * mot
        ct = {"A": dict(zip(lop, (float(x) for x in pA))), "vi_tri": vi_tri[0] if vi_tri else None}
        return lop[int(np.argmax(p))], dict(zip(lop, (float(x) for x in p))), ct

    # ---------------- B
    def hog_sang_pca(self, Hm):
        p = self.hog_pca
        Z = (Hm - p["scaler_mean"]) / p["scaler_scale"]
        return ((Z - p["pca_mean"]) @ p["pca_components"].T).astype(np.float32)

    def dac_trung_vung(self, img, K):
        if len(K) == 0:
            return np.zeros((0, self.hog_pca["pca_components"].shape[0]), np.float32)
        P = np.stack([cv2.cvtColor(cv2.resize(img[int(y1):max(int(y2), int(y1) + 1), int(x1):max(int(x2), int(x1) + 1)],
                                              (PATCH, PATCH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                      for x1, y1, x2, y2 in K])
        return self.hog_sang_pca(np.vstack([hog_lo(P[i:i + 2000]) for i in range(0, len(P), 2000)]))

    def cua_so(self, ly, buoc=BUOC_TRUOT, he_so_co=1.0):
        """Cửa sổ trượt có đáy nằm trong dải của nó và trong vùng [vạch − 0,20; vạch + 0,15] (giống notebook).

        he_so_co: nhân cỡ cửa sổ khi xe trong camera mới to/nhỏ hơn camera huấn luyện (1 = đúng notebook).
        Vùng ảnh luôn được thu về 64 × 64 trước khi tính HOG nên B vẫn nhận đầu vào cùng dạng.
        """
        lo_r, hi_r = (ly + self.roi[0]) * S, (ly + self.roi[1]) * S
        K = []
        for lo, hi, w, h in self.kich_thuoc:
            w, h = min(S, max(8, int(round(w * he_so_co)))), min(S, max(8, int(round(h * he_so_co))))
            for yb in range(int(max(lo * S, lo_r, h)), int(min(hi * S, hi_r, S)) + 1, max(int(h * buoc), 4)):
                for x in range(0, S - w + 1, max(int(w * buoc), 4)):
                    K.append((x, yb - h, x + w, yb))
        return np.array(K, float).reshape(-1, 4)

    def diem_xe(self, Z):
        m = self.mo_hinh_B
        return (m.predict_proba(Z)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Z)).astype(np.float32)

    def quet_frame(self, img640, y_vach, he_so_co=1.0):
        K = self.cua_so(y_vach, he_so_co=he_so_co)
        d = self.diem_xe(self.dac_trung_vung(img640, K)) if len(K) else np.zeros(0, np.float32)
        return KetQuaFrame(img640, float(y_vach), K, d)

    def khung_xe(self, fr: KetQuaFrame, nguong):
        """Khung xe cuối của B (NMS + bỏ phiếu), điểm, đặc trưng HOG-PCA và histogram màu (dùng cho C)."""
        key = round(float(nguong), 4)
        if key in fr._cache:
            return fr._cache[key]
        c = np.where(fr.d >= nguong)[0]
        if len(c) == 0:
            out = (fr.K[:0], fr.d[:0], np.zeros((0, self.hog_pca["pca_components"].shape[0])), np.zeros((0, 64)))
        else:
            Kc, dc = fr.K[c], fr.d[c]
            if self.tinh_chinh is not None:              # ap_dung_hoi_quy_hop của notebook
                t_ = self.tinh_chinh.predict(self.dac_trung_vung(fr.img, Kc))
                w_, h_ = Kc[:, 2] - Kc[:, 0], Kc[:, 3] - Kc[:, 1]
                Kn = np.clip(Kc + t_ * np.c_[w_, h_, w_, h_], 0, S)
                ok = (Kn[:, 2] - Kn[:, 0] >= 8) & (Kn[:, 3] - Kn[:, 1] >= 8)
                Kc = np.where(ok[:, None], Kn, Kc)
            g = nms(Kc, dc, self.nms_iou)[:60]
            if self.bo_phieu:
                I = iou_ma_tran(Kc[g], Kc) >= self.vote_iou
                Kv = np.array([(Kc[m] * dc[m, None]).sum(0) / dc[m].sum() for m in I])
            else:
                Kv = Kc[g]
            dv = dc[g]
            Hm = []
            for x1, y1, x2, y2 in Kv:
                cr = cv2.resize(fr.img[int(y1):max(int(y2), int(y1) + 2), int(x1):max(int(x2), int(x1) + 2)], (32, 32))
                h = cv2.calcHist([cv2.cvtColor(cr, cv2.COLOR_BGR2HSV)], [0, 1], None, [8, 8], [0, 180, 0, 256]).ravel()
                Hm.append(h / max(h.sum(), 1))
            out = (Kv, dv, self.dac_trung_vung(fr.img, Kv), np.array(Hm))
        fr._cache[key] = out
        return out

    # ---------------- C
    def cap_ung_vien(self, fa: KetQuaFrame, fb: KetQuaFrame, xe_di_len=True):
        """21 đặc trưng cho mọi cặp ứng viên (khung i ở frame trước, khung j ở frame sau) – giống cap_ung_vien."""
        ly = (fa.y_vach + fb.y_vach) / 2
        Ka, da, Za, Ha = self.khung_xe(fa, self.nguong_vao_C)
        Kb, db, Zb, Hb = self.khung_xe(fb, self.nguong_vao_C)
        if len(Ka) == 0 or len(Kb) == 0:
            return []
        A, B = sang_xywh(Ka), sang_xywh(Kb)
        kc = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
        lech = np.abs(np.log((A[:, 2] * A[:, 3])[:, None] / (B[:, 2] * B[:, 3])[None]))
        hang_i, hang_j = kc.argsort(1).argsort(1), kc.argsort(0).argsort(0)
        hang = []
        for i, j in zip(*np.where((kc < np.maximum(A[:, None, 3], B[None, :, 3]) + 0.03) & (lech < 0.9))):
            xa, xb = A[i], B[j]; hc = (xa[3] + xb[3]) / 2
            hang.append({"i": int(i), "j": int(j), "p_a": float(da[i]), "p_b": float(db[j]), "p_min": float(min(da[i], db[j])),
                         "dx": (xb[0] - xa[0]) / hc, "dy": (xb[1] - xa[1]) / hc, "abs_dx": abs(xb[0] - xa[0]) / hc,
                         "kc": kc[i, j] / hc, "hang_i": hang_i[i, j], "hang_j": hang_j[i, j],
                         "lech_rong": np.log(xb[2] / xa[2]), "lech_cao": np.log(xb[3] / xa[3]),
                         "iou_ab": iou_ma_tran(Ka[[i]], Kb[[j]])[0, 0], "rong": xb[2], "cao": xb[3], "cao_rong": xb[3] / xb[2],
                         "day_a_vach": xa[1] - ly, "day_b_vach": xb[1] - ly, "so_khung_a": len(Ka), "so_khung_b": len(Kb),
                         "giong_hog": float(Za[i] @ Zb[j] / (np.linalg.norm(Za[i]) * np.linalg.norm(Zb[j]) + 1e-9)),
                         "giong_mau": float(np.sqrt(Ha[i] * Hb[j]).sum()),
                         "vuot_len": vuot_vach(xa[1], xb[1], ly, xe_di_len) == 1,
                         "khung_a": Ka[i], "khung_b": Kb[j]})
        return hang

    def ghep_xe(self, fa, fb, nguong_C=None, xe_di_len=True):
        """C dự đoán xác suất cùng xe cho mọi cặp ứng viên, rồi Hungarian ghép một-một (bỏ cặp có p < ngưỡng C)."""
        nguong_C = self.nguong_C if nguong_C is None else nguong_C
        v = self.cap_ung_vien(fa, fb, xe_di_len)
        if not v:
            return []
        X = np.array([[r[c] for c in self.dac_trung_C] for r in v], float)
        p = self.mo_hinh_C.predict_proba(X)[:, 1]
        for r, q in zip(v, p):
            r["p_C"] = float(q)
        g = [k for k in range(len(v)) if v[k]["p_C"] >= nguong_C]
        if not g:
            return []
        ii, jj = np.array([v[k]["i"] for k in g]), np.array([v[k]["j"] for k in g])
        ui, uj = np.unique(ii), np.unique(jj)
        chi_phi = np.full((len(ui), len(uj)), 10.0)
        for k in g:
            chi_phi[np.searchsorted(ui, v[k]["i"]), np.searchsorted(uj, v[k]["j"])] = 1 - v[k]["p_C"]
        giu = []
        for r_, c_ in zip(*linear_sum_assignment(chi_phi)):
            if chi_phi[r_, c_] < 10:
                giu.append(next(v[k] for k in g if v[k]["i"] == ui[r_] and v[k]["j"] == uj[c_]))
        return giu


class BoTheoDoi:
    """Chạy hệ thống trên chuỗi frame liên tiếp (video hoặc nhiều ảnh), giống chay_video của notebook:
    mỗi cặp frame liền nhau → C ghép xe → xe đi qua vạch + A báo đỏ ⇒ cảnh báo.
    Thêm: nối các cặp C ghép thành vết có ID (đầu ra 'ID và quỹ đạo' của C) và chỉ cảnh báo một lần cho mỗi ID (D3)."""

    def __init__(self, ht: HeThongCanhBao, y_vach, khung_den=None, che_do_den="auto", xe_di_len=True,
                 nguong_xe=None, nguong_C=None, lam_muot=LAM_MUOT_A, he_so_co=1.0):
        self.ht, self.y_vach, self.khung_den, self.che_do_den = ht, float(y_vach), khung_den, che_do_den
        self.he_so_co = float(he_so_co)
        self.xe_di_len = xe_di_len
        self.nguong_xe = ht.nguong_xe if nguong_xe is None else float(nguong_xe)
        self.nguong_C = ht.nguong_C if nguong_C is None else float(nguong_C)
        self.lam_muot = max(1, int(lam_muot))
        self.truoc = None
        self.id_truoc = {}          # chỉ số khung (ngưỡng vào C) ở frame trước → ID xe
        self.next_id = 1
        self.lich_su_den = []
        self.da_qua_vach = set()
        self.quy_dao = {}           # ID → danh sách (thời điểm, điểm đáy khung px 640)

    def den(self, img640):
        if self.che_do_den == "red":
            return "RED", {"red_light": 1.0}
        if self.che_do_den == "green":
            return "GREEN", {"green_light": 1.0}
        if self.khung_den is None:
            return "UNKNOWN", {}
        lop, xs = self.ht.doan_den(img640, self.khung_den)
        self.lich_su_den.append(lop)
        lop = Counter(self.lich_su_den[-self.lam_muot:]).most_common(1)[0][0]
        return TEN_DEN_VN.get(lop, "UNKNOWN"), xs

    def buoc(self, img640, t):
        """Xử lý một frame 640×640. Trả về dict kết quả của frame (khung xe, cặp C giữ, cảnh báo mới)."""
        trang_thai, xs = self.den(img640)
        fr = self.ht.quet_frame(img640, self.y_vach, self.he_so_co)
        K_hien, d_hien, _, _ = self.ht.khung_xe(fr, self.nguong_xe)          # khung B hiển thị (ngưỡng B)
        K_vao, _, _, _ = self.ht.khung_xe(fr, self.ht.nguong_vao_C)          # ứng viên đưa vào C
        giu, canh_moi, vuot_ok = [], [], 0
        id_nay = {}
        if self.truoc is not None:
            giu = self.ht.ghep_xe(self.truoc, fr, self.nguong_C, self.xe_di_len)
            for r in giu:
                tid = self.id_truoc.get(r["i"])
                if tid is None:
                    tid = self.next_id; self.next_id += 1
                    self.quy_dao[tid] = [(self.t_truoc, self._day(r["khung_a"]))]
                id_nay[r["j"]] = tid; r["id"] = tid
                self.quy_dao.setdefault(tid, []).append((t, self._day(r["khung_b"])))
                if r["vuot_len"] and tid not in self.da_qua_vach:
                    self.da_qua_vach.add(tid)                    # D3: mỗi xe (ID) chỉ xét một lần
                    if trang_thai == "RED":
                        canh_moi.append(r)
                    else:
                        vuot_ok += 1
        self.id_truoc, self.truoc, self.t_truoc = id_nay, fr, t
        return dict(t=t, den=trang_thai, xac_suat_den=xs, K=K_hien, d=d_hien, so_ung_vien=len(K_vao),
                    giu=giu, canh_bao=canh_moi, vuot_hop_le=vuot_ok, so_cua_so=len(fr.K))

    @staticmethod
    def _day(k):
        return float((k[0] + k[2]) / 2), float(k[3])
