# Web giám sát vượt đèn đỏ · Nhóm 6

Web chạy đúng ba mô hình đã huấn luyện trong notebook (file `he_thong_canh_bao.joblib`):

| Khối | Mô hình | Đầu vào |
| --- | --- | --- |
| A · màu đèn | k-NN (k=9, distance, manhattan) | 44 đặc trưng HSV của vùng đèn, chuẩn hóa StandardScaler |
| B · tìm xe | SVM-RBF (C=10), bản "+ tăng cường + nền khó 2 vòng", ngưỡng 0,8 | HOG 64×64 (1.764 chiều) → PCA 150, cửa sổ trượt theo dải quanh vạch dừng, NMS 0,2 + bỏ phiếu |
| C · theo dõi phương tiện | k-NN (k=25), ngưỡng 0,4 + Hungarian | 21 đặc trưng của cặp khung xe ở hai frame liền nhau |
| Bước kết luận | luật (không huấn luyện) | xe C ghép có đáy khung đi qua vạch khi A báo đỏ, bỏ phiếu đèn 3 frame, mỗi ID cảnh báo một lần |

## Cấu trúc thư mục

```
app.py                     giao diện Streamlit
he_thong.py                lõi suy luận A → B → C (chép đúng code notebook)
requirements.txt           scikit-learn==1.6.1 (cùng phiên bản đã lưu mô hình)
he_thong_canh_bao.joblib   mô hình đã huấn luyện (notebook mục E)
ket_qua_tong_hop.json      số liệu đánh giá (tab "Hiệu năng mô hình")
video_mau_test.mp4         video ghép từ frame test (notebook mục F1)
.streamlit/config.toml     giao diện tối, giới hạn tải lên 300 MB
```

app.py tìm các file mô hình ở thư mục gốc hoặc trong `mo_hinh/`.

Chạy: `pip install -r requirements.txt` rồi `streamlit run app.py`.

## Khác biệt nhỏ so với notebook

Notebook mục B4 chọn thêm **hồi quy đáy khung (Ridge)** cho B nhưng không lưu vào joblib, nên web bỏ bước này
(khung xe lệch vài pixel; trên video test của notebook web cho cùng kết quả: 0 xe qua vạch, 0 cảnh báo, màu đèn
khớp từng frame). Muốn web dùng đủ, sửa lệnh `joblib.dump` ở ô mục E của notebook, thêm 4 khoá rồi chạy lại ô đó:

```python
"tinh_chinh_B": TINH_CHINH, "nms_iou": NMS_IOU, "vote_iou": VOTE_IOU, "bo_phieu": BO_PHIEU,
```

`he_thong.py` tự đọc các khoá này nếu có.

## Tốc độ

B chấm khoảng 1.500 cửa sổ mỗi frame bằng SVM-RBF: khoảng 3 đến 6 giây mỗi frame trên CPU (notebook đo 5,4 s).
Web chỉ xét frame cách nhau `buoc_giay` giây (mặc định 0,3 s như notebook) và tối đa N frame (mặc định 40).
