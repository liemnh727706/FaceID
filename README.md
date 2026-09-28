# NNTH FaceID Service

AI service nhận dạng khuôn mặt cho hệ thống NNTH HCMUAF, dùng
[InsightFace](https://github.com/deepinsight/insightface) (SCRFD + ArcFace `buffalo_l`)
chạy CPU qua ONNX Runtime, và PaddleOCR cho OCR thẻ sinh viên.

## Cài đặt

```bat
cd /d E:\AI\claude\FaceID
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   REM roi sua FACEID_API_KEY
```

Lần chạy đầu tiên InsightFace tự tải model `buffalo_l` (~280MB) về `%USERPROFILE%\.insightface\models`.

## Chạy

```bat
run.bat
```

Service lắng nghe tại `http://127.0.0.1:8001`. Kiểm tra: `GET /health`.

## API (yêu cầu header `X-API-Key`)

| Endpoint | Chức năng |
|---|---|
| `POST /face/detect` | Tìm khuôn mặt lớn nhất trong ảnh thẻ, trả chân dung đã cắt (base64 JPEG) |
| `POST /face/verify` | So khớp 1:1 hai ảnh → similarity + quyết định match/uncertain/no_match |
| `POST /face/embed` | Tính embedding 512 chiều cho 1..n ảnh |
| `POST /face/identify` | So khớp 1:N một embedding với gallery (dùng cho điểm danh phòng thi) |
| `POST /ocr/student-card` | OCR thẻ sinh viên → các dòng chữ + trường đoán được (MSSV, họ tên, lớp) |
| `POST /qr/decode` | Đọc mã QR (CCCD) — 3 tầng: WeChatQRCode (CNN) → pyzbar → OpenCV Aruco |

## Model WeChatQRCode

Bộ đọc QR mạnh nhất (CNN phát hiện + siêu phân giải), model tải từ GitHub chính thức
[WeChatCV/opencv_3rdparty](https://github.com/WeChatCV/opencv_3rdparty) (nhánh `wechat_qrcode`)
vào `models/wechat_qrcode/` (4 file: detect.prototxt/caffemodel, sr.prototxt/caffemodel).
Thiếu model thì service tự hạ xuống tầng pyzbar/Aruco, không lỗi.

## Ngưỡng nhận dạng

Cosine similarity giữa selfie và ảnh chân dung in trên thẻ:
- `>= FACEID_MATCH_THRESHOLD` (mặc định 0.32): khớp
- `< FACEID_REJECT_THRESHOLD` (mặc định 0.20): không khớp
- ở giữa: không chắc chắn — cần admin duyệt tay

Ảnh in trên thẻ chất lượng thấp nên ngưỡng thấp hơn so khớp selfie–selfie.
Sau khi có dữ liệu thật, tinh chỉnh hai ngưỡng này trong `.env`.
