# Triển khai FaceID service lên `cropnlu.duckdns.org`

VPS: `ubuntu@instance-20260625-1050` (161.118.254.111), compose dir `/home/ubuntu/cropguard`.
4 vCPU, 23GB RAM (đã kiểm tra 2026-09-28, còn dư nhiều) — đủ chạy thêm InsightFace + PaddleOCR
cạnh cropguard/weather_server hiện có.

Route theo path `/faceid/` dưới domain có sẵn (KHÔNG tạo subdomain mới) vì cert hiện tại chỉ có
1 tên miền `cropnlu.duckdns.org`, không có SAN cho subdomain khác.

## 1. Chuẩn bị mã nguồn

Trên máy dev (máy này), thư mục cần đưa lên VPS là `E:\AI\claude\FaceID`, gồm `app/`, `models/`,
`deploy/`. **Không copy `.venv/`, không copy `.env` cũ** (chứa key dev).

```bash
# Từ máy dev — nén phần cần thiết, loại bỏ .venv và ảnh test
cd /e/AI/claude/FaceID
tar --exclude='.venv' --exclude='test_*.jpg' -czf faceid-deploy.tar.gz app models deploy

# Copy lên VPS (thay bằng cách bạn hay dùng: scp/rsync)
scp faceid-deploy.tar.gz ubuntu@161.118.254.111:/home/ubuntu/

# Trên VPS
ssh ubuntu@161.118.254.111
mkdir -p /home/ubuntu/faceid-service
tar -xzf faceid-deploy.tar.gz -C /home/ubuntu/faceid-service
```

## 2. Tạo `.env` production riêng (không dùng lại key dev)

```bash
cat > /home/ubuntu/faceid-service/.env <<'EOF'
FACEID_API_KEY=<tạo key mới, dài, ngẫu nhiên — KHÔNG dùng dev-faceid-key-nnth-2026>
FACEID_HOST=0.0.0.0
FACEID_PORT=8001
FACEID_MATCH_THRESHOLD=0.32
FACEID_REJECT_THRESHOLD=0.20
FACEID_MIN_FACE_SIZE=60
FACEID_OCR_ENABLED=1
EOF
chmod 600 /home/ubuntu/faceid-service/.env
```

Tạo key ngẫu nhiên nhanh: `openssl rand -hex 32`.

⚠️ **Bắt buộc đặt `FACEID_API_KEY` khác rỗng.** Code hiện tại (`app/main.py`,
`require_api_key`) chỉ kiểm tra key khi `config.API_KEY` khác rỗng — nếu quên set biến này,
endpoint sẽ **mở hoàn toàn không cần xác thực** cho bất kỳ ai trên Internet gọi tới.

## 3. Ghép vào docker-compose hiện có

Mở `/home/ubuntu/cropguard/docker-compose.yml`, thêm nội dung trong
[docker-compose.faceid.snippet.yml](docker-compose.faceid.snippet.yml) vào phần `services:` và
`volumes:` đã có. **Kiểm tra tên network** cropguard_nginx đang dùng
(`docker compose config | grep -A3 "cropguard_nginx" | grep network`, hoặc xem file compose hiện
tại) — nếu không phải `default`, sửa lại giá trị `networks:` trong đoạn thêm cho khớp.

```bash
cd /home/ubuntu/cropguard
docker compose build faceid
docker compose up -d faceid
docker compose logs -f faceid   # theo dõi lần chạy đầu — InsightFace tải model buffalo_l (~280MB)
```

Kiểm tra nội bộ trước khi động vào nginx:

```bash
docker compose exec cropguard_nginx wget -qO- http://faceid:8001/health
```

## 4. Thêm route Nginx

Thêm nội dung [nginx-faceid.snippet.conf](nginx-faceid.snippet.conf) vào
`/home/ubuntu/cropguard/nginx/nginx.conf`, trong `server{}` block HTTPS (443), đặt **trước**
`location /` của cropguard.

**Bẫy đã biết**: `nginx.conf` là bind-mount file đơn lẻ read-only. Sửa xong bắt buộc:

```bash
docker restart cropguard_nginx    # KHÔNG dùng `nginx -s reload`, không nhận file mới
```

## 5. Kiểm tra từ bên ngoài

```bash
curl https://cropnlu.duckdns.org/faceid/health

curl -X POST https://cropnlu.duckdns.org/faceid/face/verify \
  -H "X-API-Key: <key production vừa tạo>" \
  -F "image_a=@test_cccd.jpg" \
  -F "image_b=@test_selfie.jpg"
```

## 6. Cập nhật app Android

Trong app, đổi:
- Địa chỉ FaceID service: `https://cropnlu.duckdns.org/faceid`
- API key: key production vừa tạo (không phải key dev)

Sau khi xác nhận chạy ổn qua HTTPS, có thể bỏ `android:usesCleartextTraffic="true"` trong
`AndroidManifest.xml` vì không còn cần gọi HTTP thường trong LAN nữa.

## Việc chưa làm / cần cân nhắc thêm

- **Giới hạn tốc độ gọi (rate limit)**: endpoint giờ lộ ra Internet công khai (dù có API key),
  nên thêm `limit_req` ở Nginx để tránh bị dò key hoặc gây quá tải CPU của InsightFace.
- **Log/giám sát**: chưa có cơ chế theo dõi request thất bại hay dung lượng RAM khi nhiều app
  cùng gọi đồng thời — InsightFace CPU inference là tác vụ nặng, nên theo dõi `docker stats` sau
  khi có traffic thật.
- **Backup `.env` production**: lưu key ở nơi an toàn (không commit vào git).
