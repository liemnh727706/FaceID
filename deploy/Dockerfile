# Build: từ thư mục gốc FaceID (không phải deploy/), vì cần COPY app/ và models/
#   docker build -f deploy/Dockerfile -t faceid-service .
FROM python:3.12-slim

# Thư viện hệ thống cần khi chạy trên Linux không có sẵn X11/GL:
#   libgl1, libglib2.0-0: opencv (kể cả bản headless vẫn link tới libGL ở một số bản build)
#   libgomp1: OpenMP runtime cho onnxruntime
#   build-essential, python3-dev: PyPI không có wheel insightface dựng sẵn cho ARM64/aarch64,
#     pip phải tự biên dịch phần mở rộng C++ (mesh_core_cython) từ mã nguồn khi cài trên VPS ARM64.
#   libzbar0: thư viện hệ thống mà pyzbar (fallback giải mã QR tầng 2) cần qua ctypes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 build-essential python3-dev libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/faceid

COPY deploy/requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/

# InsightFace tự tải model buffalo_l (~280MB) về ~/.insightface lần chạy đầu tiên.
# Container chạy bằng user root (mặc định của base image) nên đường dẫn thật là /root/.insightface -
# mount volume vào đúng đường dẫn này (xem docker-compose.faceid.yml) để không tải lại mỗi lần
# dựng lại container. Nếu sau này đổi sang non-root user, phải sửa lại đường dẫn volume tương ứng.

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
