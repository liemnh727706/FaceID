# Build: từ thư mục gốc FaceID (không phải deploy/), vì cần COPY app/ và models/
#   docker build -f deploy/Dockerfile -t faceid-service .
FROM python:3.12-slim

# Thư viện hệ thống opencv/paddle cần khi chạy trên Linux không có sẵn X11/GL:
#   libgl1, libglib2.0-0: opencv (kể cả bản headless vẫn link tới libGL ở một số bản build)
#   libgomp1: OpenMP runtime cho onnxruntime/paddlepaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
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
