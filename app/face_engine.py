"""Loi nhan dang khuon mat dua tren InsightFace (SCRFD + ArcFace buffalo_l), chay CPU."""
import threading

import cv2
import numpy as np

from . import config

_lock = threading.Lock()
_app = None


def get_app():
    global _app
    if _app is None:
        with _lock:
            if _app is None:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(config.DET_SIZE, config.DET_SIZE))
                _app = app
    return _app


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Khong doc duoc du lieu anh")
    # Thu nho anh qua lon (anh chup tu dien thoai co the 4000px+)
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > 1920:
        scale = 1920 / max_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def detect_faces(img: np.ndarray):
    return get_app().get(img)


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def face_size(face) -> int:
    x1, y1, x2, y2 = face.bbox
    return int(min(x2 - x1, y2 - y1))


def crop_portrait(img: np.ndarray, face, margin: float = 0.45) -> np.ndarray:
    """Cat chan dung quanh bbox khuon mat, them le de giu ca toc/cam."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * margin))
    y1 = int(max(0, y1 - bh * margin * 1.3))
    x2 = int(min(w, x2 + bw * margin))
    y2 = int(min(h, y2 + bh * margin))
    return img[y1:y2, x1:x2]


def encode_jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Khong ma hoa duoc anh JPEG")
    return buf.tobytes()


def normed_embedding(face) -> np.ndarray:
    emb = face.embedding.astype(np.float32)
    return emb / np.linalg.norm(emb)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def identify(probe: np.ndarray, gallery: list[dict]) -> dict | None:
    """So khop 1:N. gallery: [{"id": str, "embeddings": [[512 floats], ...]}]."""
    best = None
    for person in gallery:
        for emb in person["embeddings"]:
            sim = cosine(probe, emb)
            if best is None or sim > best["similarity"]:
                best = {"id": person["id"], "similarity": sim}
    return best
