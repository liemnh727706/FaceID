"""Doc ma QR tren CCCD — 3 tang, tu manh den yeu:

1. WeChatQRCode (opencv-contrib): CNN phat hien QR + mang sieu phan giai,
   xu ly duoc QR nho, mo, loa mot phan, nghieng. Model tu GitHub chinh thuc:
   https://github.com/WeChatCV/opencv_3rdparty (nhanh wechat_qrcode)
2. pyzbar (ZBar): decoder truyen thong rat ben bi voi anh nhieu.
3. cv2.QRCodeDetectorAruco: du phong cuoi, kem tien xu ly CLAHE/threshold.
"""
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("faceid.qr")

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "wechat_qrcode"
_lock = threading.Lock()
_wechat = None
_wechat_failed = False


def _get_wechat():
    global _wechat, _wechat_failed
    if _wechat is None and not _wechat_failed:
        with _lock:
            if _wechat is None and not _wechat_failed:
                try:
                    files = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
                    paths = [str(_MODEL_DIR / f) for f in files]
                    if not all((_MODEL_DIR / f).exists() for f in files):
                        raise FileNotFoundError(f"Thieu model WeChatQRCode trong {_MODEL_DIR}")
                    _wechat = cv2.wechat_qrcode.WeChatQRCode(*paths)
                except Exception as e:
                    logger.warning(f"WeChatQRCode khong kha dung, dung tang du phong: {e}")
                    _wechat_failed = True
    return _wechat


def _resize_max(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _gray_variants(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    yield gray
    yield cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)


def _decode_wechat(img) -> str | None:
    det = _get_wechat()
    if det is None:
        return None
    # WeChatQRCode tu phat hien vung QR; thu o vai do phan giai
    for max_side in (1600, 2400, 1000):
        try:
            texts, _ = det.detectAndDecode(_resize_max(img, max_side))
        except cv2.error:
            continue
        for t in texts:
            if t:
                return t
    return None


def _decode_zbar(img) -> str | None:
    try:
        from pyzbar import pyzbar
    except Exception as e:
        logger.warning(f"pyzbar khong kha dung: {e}")
        return None
    for variant in _gray_variants(_resize_max(img, 2400)):
        for scale in (1.0, 1.5):
            v = variant if scale == 1.0 else cv2.resize(
                variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            try:
                results = pyzbar.decode(v)
            except Exception:
                continue
            for r in results:
                if r.type == "QRCODE" and r.data:
                    try:
                        return r.data.decode("utf-8")
                    except UnicodeDecodeError:
                        return r.data.decode("utf-8", errors="replace")
    return None


def _decode_aruco(img) -> str | None:
    det = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else cv2.QRCodeDetector()
    h, w = img.shape[:2]
    regions = [img, img[: h * 2 // 3, w // 2 :], img[:, w // 2 :], img[: h // 2, :]]
    for region in regions:
        if min(region.shape[:2]) < 50:
            continue
        for variant in _gray_variants(_resize_max(region, 2000)):
            try:
                data, _, _ = det.detectAndDecode(variant)
            except cv2.error:
                continue
            if data:
                return data
    return None


def decode_qr(image_bytes: bytes) -> str | None:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    for name, fn in (("wechat", _decode_wechat), ("zbar", _decode_zbar), ("aruco", _decode_aruco)):
        data = fn(img)
        if data:
            logger.info(f"QR decoded boi tang '{name}'")
            return data
    return None
