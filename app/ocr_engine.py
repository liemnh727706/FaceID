"""OCR the sinh vien bang PaddleOCR (model latin doc duoc tieng Viet co dau)."""
import re
import threading

import numpy as np

from . import config

_lock = threading.Lock()
_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        with _lock:
            if _ocr is None:
                from paddleocr import PaddleOCR

                # PaddleOCR 3.x; tat cac module phu de nhanh tren CPU
                # enable_mkldnn=False: ne bug oneDNN cua paddlepaddle 3.3 tren CPU
                # ("ConvertPirAttribute2RuntimeAttribute not support")
                _ocr = PaddleOCR(
                    lang="vi",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                    enable_mkldnn=False,
                )
    return _ocr


def run_ocr(img: np.ndarray) -> list[dict]:
    """Tra ve danh sach dong chu: [{text, confidence}] theo thu tu tren xuong."""
    result = get_ocr().predict(img)
    lines = []
    for res in result or []:
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        polys = res.get("rec_polys")
        if polys is None:
            polys = res.get("dt_polys") or []
        for i, text in enumerate(texts):
            text = (text or "").strip()
            if not text:
                continue
            conf = float(scores[i]) if i < len(scores) else 0.0
            y = float(min(p[1] for p in polys[i])) if i < len(polys) else float(i)
            lines.append({"text": text, "confidence": round(conf, 4), "y": y})
    lines.sort(key=lambda l: l["y"])
    return [{"text": l["text"], "confidence": l["confidence"]} for l in lines]


_MSSV_RE = re.compile(r"\b(\d{8,10})\b")


def _ascii(s: str) -> str:
    """Bo dau tieng Viet (OCR hay lam rot dau nen phai so khop khong dau)."""
    import unicodedata

    s = unicodedata.normalize("NFD", s.replace("Đ", "D").replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _ascii(s))


_NAME_LABELS = ("hovaten", "hoten", "hova1en")
_MSSV_LABELS = ("mssv", "masv", "masosinhvien", "masosv", "msv")
_EXCLUDE_SQUASHED = ("truong", "daihoc", "thesinhvien", "sinhvien", "khoa", "nganh", "nienkhoa")

# Dong chu hoa (ho ten in tren the) — cho phep mat dau sau OCR
_UPPER_LINE_RE = re.compile(r"^[^\Wa-z0-9_][^a-z0-9]{5,}$")


def extract_student_card_fields(lines: list[dict]) -> dict:
    """Doan cac truong tren the sinh vien tu ket qua OCR.

    Chi la goi y tot nhat — phia server se doi chieu voi tai khoan,
    khong bao gio tin tuyet doi vao OCR.
    """
    fields = {"student_code": None, "full_name": None, "class_name": None}
    texts = [l["text"] for l in lines]

    # Uu tien cac dong co nhan "Nhan: gia tri"
    for i, text in enumerate(texts):
        if ":" not in text:
            continue
        label_raw, value = text.split(":", 1)
        label = _squash(label_raw)
        value = value.strip(" :.-")

        if fields["full_name"] is None and any(k in label for k in _NAME_LABELS):
            fields["full_name"] = value if len(value) >= 4 else (
                texts[i + 1].strip(" :.-") if i + 1 < len(texts) else None
            )
        elif fields["student_code"] is None and any(k in label for k in _MSSV_LABELS):
            m = _MSSV_RE.search(value)
            if m:
                fields["student_code"] = m.group(1)
        elif fields["class_name"] is None and label.endswith("lop") and value:
            fields["class_name"] = value

    # Du phong khi khong co nhan ro rang
    for text in texts:
        squashed = _squash(text)
        if fields["student_code"] is None:
            m = _MSSV_RE.search(text)
            if m and not any(k in squashed for k in ("khoa", "nam", "nien")):
                fields["student_code"] = m.group(1)
        if (
            fields["full_name"] is None
            and _UPPER_LINE_RE.match(text.strip())
            and not any(k in squashed for k in _EXCLUDE_SQUASHED)
            and not any(ch.isdigit() for ch in text)
        ):
            fields["full_name"] = text.strip()

    return fields
