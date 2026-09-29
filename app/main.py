import base64
import logging

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import config, face_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("faceid")

app = FastAPI(title="NNTH FaceID Service", version="0.1.0")


async def require_api_key(x_api_key: str | None = Header(default=None)):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="API key khong hop le")


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File anh rong")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File anh qua lon (toi da 15MB)")
    return data


def _decode_detect_embed(data: bytes):
    """Chay dong bo, CPU-bound (decode anh + inference ONNX) - luon goi qua run_in_threadpool
    de khong chan event loop cua Uvicorn trong luc suy luan."""
    img = face_engine.decode_image(data)
    face = face_engine.largest_face(face_engine.detect_faces(img))
    if face is None:
        return img, None, None
    return img, face, face_engine.normed_embedding(face)


def _crop_and_encode(img, face) -> bytes:
    return face_engine.encode_jpeg(face_engine.crop_portrait(img, face))


def face_info(face) -> dict:
    return {
        "bbox": [round(float(v), 1) for v in face.bbox],
        "det_score": round(float(face.det_score), 4),
        "size": face_engine.face_size(face),
    }


@app.get("/")
def root():
    return {
        "service": "NNTH FaceID",
        "message": "Service dang chay. Xem /health de kiem tra, /docs de thu API.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "face_model_loaded": face_engine._app is not None,
        "ocr_enabled": config.OCR_ENABLED,
        "thresholds": {"match": config.MATCH_THRESHOLD, "reject": config.REJECT_THRESHOLD},
    }


@app.post("/face/detect", dependencies=[Depends(require_api_key)])
async def detect(image: UploadFile = File(...)):
    """Tim khuon mat lon nhat trong anh (vd anh the) va cat chan dung."""
    img, face, _ = await run_in_threadpool(_decode_detect_embed, await read_upload(image))
    if face is None:
        raise HTTPException(status_code=422, detail="Khong tim thay khuon mat trong anh")
    if face_engine.face_size(face) < config.MIN_FACE_SIZE:
        raise HTTPException(
            status_code=422,
            detail="Khuon mat qua nho — vui long chup gan hon hoac anh ro hon",
        )
    portrait_jpeg = await run_in_threadpool(_crop_and_encode, img, face)
    return {
        "face": face_info(face),
        "portrait_jpeg_b64": base64.b64encode(portrait_jpeg).decode(),
    }


@app.post("/face/embed", dependencies=[Depends(require_api_key)])
async def embed(images: list[UploadFile] = File(...)):
    """Tinh embedding 512 chieu cho khuon mat lon nhat cua tung anh."""
    embeddings, faces = [], []
    for upload in images:
        _, face, emb = await run_in_threadpool(_decode_detect_embed, await read_upload(upload))
        if face is None:
            raise HTTPException(
                status_code=422, detail=f"Khong tim thay khuon mat trong anh '{upload.filename}'"
            )
        embeddings.append([round(float(v), 6) for v in emb])
        faces.append(face_info(face))
    return {"embeddings": embeddings, "faces": faces}


@app.post("/face/verify", dependencies=[Depends(require_api_key)])
async def verify(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    """So khop 1:1 hai anh (vd chan dung tren the vs selfie)."""
    embs = []
    for upload in (image_a, image_b):
        _, face, emb = await run_in_threadpool(_decode_detect_embed, await read_upload(upload))
        if face is None:
            raise HTTPException(
                status_code=422, detail=f"Khong tim thay khuon mat trong anh '{upload.filename}'"
            )
        if face_engine.face_size(face) < config.MIN_FACE_SIZE:
            raise HTTPException(
                status_code=422, detail=f"Khuon mat qua nho trong anh '{upload.filename}'"
            )
        embs.append(emb)

    similarity = await run_in_threadpool(face_engine.cosine, embs[0], embs[1])
    if similarity >= config.MATCH_THRESHOLD:
        decision = "match"
    elif similarity < config.REJECT_THRESHOLD:
        decision = "no_match"
    else:
        decision = "uncertain"
    return {
        "similarity": round(similarity, 4),
        "decision": decision,
        "thresholds": {"match": config.MATCH_THRESHOLD, "reject": config.REJECT_THRESHOLD},
    }


class GalleryPerson(BaseModel):
    id: str
    embeddings: list[list[float]]


class IdentifyRequest(BaseModel):
    probe_embedding: list[float]
    gallery: list[GalleryPerson]


@app.post("/face/identify", dependencies=[Depends(require_api_key)])
async def identify(req: IdentifyRequest):
    """So khop 1:N probe voi danh sach gallery (dung cho diem danh phong thi)."""
    if not req.gallery:
        raise HTTPException(status_code=400, detail="Gallery rong")
    best = await run_in_threadpool(
        face_engine.identify, req.probe_embedding, [p.model_dump() for p in req.gallery]
    )
    matched = best is not None and best["similarity"] >= config.MATCH_THRESHOLD
    return {
        "best_id": best["id"] if best else None,
        "similarity": round(best["similarity"], 4) if best else None,
        "matched": matched,
    }


@app.post("/qr/decode", dependencies=[Depends(require_api_key)])
async def qr_decode(image: UploadFile = File(...)):
    """Doc ma QR trong anh (du phong khi trinh duyet khong doc duoc)."""
    from . import qr_engine

    data = await run_in_threadpool(qr_engine.decode_qr, await read_upload(image))
    if not data:
        raise HTTPException(
            status_code=422,
            detail="Khong tim thay ma QR trong anh. Chup thang, du sang, ro o QR.",
        )
    return {"qr_raw": data}


@app.post("/ocr/student-card", dependencies=[Depends(require_api_key)])
async def ocr_student_card(image: UploadFile = File(...)):
    """OCR the sinh vien: tra ve cac dong chu va cac truong doan duoc."""
    if not config.OCR_ENABLED:
        raise HTTPException(status_code=503, detail="OCR dang tat (FACEID_OCR_ENABLED=0)")
    from . import ocr_engine

    data = await read_upload(image)

    def _ocr(data: bytes):
        img = face_engine.decode_image(data)
        lines = ocr_engine.run_ocr(img)
        return lines, ocr_engine.extract_student_card_fields(lines)

    lines, fields = await run_in_threadpool(_ocr, data)
    return {"lines": lines, "fields": fields}
