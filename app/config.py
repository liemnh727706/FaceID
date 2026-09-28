import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FACEID_API_KEY", "")
HOST = os.getenv("FACEID_HOST", "127.0.0.1")
PORT = int(os.getenv("FACEID_PORT", "8001"))

# Nguong cosine similarity giua selfie va anh chan dung tren the.
# Anh in tren the chat luong thap nen nguong thap hon nguong selfie-selfie thong thuong.
MATCH_THRESHOLD = float(os.getenv("FACEID_MATCH_THRESHOLD", "0.32"))
REJECT_THRESHOLD = float(os.getenv("FACEID_REJECT_THRESHOLD", "0.20"))

# Kich thuoc toi thieu (pixel) cua khuon mat trong anh de chap nhan xu ly
MIN_FACE_SIZE = int(os.getenv("FACEID_MIN_FACE_SIZE", "60"))

DET_SIZE = int(os.getenv("FACEID_DET_SIZE", "640"))
OCR_ENABLED = os.getenv("FACEID_OCR_ENABLED", "1") == "1"
