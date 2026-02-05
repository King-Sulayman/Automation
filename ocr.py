import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import cv2
import numpy as np
import pytesseract


ALNUM_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
RE_ALNUM = re.compile(r"[^A-Z0-9]+")


def _read_png_bytes_truncated(path: Path) -> bytes:
    """
    Some of your samples have trailing HTML appended after the PNG IEND chunk.
    This safely truncates to the true PNG payload so decoders don't fail.
    """
    b = path.read_bytes()
    iend = b.rfind(b"IEND")
    if iend == -1:
        return b
    # Include 'IEND' (4 bytes) + CRC (4 bytes)
    end = iend + 8
    if end <= len(b):
        return b[:end]
    return b


def _load_image_bgr(path: Path) -> np.ndarray:
    raw = _read_png_bytes_truncated(path)
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image (is it a valid PNG/JPG?).")
    return img


def _ensure_black_text_on_white(bin_img: np.ndarray) -> np.ndarray:
    # We want black text (0) on white background (255) for Tesseract.
    # If the image is mostly dark, invert it.
    if float(np.mean(bin_img)) < 127.0:
        return cv2.bitwise_not(bin_img)
    return bin_img


def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    # Grayscale
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale for better character shapes
    g = cv2.resize(g, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # Mild denoise
    g = cv2.GaussianBlur(g, (3, 3), 0)

    # Binarize (fast + robust for simple CAPTCHAs)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = _ensure_black_text_on_white(th)

    # Small morphology to clean noise without eating characters
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return th


def _tesseract_ocr(bin_img: np.ndarray, psm: int) -> str:
    # psm 7: single text line, psm 8: single word. We'll try both.
    config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={ALNUM_UPPER}"
    s = pytesseract.image_to_string(bin_img, config=config)
    s = (s or "").upper()
    s = RE_ALNUM.sub("", s)
    return s


def _best_6char_candidate(cands: Iterable[str]) -> Optional[str]:
    cands = [c for c in cands if c]
    exact = [c for c in cands if len(c) == 6]
    if exact:
        # Prefer the most frequent exact match if OCR returns duplicates
        return max(set(exact), key=exact.count)
    # Otherwise pick something close-ish (often 5 or 7 due to noise)
    if cands:
        return min(cands, key=lambda c: abs(len(c) - 6))
    return None


@dataclass(frozen=True)
class OcrResult:
    path: str
    text: str


def ocr_captcha(path: str) -> OcrResult:
    p = Path(path)
    img = _load_image_bgr(p)
    th = _preprocess_for_ocr(img)

    c1 = _tesseract_ocr(th, psm=7)
    c2 = _tesseract_ocr(th, psm=8)
    # Also try inverted binary (sometimes captchas come inverted)
    inv = cv2.bitwise_not(th)
    c3 = _tesseract_ocr(inv, psm=7)
    c4 = _tesseract_ocr(inv, psm=8)

    best = _best_6char_candidate([c1, c2, c3, c4]) or ""
    return OcrResult(path=str(p), text=best)

def ocr_one_image(image_path: str) -> str:
    """
    OCR exactly one local image path and return the best 6-character A-Z0-9 result.

    Example:
        from ocr import ocr_one_image
        text = ocr_one_image("captcha.png")
    """
    return ocr_captcha(image_path).text