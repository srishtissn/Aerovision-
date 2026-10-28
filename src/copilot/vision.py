"""
Optional second opinion from a vision LLM (Gemini) on a flagged crop.

The model is asked *blind* (it is not told what the detector found) so the
two opinions are independent; agreement is computed afterwards. The second
opinion is advisory: it can raise a "needs human review" flag but never
lowers severity or changes the rule-based risk decision.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .schemas import VisionOpinion

PROMPT = """You are inspecting a cropped photo of an aircraft skin panel region.
Decide independently whether it shows surface damage. Choose defect_type from:
crack, corrosion, dent, rivet_damage, other, none.
Return ONLY JSON: {"defect_present": bool, "defect_type": str, "confidence": 0-1, "rationale": "one or two sentences describing only what is visible"}"""


def crop_region(image: np.ndarray, bbox, pad_frac: float = 0.25, min_side: int = 96) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox]
    px, py = int(bw * pad_frac) + 8, int(bh * pad_frac) + 8
    x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(w, x + bw + px), min(h, y + bh + py)
    crop = image[y0:y1, x0:x1]
    s = max(1.0, min_side / max(1, min(crop.shape[:2])))
    if s > 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return crop


def second_opinion(image: np.ndarray, detection: Any, gemini) -> Dict[str, Any]:
    """`gemini` is a GeminiClient (needs .generate_with_image). Returns a plain dict."""
    bbox = detection["bbox"] if isinstance(detection, dict) else detection.bbox
    dtype = detection["defect_type"] if isinstance(detection, dict) else detection.defect_type
    crop = crop_region(image, bbox)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return {"error": "could not encode crop"}
    try:
        raw = gemini.generate_with_image(PROMPT, buf.tobytes())
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        op = VisionOpinion.model_validate(json.loads(t))
    except Exception as e:
        return {"error": f"vision second opinion failed: {e}"}
    if not op.defect_present or op.defect_type == "none":
        agreement = "disagree"
    elif op.defect_type == dtype:
        agreement = "agree"
    else:
        agreement = "type_disagree"
    return {
        "detector_type": dtype, "vision_defect_present": op.defect_present,
        "vision_type": op.defect_type, "vision_confidence": round(op.confidence, 2),
        "vision_rationale": op.rationale, "agreement": agreement,
        "needs_human_review": agreement != "agree",
        "model": getattr(gemini, "name", "gemini"),
    }
