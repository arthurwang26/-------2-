import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont

# COCO 17-keypoint skeleton connections
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

# Try loading Chinese font
_FONT = None
_FONT_SMALL = None
def _get_fonts():
    global _FONT, _FONT_SMALL
    if _FONT is None:
        for fp in ["C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msyh.ttc",
                    "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/arial.ttf"]:
            if Path(fp).exists():
                _FONT = ImageFont.truetype(fp, 20)
                _FONT_SMALL = ImageFont.truetype(fp, 14)
                break
        if _FONT is None:
            _FONT = ImageFont.load_default()
            _FONT_SMALL = _FONT
    return _FONT, _FONT_SMALL


def _put_chinese_text(img: np.ndarray, text: str, pos: tuple, color: tuple, font=None):
    """Use PIL to render Chinese text on OpenCV image."""
    if font is None:
        font, _ = _get_fonts()
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    # Background rectangle for readability
    bbox = draw.textbbox(pos, text, font=font)
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(0, 0, 0))
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


IDENTITY_COLORS = {
    "王奶奶": (0, 255, 0),    # Green
    "陳爺爺": (255, 165, 0),  # Orange
}


def draw_frame(
    frame: np.ndarray,
    detections: List[dict] = None,
    skeletons: Dict[int, np.ndarray] = None,
    identities: Dict[int, str] = None,
    actions: Dict[int, str] = None,
    emotions: Dict[int, str] = None,
    objects: List[dict] = None,
) -> np.ndarray:
    """Draw bbox, skeleton, ID, action, emotion on a single frame."""
    out = frame.copy()
    detections = detections or []
    skeletons = skeletons or {}
    identities = identities or {}
    actions = actions or {}
    emotions = emotions or {}

    font, font_small = _get_fonts()

    # Draw object detections (non-person)
    if objects:
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label = obj.get("class_name", "?")
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 0), 1)
            out = _put_chinese_text(out, label, (x1, y1 - 18), (200, 200, 0), font_small)

    # Draw person detections with identity
    for det in detections:
        tid = det.get("track_id", -1)
        if tid == -1:
            continue

        x1, y1, x2, y2 = map(int, det["bbox"])
        pid = identities.get(tid, f"ID:{tid}")
        color = IDENTITY_COLORS.get(pid, (128, 128, 128))

        # Bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label
        parts = [pid]
        if tid in actions and actions[tid]:
            parts.append(actions[tid])
        if tid in emotions and emotions[tid]:
            parts.append(emotions[tid])
        label = " | ".join(parts)
        out = _put_chinese_text(out, label, (x1, max(0, y1 - 24)), color, font)

        # Draw skeleton
        if tid in skeletons:
            kps = skeletons[tid]
            for kp in kps:
                x, y, c = kp
                if c > 0.3:
                    cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)
            for e1, e2 in SKELETON_EDGES:
                if e1 < len(kps) and e2 < len(kps):
                    kp1, kp2 = kps[e1], kps[e2]
                    if kp1[2] > 0.3 and kp2[2] > 0.3:
                        cv2.line(out, (int(kp1[0]), int(kp1[1])),
                                 (int(kp2[0]), int(kp2[1])), (255, 0, 0), 2)
    return out
