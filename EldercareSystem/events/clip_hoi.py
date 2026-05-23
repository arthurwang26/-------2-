"""
Zero-Shot HOI Detection using OpenAI CLIP.
Extracts the union bounding box of a person and an object, and uses CLIP to score
the interaction against a set of predefined textual prompts.
"""
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger("clip_hoi")

CLIP_HOI_PROMPTS = {
    "No_Interaction": "A photo of a person and an object.",
    "Touching": "A photo of a person touching an object.",
    "Holding": "A photo of a person holding an object.",
    "Sitting_On": "A photo of a person sitting on a chair or couch.",
    "Looking_At": "A photo of a person looking at an object or screen.",
    "Using": "A photo of a person using an object.",
    "Walking_With": "A photo of a person walking with an object."
}

class CLIPHOIPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.labels = list(CLIP_HOI_PROMPTS.keys())
        self.prompts = list(CLIP_HOI_PROMPTS.values())

    def load(self):
        if self.model is None:
            logger.info("Initializing Zero-Shot CLIP HOI (clip-vit-base-patch32)...")
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("CLIP loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
                self.model = None

    def unload(self):
        self.model = None
        self.processor = None

    def _get_person_bbox(self, skeleton: np.ndarray) -> List[float]:
        """Get bounding box from skeleton keypoints."""
        kps = skeleton.reshape(17, 3) if skeleton.ndim == 1 else skeleton
        valid_kps = kps[kps[:, 2] > 0.3][:, :2]
        if len(valid_kps) == 0:
            return None
        x1, y1 = valid_kps.min(axis=0)
        x2, y2 = valid_kps.max(axis=0)
        return [float(x1), float(y1), float(x2), float(y2)]

    def _get_union_bbox(self, b1: List[float], b2: List[float], img_w: int, img_h: int) -> List[int]:
        """Compute union bounding box with padding."""
        x1 = max(0, min(b1[0], b2[0]) - 20)
        y1 = max(0, min(b1[1], b2[1]) - 20)
        x2 = min(img_w, max(b1[2], b2[2]) + 20)
        y2 = min(img_h, max(b1[3], b2[3]) + 20)
        return [int(x1), int(y1), int(x2), int(y2)]

    def predict(self, sampled_frame: np.ndarray, skeletons: Dict[int, np.ndarray], 
                object_detections: List[Dict]) -> List[dict]:
        """
        Run Zero-Shot CLIP on the sampled frame for all person-object pairs.
        skeletons: mapping of track_id -> 17x3 numpy array (for the specific frame).
        object_detections: list of object dicts for the specific frame.
        """
        hoi_events = []
        if self.model is None or sampled_frame is None or not object_detections:
            return hoi_events

        h, w, _ = sampled_frame.shape
        pil_img = Image.fromarray(sampled_frame[..., ::-1]) # BGR to RGB

        for tid, skel in skeletons.items():
            p_bbox = self._get_person_bbox(skel)
            if not p_bbox:
                continue

            for obj in object_detections:
                o_bbox = obj["bbox"]
                
                # Check rough distance to avoid running CLIP on objects 10 meters away
                p_cx, p_cy = (p_bbox[0]+p_bbox[2])/2, (p_bbox[1]+p_bbox[3])/2
                o_cx, o_cy = (o_bbox[0]+o_bbox[2])/2, (o_bbox[1]+o_bbox[3])/2
                dist = np.sqrt((p_cx - o_cx)**2 + (p_cy - o_cy)**2)
                if dist > max(w, h) * 0.4:  # Too far
                    continue

                u_bbox = self._get_union_bbox(p_bbox, o_bbox, w, h)
                if u_bbox[2] <= u_bbox[0] or u_bbox[3] <= u_bbox[1]:
                    continue

                crop = pil_img.crop(u_bbox)
                
                try:
                    inputs = self.processor(text=self.prompts, images=crop, return_tensors="pt", padding=True).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
                    
                    pred_idx = int(np.argmax(probs))
                    label = self.labels[pred_idx]
                    conf = float(probs[pred_idx])

                    if label != "No_Interaction" and conf > 0.2:
                        # Map generalized labels to specific
                        if label == "Holding" and obj["class_name"] in ["tv"]:
                            label = "Looking_At"
                        hoi_events.append({
                            "track_id": tid,
                            "action": label,
                            "object": obj["class_name"],
                            "confidence": conf,
                            "type": "HOI-CLIP"
                        })
                        logger.debug(f"CLIP HOI: Track {tid} {label} {obj['class_name']} ({conf:.2f})")
                except Exception as e:
                    logger.warning(f"CLIP evaluation failed: {e}")

        return hoi_events
