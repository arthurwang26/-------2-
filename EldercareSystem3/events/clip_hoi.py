"""
Zero-Shot HOI Detection using OpenAI CLIP (ViT-B/32).
Dynamically generates prompts based on YOLO-detected objects.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import cfg
from utils.logger import get_logger
from utils.reset_memory import ModelGuard

logger = get_logger("clip_hoi")

# Base action templates per object category
OBJECT_ACTION_TEMPLATES = {
    "cup": [
        ("A person drinking from a cup", "Drinking_From", "Cup"),
        ("A person holding a cup", "Holding", "Cup"),
    ],
    "bottle": [
        ("A person drinking from a bottle", "Drinking_From", "Bottle"),
        ("A person holding a bottle", "Holding", "Bottle"),
    ],
    "book": [
        ("A person reading a book", "Reading", "Book"),
        ("A person holding a book", "Holding", "Book"),
    ],
    "tv": [
        ("A person watching television", "Watching", "TV"),
    ],
    "remote": [
        ("A person using a remote control", "Using", "Remote"),
        ("A person holding a remote control", "Holding", "Remote"),
    ],
    "cell phone": [
        ("A person using a phone", "Using", "Phone"),
        ("A person looking at a phone", "Looking_At", "Phone"),
    ],
    "chair": [
        ("A person sitting on a chair", "Sitting_On", "Chair"),
    ],
    "couch": [
        ("A person sitting on a couch", "Sitting_On", "Couch"),
    ],
    "dining table": [
        ("A person sitting at a dining table", "Sitting_At", "Table"),
        ("A person eating at a table", "Eating_At", "Table"),
    ],
    "vase": [
        ("A person looking at a vase", "Looking_At", "Vase"),
    ],
}

# Baseline prompts (always included as negative anchors)
BASELINE_PROMPTS = [
    "A person standing idle",
    "A person walking",
]


class CLIPHOIPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocess = None
        self._clip_module = None

    def load(self):
        if self.model is not None:
            return
            
        import clip
        self._clip_module = clip
        logger.info(f"Loading CLIP (ViT-B/32) on {self.device} for Zero-Shot HOI...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        logger.info("CLIP loaded and text prompts encoded.")

    def unload(self):
        self.model = None
        self.preprocess = None
        self._clip_module = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_prompts_from_objects(self, detected_objects: set):
        """Dynamically build CLIP text prompts based on YOLO-detected objects."""
        prompts = list(BASELINE_PROMPTS)
        mapping = {}
        for p in BASELINE_PROMPTS:
            mapping[p] = (None, None)
        
        for obj_class in detected_objects:
            templates = OBJECT_ACTION_TEMPLATES.get(obj_class, [])
            for prompt_text, action, obj_name in templates:
                if prompt_text not in mapping:
                    prompts.append(prompt_text)
                    mapping[prompt_text] = (action, obj_name)
        
        return prompts, mapping

    def predict(self, frames: List[np.ndarray], tracks, objects_per_frame: List[List[Dict]] = None, sample_rate: int = 30) -> List[Dict]:
        """
        Dynamic HOI prediction using YOLO-detected objects to build CLIP prompts.
        """
        if self.model is None or not frames or not tracks:
            return []
        
        # Collect all unique detected object classes across frames
        detected_objects = set()
        if objects_per_frame:
            for frame_objs in objects_per_frame:
                for obj in frame_objs:
                    detected_objects.add(obj.get("class_name", ""))
        
        # Build dynamic prompts
        prompts, mapping = self._build_prompts_from_objects(detected_objects)
        logger.info(f"Dynamic HOI prompts ({len(prompts)}): {[p for p in prompts if mapping.get(p, (None,))[0] is not None]}")
        
        # Encode text features
        clip = self._clip_module
        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        hoi_events = []
        unique_events = set()
        
        frame_indices = list(range(0, len(frames), sample_rate))
        if len(frames) - 1 not in frame_indices:
            frame_indices.append(len(frames) - 1)
            
        for f_idx in frame_indices:
            frame = frames[f_idx]
            frame_tracks = tracks[f_idx] if f_idx < len(tracks) else []
            
            for det in frame_tracks:
                track_id = det.get("track_id")
                if track_id is None:
                    continue
                    
                bbox = det["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Add context padding
                h, w = frame.shape[:2]
                pad = 50
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                crop_rgb = crop[:, :, ::-1]
                pil_img = Image.fromarray(crop_rgb)
                
                img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.encode_image(img_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    val, idx = similarity[0].topk(1)
                    
                best_idx = idx.item()
                conf = val.item()
                
                if conf > 0.4:
                    prompt = prompts[best_idx]
                    action, obj = mapping.get(prompt, (None, None))
                    if action and obj:
                        event_key = f"{track_id}_{action}_{obj}"
                        if event_key not in unique_events:
                            unique_events.add(event_key)
                            hoi_events.append({
                                "type": "HOI-CLIP",
                                "action": action,
                                "object": obj,
                                "track_id": track_id,
                                "frame_idx": f_idx,
                                "start_frame": f_idx,
                                "end_frame": f_idx,
                                "confidence": round(float(conf), 3)
                            })
                            
        return hoi_events


if __name__ == "__main__":
    with ModelGuard("CLIP"):
        predictor = CLIPHOIPredictor()
        predictor.load()
        logger.info("CLIP HOI Predictor loaded for testing.")
