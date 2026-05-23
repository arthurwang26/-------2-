"""
Zero-Shot HOI Detection using OpenAI CLIP (ViT-B/32).
Performs continuous 30-frame interval sampling.
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

# Prompts for zero-shot classification
HOI_PROMPTS = [
    "A person sitting on a couch",
    "A person drinking from a cup",
    "A person watching TV",
    "A person reading a book",
    "A person using a smartphone",
    "A person falling down",
    "A person eating food",
    "A person cooking in the kitchen",
    "A person standing idle",
    "A person walking"
]

# Map prompt to structured event
HOI_MAPPING = {
    "A person sitting on a couch": ("Sitting_On", "Couch"),
    "A person drinking from a cup": ("Drinking_From", "Cup"),
    "A person watching TV": ("Watching", "TV"),
    "A person reading a book": ("Reading", "Book"),
    "A person using a smartphone": ("Using", "Smartphone"),
    "A person falling down": ("Falling", "Ground"),
    "A person eating food": ("Eating", "Food"),
    "A person cooking in the kitchen": ("Cooking", "Kitchen"),
    "A person standing idle": (None, None),
    "A person walking": (None, None)
}


class CLIPHOIPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocess = None
        self.text_features = None

    def load(self):
        if self.model is not None:
            return
            
        import clip
        logger.info(f"Loading CLIP (ViT-B/32) on {self.device} for Zero-Shot HOI...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        
        # Precompute text embeddings
        text_tokens = clip.tokenize(HOI_PROMPTS).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
        logger.info("CLIP loaded and text prompts encoded.")

    def unload(self):
        self.model = None
        self.preprocess = None
        self.text_features = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, frames: List[np.ndarray], tracks: Dict[int, List[Dict]], sample_rate: int = 30) -> List[Dict]:
        """
        Continuous interval sampling: Runs CLIP on bounding box crops every `sample_rate` frames.
        Returns deduplicated unique events.
        """
        if self.model is None or not frames or not tracks:
            return []
            
        hoi_events = []
        unique_events = set()
        
        # We only sample frames every `sample_rate` frames (default: 30)
        # This solves the issue of missing interactions and removes reliance on rule engines.
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
                    
                # RGB conversion for PIL
                crop_rgb = crop[:, :, ::-1]
                pil_img = Image.fromarray(crop_rgb)
                
                img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.encode_image(img_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                    val, idx = similarity[0].topk(1)
                    
                best_idx = idx.item()
                conf = val.item()
                
                # Threshold for valid HOI (CLIP zero-shot confidence)
                if conf > 0.4:
                    prompt = HOI_PROMPTS[best_idx]
                    action, obj = HOI_MAPPING[prompt]
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
                                "confidence": round(float(conf), 3)
                            })
                            
        return hoi_events

if __name__ == "__main__":
    with ModelGuard("CLIP"):
        predictor = CLIPHOIPredictor()
        predictor.load()
        logger.info("CLIP HOI Predictor loaded for testing.")
