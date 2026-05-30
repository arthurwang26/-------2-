import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from PIL import Image
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from utils.reset_memory import ModelGuard

logger = get_logger("vlm_caption")


class SmolVLMCaptioner:
    """SmolVLM2-256M for scene description of keyframes."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return
        logger.info("Loading SmolVLM2-256M...")
        try:
            try:
                from transformers import AutoModelForImageTextToText as VLMModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as VLMModel
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
            self.model = VLMModel.from_pretrained(
                "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)
            logger.info("SmolVLM2 loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load SmolVLM2: {e}. Using perception-based captioning.")
            self.model = "FALLBACK"

    def unload(self):
        self.model = None
        self.processor = None

    def caption_keyframes(self, frames: List[np.ndarray],
                          objects_per_frame: List[List[Dict]] = None,
                          identities: Dict[int, str] = None,
                          interval: int = 10) -> List[Dict]:
        """Generate scene descriptions for keyframes."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        logger.info(f"Generating captions (interval={interval})...")
        captions = []

        for i in range(0, len(frames), interval):
            if isinstance(self.model, str):
                # Fallback: build description from perception data
                caption = self._perception_caption(
                    objects_per_frame[i] if objects_per_frame and i < len(objects_per_frame) else [],
                    identities)
                captions.append({"frame_idx": i, "caption": caption})
                continue

            image = Image.fromarray(frames[i][..., ::-1])
            detected_obj_names = set()
            if objects_per_frame and i < len(objects_per_frame):
                detected_obj_names = set([o["class_name"] for o in objects_per_frame[i] if o["class_name"] != "person"])
            
            obj_context = f" The ONLY objects present in this room are: {', '.join(detected_obj_names)}. Do not hallucinate any other objects." if detected_obj_names else " Do not mention any specific objects as none were detected."
            
            prompt = f"Describe this scene briefly. Focus on people, their social interactions (e.g. talking, arguing, ignoring each other, greeting), their specific actions (e.g. walking, sitting, reading), and their emotional state.{obj_context}"

            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
            inputs = inputs.to(self.device)

            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=80)
            decoded = self.processor.batch_decode(gen, skip_special_tokens=True)
            caption = decoded[0].split("Assistant:")[-1].strip()
            captions.append({"frame_idx": i, "caption": caption})
            logger.debug(f"Frame {i}: {caption}")

        return captions

    def _perception_caption(self, objects: List[Dict],
                            identities: Dict[int, str] = None) -> str:
        """Build a caption from detected objects and identities."""
        parts = []
        if identities:
            names = [n for n in identities.values() if n != "Unknown"]
            if names:
                parts.append(f"場景中有{'、'.join(set(names))}")
        if objects:
            obj_names = list(set(o["class_name"] for o in objects))
            if obj_names:
                parts.append(f"偵測到物件：{'、'.join(obj_names)}")
        return "。".join(parts) if parts else "室內場景"

    def verify_hoi_events(self, frames: List[np.ndarray], hoi_events: List[Dict], tracks: List[List[Dict]] = None) -> List[Dict]:
        """Cross-validate HOI events proposed by CLIP with visual grounding.
        Returns a filtered list of events where the VLM answered 'Yes'.
        """
        if self.model is None or isinstance(self.model, str):
            logger.info("VLM not loaded or using fallback. Skipping verification.")
            return hoi_events

        verified_events = []
        for ev in hoi_events:
            if "frame_idx" not in ev:
                verified_events.append(ev)
                continue
            
            f_idx = ev["frame_idx"]
            if f_idx >= len(frames):
                continue
                
            action_desc = ev.get("action", "")
            if not action_desc:
                verified_events.append(ev)
                continue

            frame_copy = frames[f_idx].copy()
            if tracks and "track_id" in ev:
                track_id = ev["track_id"]
                for det in (tracks[f_idx] if f_idx < len(tracks) else []):
                    if det.get("track_id") == track_id:
                        box = det["bbox"]
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        break

            image = Image.fromarray(frame_copy[..., ::-1])
            prompt = f"Is the person inside the RED bounding box doing the following action: '{action_desc}'? Please answer exactly 'Yes' or 'No', followed by a brief 1-sentence reason."

            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
            inputs = inputs.to(self.device)

            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=30)
            decoded = self.processor.batch_decode(gen, skip_special_tokens=True)
            response = decoded[0].split("Assistant:")[-1].strip()
            
            logger.info(f"VLM verification for '{action_desc}': {response}")
            if response.lower().startswith("yes"):
                ev["vlm_reasoning"] = response
                verified_events.append(ev)

        return verified_events

    def resolve_unknown_action(self, frames: List[np.ndarray], start_frame: int, end_frame: int, person_name: str, bbox=None) -> str:
        """Resolve 'Unknown' action using VLM on 4 frames of the sequence with visual grounding."""
        if self.model is None or isinstance(self.model, str):
            return "Unknown"
            
        import numpy as np
        # Sample 4 frames evenly
        f_idxs = np.linspace(start_frame, end_frame, 4, dtype=int)
        f_idxs = [min(max(0, idx), len(frames) - 1) for idx in f_idxs]
        images = []
        for idx in f_idxs:
            frame_copy = frames[idx].copy()
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            images.append(Image.fromarray(frame_copy[..., ::-1]))
        
        prompt = f"Look at these 4 frames spanning from frame {start_frame} to {end_frame}. What is the person inside the RED bounding box doing in this sequence? Choose EXACTLY ONE from this list: 'Standing', 'Sitting', 'Walking', 'Lying Down', 'Stand up', 'Sit down'. Reply with ONLY the chosen word, nothing else."
        
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=10)
        decoded = self.processor.batch_decode(gen, skip_special_tokens=True)
        response = decoded[0].split("Assistant:")[-1].strip()
        
        logger.info(f"VLM resolved Unknown for {person_name} ({start_frame}->{end_frame}) -> {response}")
        
        # Parse response
        response_lower = response.lower()
        if "sit down" in response_lower: return "Sit down"
        if "stand up" in response_lower: return "Stand up"
        if "sit" in response_lower: return "Sitting"
        if "stand" in response_lower: return "Standing"
        if "walk" in response_lower: return "Walking"
        if "ly" in response_lower or "lay" in response_lower: return "Lying Down"
        
        return "Unknown"

if __name__ == "__main__":
    dummy = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(20)]
    with ModelGuard("VLM"):
        vlm = SmolVLMCaptioner()
        vlm.load()
        res = vlm.caption_keyframes(dummy, interval=10)
        logger.info(f"Captions: {res}")
