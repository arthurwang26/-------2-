import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.logger import get_logger

logger = get_logger("blip_caption")

class BLIPCaptioner:
    def __init__(self, model_id="Salesforce/blip-image-captioning-base"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None

    def load(self):
        if self.model is None:
            logger.info(f"Loading BLIP: {self.model_id} on {self.device}")
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            self.model.eval()

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()

    def generate_caption(self, frame: np.ndarray, text_prompt: str = "") -> str:
        """Generate a caption. If text_prompt is provided, it acts as a prefix (e.g. 'a photography of')."""
        if self.model is None:
            self.load()
        
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)

        if text_prompt:
            inputs = self.processor(pil_image, text_prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
        return caption

    def verify_action(self, frame: np.ndarray, action: str) -> dict:
        """
        VQA style verification using BLIP.
        BLIP base is mostly for captioning, but we can do conditional generation
        by giving a prompt like 'Question: is the person drinking? Answer:'
        """
        if self.model is None:
            self.load()

        rgb_frame = frame[:, :, ::-1]
        pil_image = Image.fromarray(rgb_frame)
        
        prompt = f"Question: is the person {action}? Answer:"
        inputs = self.processor(pil_image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=10)
            answer = self.processor.decode(out[0], skip_special_tokens=True).strip().lower()
            
        # If it answers 'yes', confidence is higher
        is_yes = 'yes' in answer
        return {
            "verified": is_yes,
            "blip_answer": answer,
            "confidence": 0.8 if is_yes else 0.2
        }
