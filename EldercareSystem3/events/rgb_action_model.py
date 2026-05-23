import torch
import numpy as np
import cv2
from typing import List
import logging
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

logger = logging.getLogger(__name__)

class RGBActionModel:
    def __init__(self, device='cpu', model_name='MCG-NJU/videomae-base-finetuned-kinetics'):
        self.device = device
        self.model_name = model_name
        self.processor = None
        self.model = None

    def load(self):
        logger.info(f"Loading RGB Action Model ({self.model_name}) on {self.device}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("RGB Action Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RGB Action Model: {e}")
            self.model = None

    def unload(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, frame_crops: List[np.ndarray]) -> str:
        """
        Predict action from a sequence of RGB crops.
        frame_crops: list of numpy arrays (H, W, 3) in BGR format.
        """
        if not self.model or not frame_crops:
            return "Unknown"

        # VideoMAE usually expects 16 frames
        num_expected_frames = self.model.config.num_frames
        
        # Sample or pad to exact num_expected_frames
        if len(frame_crops) == 0:
            return "Unknown"
        
        # Convert BGR to RGB
        rgb_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in frame_crops]
        
        sampled_crops = []
        if len(rgb_crops) >= num_expected_frames:
            # Evenly sample
            indices = np.linspace(0, len(rgb_crops) - 1, num_expected_frames, dtype=int)
            sampled_crops = [rgb_crops[i] for i in indices]
        else:
            # Pad by repeating the last frame
            sampled_crops = rgb_crops + [rgb_crops[-1]] * (num_expected_frames - len(rgb_crops))

        # Processor expects list of numpy arrays or PIL images
        try:
            inputs = self.processor(list(sampled_crops), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                
            return self.model.config.id2label[predicted_class_idx]
        except Exception as e:
            logger.error(f"Error during RGB action prediction: {e}")
            return "Unknown"
