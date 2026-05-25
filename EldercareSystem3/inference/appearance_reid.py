import cv2
import numpy as np
from typing import List, Dict

class AppearanceReID:
    """
    Lightweight ReID using HSV Color Histograms.
    Requires 0 VRAM and is extremely robust for tracking across clips in the same day
    when faces are not visible (e.g. back to camera).
    """
    def __init__(self):
        self.gallery = {} # name -> list of histograms

    def extract_histogram(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        # Crop the person
        crop = image[max(0,y1):min(image.shape[0],y2), max(0,x1):min(image.shape[1],x2)]
        # Ignore extremely small or invalid crops (blurry artifacts)
        if crop.size == 0 or crop.shape[0] < 50 or crop.shape[1] < 20:
            return np.zeros((512,), dtype=np.float32)
        
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Calculate 3D histogram
        # Use 8 bins for H, S, V to keep it compact (8x8x8 = 512 dimensions)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def update_gallery(self, name: str, hist: np.ndarray):
        """Add a known histogram to the gallery."""
        if np.sum(hist) == 0: return
        if name not in self.gallery:
            self.gallery[name] = []
        self.gallery[name].append(hist)
        # Keep only recent 50 samples to avoid memory drift
        if len(self.gallery[name]) > 50:
            self.gallery[name] = self.gallery[name][-50:]

    def identify(self, hist: np.ndarray, threshold: float = 0.75) -> str:
        """Identify an unknown histogram based on the gallery."""
        if np.sum(hist) == 0 or not self.gallery:
            return "Unknown"
        
        best_name = "Unknown"
        best_sim = threshold

        for name, hists in self.gallery.items():
            # Calculate similarity against all known histograms for this person
            sims = [cv2.compareHist(hist, h, cv2.HISTCMP_CORREL) for h in hists]
            if not sims: continue
            
            # Use top 3 best matches to handle varying lighting/poses
            sims = sorted(sims, reverse=True)[:3]
            avg_sim = sum(sims) / len(sims)
            
            if avg_sim > best_sim:
                best_sim = avg_sim
                best_name = name
                
        return best_name
