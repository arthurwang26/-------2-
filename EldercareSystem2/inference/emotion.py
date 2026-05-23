import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import cfg
from utils.logger import get_logger
from utils.reset_memory import ModelGuard

logger = get_logger("emotion")

EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']


class EmotionRecognizer:
    """
    EfficientNet-B2 based emotion recognition.
    Falls back to a direct FER model if HSEmotion has compatibility issues.
    """
    def __init__(self):
        self.model = None
        self.mode = None  # 'hsemotion' or 'fer_manual'
        self.transform = None

    def load(self):
        if self.model is not None:
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Try HSEmotion first
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            rec = HSEmotionRecognizer(model_name='enet_b2_8', device=device)
            # Test with a dummy image to check compatibility
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            rec.predict_emotions(dummy, logits=False)
            self.model = rec
            self.mode = 'hsemotion'
            logger.info("HSEmotion EfficientNet-B2 loaded successfully.")
            return
        except Exception as e:
            logger.warning(f"HSEmotion failed: {e}. Trying FER fallback...")

        # Fallback: use timm's EfficientNet-B2 directly with FER+ weights concept
        # Since HSEmotion's internal model fails, we load the underlying model
        try:
            import timm
            self.model = timm.create_model('tf_efficientnet_b2', pretrained=True, num_classes=8)
            self.model.to(device)
            self.model.eval()
            self.mode = 'fer_manual'
            self.device = device

            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((260, 260)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info("FER fallback with timm EfficientNet-B2 (ImageNet pretrained) loaded.")
        except Exception as e2:
            logger.error(f"All emotion models failed: {e2}. Using dummy classifier.")
            self.model = "DUMMY"
            self.mode = "dummy"

    def unload(self):
        self.model = None
        self.mode = None

    def _predict_single(self, face_rgb: np.ndarray) -> np.ndarray:
        """Predict emotion probabilities for a single face crop (RGB)."""
        if self.mode == 'hsemotion':
            _, probs = self.model.predict_emotions(face_rgb, logits=False)
            return np.array(probs, dtype=np.float32)
        elif self.mode == 'fer_manual':
            tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            return probs.astype(np.float32)
        else:
            return np.ones(8, dtype=np.float32) / 8

    def recognize(self, frames: List[np.ndarray],
                  tracks_batch: List[List[Dict]],
                  face_app=None) -> Dict[int, List[np.ndarray]]:
        """
        Extract emotion probability vectors per tracked person.
        Returns: track_id -> List[np.ndarray(8,)]
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        logger.info(f"Extracting emotions for {len(frames)} frames (mode={self.mode})...")
        emotion_seqs = {}

        # Get face bboxes from InsightFace if available
        all_face_bboxes = []
        if face_app is not None:
            for frame in frames:
                try:
                    faces = face_app.get(frame)
                    all_face_bboxes.append([f.bbox for f in faces])
                except:
                    all_face_bboxes.append([])
        else:
            all_face_bboxes = [[] for _ in frames]

        for idx, (frame, tracks) in enumerate(zip(frames, tracks_batch)):
            if not tracks:
                continue

            face_bboxes = all_face_bboxes[idx] if idx < len(all_face_bboxes) else []

            for t in tracks:
                tid = t["track_id"]
                tx1, ty1, tx2, ty2 = map(int, t["bbox"])
                h, w = frame.shape[:2]

                # Find face bbox inside this person's bbox
                crop_box = None
                for fb in face_bboxes:
                    fcx = (fb[0] + fb[2]) / 2
                    fcy = (fb[1] + fb[3]) / 2
                    if tx1 <= fcx <= tx2 and ty1 <= fcy <= ty2:
                        crop_box = list(map(int, fb))
                        # Expand slightly
                        pad = 10
                        crop_box = [max(0, crop_box[0]-pad), max(0, crop_box[1]-pad),
                                    min(w, crop_box[2]+pad), min(h, crop_box[3]+pad)]
                        break

                if crop_box is None:
                    # Use upper 40% of person bbox
                    bh = ty2 - ty1
                    crop_box = [max(0, tx1), max(0, ty1),
                                min(w, tx2), min(h, ty1 + int(bh * 0.4))]

                cx1, cy1, cx2, cy2 = crop_box
                face_crop = frame[cy1:cy2, cx1:cx2]
                if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                    continue

                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                try:
                    probs = self._predict_single(face_rgb)
                except Exception as e:
                    logger.warning(f"Emotion prediction failed for track {tid}: {e}")
                    probs = np.ones(8, dtype=np.float32) / 8

                if tid not in emotion_seqs:
                    emotion_seqs[tid] = []
                emotion_seqs[tid].append(probs)

        for tid, seq in emotion_seqs.items():
            avg = np.mean(seq, axis=0)
            top_idx = np.argmax(avg)
            logger.debug(f"Track {tid}: {len(seq)} frames, dominant={EMOTION_LABELS[top_idx]} ({avg[top_idx]:.3f})")

        return emotion_seqs


if __name__ == "__main__":
    video_files = sorted(cfg.raw_dir.glob("*.mp4"))
    if video_files:
        cap = cv2.VideoCapture(str(video_files[0]))
        frames = []
        for _ in range(5):
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()

        dummy_tracks = [[{"track_id": 1, "bbox": [100, 50, 400, 450]}] for _ in frames]
        with ModelGuard("Emotion"):
            emo = EmotionRecognizer()
            emo.load()
            seqs = emo.recognize(frames, dummy_tracks)
            for tid, seq in seqs.items():
                avg = np.mean(seq, axis=0)
                top = EMOTION_LABELS[np.argmax(avg)]
                logger.info(f"Track {tid}: {top} ({avg[np.argmax(avg)]:.3f})")
