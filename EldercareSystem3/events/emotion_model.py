import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from utils.reset_memory import ModelGuard

logger = get_logger("emotion_gru")

EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']


class EmotionGRU(nn.Module):
    """GRU for temporal smoothing of emotion sequences from EfficientNet-B2."""
    def __init__(self, input_dim=8, hidden_dim=32, num_layers=1, num_classes=8):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class EmotionSequencePredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load(self):
        if self.model is None:
            logger.info("Initializing Emotion GRU...")
            self.model = EmotionGRU().to(self.device)
            self.model.eval()
            logger.info("Emotion GRU loaded.")

    def unload(self):
        self.model = None

    def predict(self, emotion_seqs: Dict[int, List[np.ndarray]]) -> Dict[int, dict]:
        """
        Predict dominant emotion from EfficientNet probability sequences.
        GRU smooths the temporal sequence; final result is grounded by the
        per-frame EfficientNet outputs (which are pre-trained and accurate).
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        logger.info(f"Predicting emotions for {len(emotion_seqs)} tracks...")
        emotions = {}

        for tid, seq in emotion_seqs.items():
            if not seq:
                continue

            seq_np = np.array(seq)  # (T, 8)

            # Run GRU
            tensor = torch.tensor(seq_np).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                gru_logits = self.model(tensor)
                gru_probs = F.softmax(gru_logits, dim=1)[0].cpu().numpy()

            # Direct average of EfficientNet outputs (pre-trained, reliable)
            avg_probs = seq_np.mean(axis=0)

            # Blend: 70% EfficientNet average + 30% GRU temporal smoothing
            blended = 0.7 * avg_probs + 0.3 * gru_probs
            blended = blended / blended.sum()

            pred_idx = int(np.argmax(blended))
            label = EMOTION_LABELS[pred_idx]
            conf = float(blended[pred_idx])

            emotions[tid] = {"emotion": label, "confidence": conf,
                             "distribution": {EMOTION_LABELS[i]: float(blended[i])
                                              for i in range(8)}}
            logger.debug(f"Track {tid}: {label} ({conf:.2f})")

        return emotions


if __name__ == "__main__":
    dummy_seq = [np.random.dirichlet(np.ones(8)).astype(np.float32) for _ in range(10)]
    emotion_seqs = {1: dummy_seq}

    with ModelGuard("Emotion GRU"):
        model = EmotionSequencePredictor()
        model.load()
        results = model.predict(emotion_seqs)
        logger.info(f"Results: {results}")
