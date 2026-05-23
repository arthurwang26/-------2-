import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from utils.reset_memory import ModelGuard

logger = get_logger("cross_day_gru")

TREND_CLASSES = ["Improving", "Stable", "Declining"]

# Event encoding dimensions
ACTION_VOCAB = ["Standing", "Walking", "Sitting", "Talking", "Arguing",
                "Reading", "Looking", "Leaving", "Unknown"]
EMOTION_VOCAB = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness',
                 'Neutral', 'Sadness', 'Surprise']


class CrossDayBiGRU(nn.Module):
    """BiGRU + Attention for cross-day behavior reasoning."""
    def __init__(self, input_dim=32, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(gru_out * attn_weights, dim=1)
        risk = self.risk_head(context)
        trend = self.trend_head(context)
        return risk, trend, attn_weights


def encode_clip_events(events: List[Dict], anomaly_score: float = 0.0) -> np.ndarray:
    """
    Encode events from a single clip into a 32-dim feature vector.
    Uses real event data (actions, emotions, HOI, anomaly scores).
    """
    feat = np.zeros(32, dtype=np.float32)

    # Dimensions 0-8: action one-hot (accumulated across persons)
    # Dimensions 9-16: emotion distribution (accumulated)
    # Dimension 17: number of persons
    # Dimension 18: number of HOI events
    # Dimension 19: anomaly score
    # Dimensions 20-26: HOI type flags
    # Dimensions 27-31: reserved

    person_count = set()
    for ev in events:
        person_count.add(ev.get("person", ""))

        if ev.get("type") == "Action":
            action = ev.get("action", "Unknown")
            if action in ACTION_VOCAB:
                feat[ACTION_VOCAB.index(action)] += 1.0

        elif ev.get("type") == "Emotion":
            emotion = ev.get("emotion", "Neutral")
            if emotion in EMOTION_VOCAB:
                feat[9 + EMOTION_VOCAB.index(emotion)] += 1.0

        elif ev.get("type") == "HOI":
            feat[18] += 1.0

    feat[17] = len(person_count)
    feat[19] = anomaly_score

    # Normalize
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm

    return feat


class TemporalReasoningModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load(self):
        if self.model is None:
            logger.info("Initializing Cross-Day BiGRU...")
            self.model = CrossDayBiGRU().to(self.device)
            self.model.eval()
            logger.info("BiGRU loaded.")

    def unload(self):
        self.model = None

    def reason(self, events_by_clip: Dict[str, List[Dict]],
               anomaly_by_clip: Dict[str, float]) -> Dict[str, Any]:
        """
        Process event sequences across all clips.
        Uses REAL event features (from perception + event layers).
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        # Encode each clip's events into a feature vector
        clip_names = sorted(events_by_clip.keys())
        logger.info(f"Running Cross-Day Reasoning over {len(clip_names)} clips...")

        feature_seq = []
        for clip in clip_names:
            clip_events = events_by_clip.get(clip, [])
            anomaly = anomaly_by_clip.get(clip, 0.0)
            feat = encode_clip_events(clip_events, anomaly)
            feature_seq.append(feat)

        if not feature_seq:
            return {"risk_score": 0.0, "behavior_trend": "Stable"}

        # (1, num_clips, 32)
        seq_tensor = torch.tensor(np.array(feature_seq)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            risk, trend_logits, attn_weights = self.model(seq_tensor)
            risk_score = risk.item()
            probs = F.softmax(trend_logits, dim=1)[0]
            trend_idx = torch.argmax(probs).item()
            behavior_trend = TREND_CLASSES[trend_idx]

        # Log attention distribution
        attn = attn_weights[0, :, 0].cpu().numpy()
        logger.debug(f"Risk={risk_score:.4f}, Trend={behavior_trend}")
        for i, clip in enumerate(clip_names):
            logger.debug(f"  {clip}: attention={attn[i]:.4f}")

        return {
            "risk_score": risk_score,
            "behavior_trend": behavior_trend,
            "attention_weights": {clip: float(attn[i]) for i, clip in enumerate(clip_names)}
        }


if __name__ == "__main__":
    dummy_events = {
        "day1_clip01": [{"type": "Action", "action": "Walking", "person": "A"}],
        "day1_clip02": [{"type": "Action", "action": "Arguing", "person": "A"},
                        {"type": "Emotion", "emotion": "Anger", "person": "A"}],
    }
    dummy_anomaly = {"day1_clip01": 0.1, "day1_clip02": 0.5}

    with ModelGuard("BiGRU"):
        model = TemporalReasoningModel()
        model.load()
        res = model.reason(dummy_events, dummy_anomaly)
        logger.info(f"Results: {res}")
