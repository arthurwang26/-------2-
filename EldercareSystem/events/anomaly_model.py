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

logger = get_logger("anomaly_ae")


class SkeletonAutoencoder(nn.Module):
    """1D Conv Autoencoder for skeleton anomaly detection (self-supervised)."""
    def __init__(self, seq_len=30, feature_dim=51):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AnomalyDetector:
    """
    Self-supervised anomaly detection.
    Trains the autoencoder on "normal" skeleton data, then measures
    reconstruction error as anomaly score.
    """
    def __init__(self, seq_len=30):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.model = None
        self.baseline_error = None  # Learned from normal data

    def load(self):
        if self.model is None:
            logger.info("Initializing Skeleton Autoencoder...")
            self.model = SkeletonAutoencoder(seq_len=self.seq_len).to(self.device)
            self.model.eval()
            logger.info("Anomaly AE loaded.")

    def unload(self):
        self.model = None
        self.baseline_error = None

    def _prepare_sequence(self, seq: List[np.ndarray]) -> torch.Tensor:
        """Prepare skeleton sequence as tensor, pad/truncate to seq_len."""
        seq_np = np.array([kp.flatten() for kp in seq])  # (T, 51)
        if len(seq_np) > self.seq_len:
            seq_np = seq_np[:self.seq_len]
        elif len(seq_np) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq_np), 51), dtype=np.float32)
            seq_np = np.vstack([seq_np, pad])
        # (1, 51, seq_len)
        return torch.tensor(seq_np).permute(1, 0).unsqueeze(0).float().to(self.device)

    def train_baseline(self, normal_skeleton_seqs: Dict[int, List[np.ndarray]], epochs=20):
        """
        Self-supervised training on normal behavior skeleton data.
        This is the KEY self-built component: learning what "normal" looks like.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        logger.info(f"Training anomaly baseline on {len(normal_skeleton_seqs)} tracks, {epochs} epochs...")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        all_tensors = []
        for tid, seq in normal_skeleton_seqs.items():
            if len(seq) >= 5:
                all_tensors.append(self._prepare_sequence(seq))

        if not all_tensors:
            logger.warning("No valid sequences for training. Using random baseline.")
            self.baseline_error = 0.1
            self.model.eval()
            return

        data = torch.cat(all_tensors, dim=0)

        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.model(data)
            loss = F.mse_loss(reconstructed, data)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                logger.debug(f"  Epoch {epoch}: loss={loss.item():.6f}")

        # Record baseline error
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data)
            self.baseline_error = F.mse_loss(reconstructed, data).item()

        logger.info(f"Baseline training complete. Normal error: {self.baseline_error:.6f}")

    def predict(self, skeleton_seqs: Dict[int, List[np.ndarray]]) -> Dict[int, float]:
        """Calculate anomaly scores (reconstruction error relative to baseline)."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        logger.info(f"Calculating anomaly scores for {len(skeleton_seqs)} tracks...")
        self.model.eval()
        scores = {}

        for tid, seq in skeleton_seqs.items():
            if len(seq) < 3:
                scores[tid] = 0.0
                continue

            tensor = self._prepare_sequence(seq)
            with torch.no_grad():
                reconstructed = self.model(tensor)
                mse = F.mse_loss(reconstructed, tensor).item()

            # Normalize relative to baseline
            if self.baseline_error and self.baseline_error > 0:
                score = mse / self.baseline_error
            else:
                score = mse

            # Clamp to 0-1 range
            score = min(1.0, max(0.0, score / 5.0))
            scores[tid] = score
            logger.debug(f"Track {tid}: anomaly={score:.4f} (raw_mse={mse:.6f})")

        return scores


if __name__ == "__main__":
    # Simulate: train on normal, test on normal + anomalous
    normal_seq = [np.random.rand(17, 3).astype(np.float32) * 0.5 for _ in range(30)]
    anomalous_seq = [np.random.rand(17, 3).astype(np.float32) * 2.0 for _ in range(30)]

    with ModelGuard("Anomaly AE"):
        ae = AnomalyDetector()
        ae.load()
        ae.train_baseline({1: normal_seq})
        scores = ae.predict({1: normal_seq, 2: anomalous_seq})
        logger.info(f"Scores: {scores}")
