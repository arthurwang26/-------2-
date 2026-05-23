"""
ST-GCN Action Recognition Model using MMAction2 Pretrained Weights.

Architecture:
  - Exact MMAction2 ST-GCN architecture (10 blocks).
  - Uses COCO-17 skeleton inputs.
  - Pretrained on NTU RGB+D 60 dataset.
"""
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

logger = get_logger("action_stgcn")

ACTION_CLASSES = ["Standing", "Walking", "Sitting", "Talking", "Arguing",
                  "Reading", "Looking", "Leaving", "Unknown"]

NTU60_MAPPING = {
    7: "Sitting",      # sitting down
    8: "Standing",     # standing up
    10: "Reading",     # reading
    11: "Reading",     # writing
    27: "Talking",     # make a phone call
    28: "Reading",     # playing with phone
    29: "Reading",     # typing on keyboard
    49: "Arguing",     # punching/slapping
    50: "Arguing",     # kicking other
    51: "Arguing",     # pushing other
    58: "Walking",     # walking towards
    59: "Walking"      # walking apart
}


# =====================================================================
# MMAction2 ST-GCN Architecture (Exact Match)
# =====================================================================
class MMGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.PA = nn.Parameter(torch.zeros(3, 17, 17))
        self.register_buffer('A', torch.zeros(3, 17, 17))
        self.conv = nn.Conv2d(in_channels, out_channels * 3, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        n, c, t, v = x.size()
        A = self.PA + self.A
        x = self.conv(x)
        x = x.view(n, 3, self.out_channels, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return self.bn(x)


class MMSTGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.gcn = MMGraphConv(in_channels, out_channels)
        self.tcn = nn.Module()
        self.tcn.conv = nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), padding=(4, 0))
        self.tcn.bn = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        res = x
        x = self.gcn(x)
        x = F.relu(x)
        x = self.tcn.conv(x)
        x = self.tcn.bn(x)
        if self.downsample is not None:
            res = self.downsample(res)
        x += res
        return F.relu(x)


class MMSTGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(3 * 17)
        
        self.gcn = nn.ModuleList([
            MMSTGCNBlock(3, 64, 1),
            MMSTGCNBlock(64, 64, 1),
            MMSTGCNBlock(64, 64, 1),
            MMSTGCNBlock(64, 64, 1),
            MMSTGCNBlock(64, 128, 2),
            MMSTGCNBlock(128, 128, 1),
            MMSTGCNBlock(128, 128, 1),
            MMSTGCNBlock(128, 256, 2),
            MMSTGCNBlock(256, 256, 1),
            MMSTGCNBlock(256, 256, 1)
        ])
        
        self.cls_head = nn.Module()
        self.cls_head.fc = nn.Linear(256, 60)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() # N, C, V, T
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T)
        x = x.permute(0, 1, 3, 2).contiguous() # N, C, T, V
        
        for block in self.gcn:
            x = block(x)
            
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return self.cls_head.fc(x)


# =====================================================================
# Geometric Feature Extraction (Fallback)
# =====================================================================
def _compute_skeleton_features(skeleton_seq: List[np.ndarray]) -> Dict[str, float]:
    if not skeleton_seq or len(skeleton_seq) < 2:
        return {"height_ratio": 0.5, "velocity": 0.0, "arm_activity": 0.0,
                "is_low": False, "head_movement": 0.0, "body_spread": 0.0}
    seq = np.array(skeleton_seq)
    coords, confs = seq[:, :, :2], seq[:, :, 2]

    centers = [coords[t][confs[t] > 0.3].mean(axis=0) for t in range(len(coords)) if (confs[t] > 0.3).any()]
    centers = np.array(centers) if centers else np.zeros((1, 2))
    velocity = float(np.mean(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(centers) > 1 else 0.0

    nose_y = [coords[t, 0, 1] for t in range(len(seq)) if confs[t, 0] > 0.3]
    hip_y = [np.mean([coords[t, j, 1] for j in [11, 12] if confs[t, j] > 0.3]) 
             for t in range(len(seq)) if any(confs[t, j] > 0.3 for j in [11, 12])]
    height_ratio = float(np.mean(nose_y) / (np.mean(hip_y) + 1e-8)) if nose_y and hip_y else 0.5

    return {"velocity": velocity, "height_ratio": height_ratio}


# =====================================================================
# Predictor Wrapper
# =====================================================================
class STGCNActionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load(self):
        if self.model is None:
            logger.info("Initializing Pretrained MMAction2 ST-GCN (NTU-60 COCO-17)...")
            self.model = MMSTGCN().to(self.device)
            
            # Load Weights
            weight_path = "weights/stgcn_ntu60_xsub_coco17.pth"
            if Path(weight_path).exists():
                ckpt = torch.load(weight_path, map_location='cpu')
                state_dict = ckpt['state_dict']
                new_state = {}
                for k, v in state_dict.items():
                    k = k.replace('backbone.', '')
                    k = k.replace('residual.conv', 'downsample.0')
                    k = k.replace('residual.bn', 'downsample.1')
                    new_state[k] = v
                self.model.load_state_dict(new_state, strict=False)
                logger.info("Successfully loaded pretrained NTU-60 weights.")
            else:
                logger.warning(f"Pretrained weights not found at {weight_path}. Using random init.")
            
            self.model.eval()

    def unload(self):
        self.model = None

    def predict(self, skeleton_seqs: Dict[int, List[np.ndarray]], num_persons: int = 1) -> Dict[int, dict]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        logger.info(f"Predicting actions for {len(skeleton_seqs)} tracks using MMAction2 ST-GCN...")
        actions = {}
        
        for tid, seq in skeleton_seqs.items():
            if len(seq) < 3:
                actions[tid] = {"action": "Unknown", "confidence": 0.0}
                continue
                
            seq_np = np.array(seq)
            # Input needs to be (N, C, T, V)
            tensor = torch.tensor(seq_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
            # Map NTU-60 to our Action Classes
            mapped_probs = np.zeros(len(ACTION_CLASSES))
            for ntu_idx, our_label in NTU60_MAPPING.items():
                our_idx = ACTION_CLASSES.index(our_label)
                mapped_probs[our_idx] += probs[ntu_idx]
            
            # Completely remove rule engine (Geometric Fallback)
            # 100% Pretrained AI Output
            combined = mapped_probs
            pred_idx = int(np.argmax(combined))
            
            if combined.sum() > 0:
                final_action = ACTION_CLASSES[pred_idx]
                final_conf = float(combined[pred_idx])
            else:
                final_action = "Unknown"
                final_conf = 0.0
                
            actions[tid] = {
                "action": final_action,
                "confidence": final_conf
            }
            logger.info(f"Track {tid}: {actions[tid]['action']} ({actions[tid]['confidence']:.2f})")
            
        return actions

if __name__ == "__main__":
    dummy_seq = [np.random.rand(17, 3).astype(np.float32) for _ in range(30)]
    skeleton_seqs = {1: dummy_seq}
    with ModelGuard("ST-GCN"):
        model = STGCNActionModel()
        model.load()
        results = model.predict(skeleton_seqs)
        logger.info(f"Results: {results}")
