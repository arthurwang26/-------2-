"""
MotionBERT Action Recognition Model using DSTformer.
"""
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
# Add MotionBERT to path so it can find 'lib.model'
sys.path.append(str(Path(__file__).resolve().parent.parent / "libs" / "MotionBERT"))

from config import cfg
from utils.logger import get_logger
from utils.reset_memory import ModelGuard
from libs.MotionBERT.lib.model.DSTformer import DSTformer

logger = get_logger("action_motionbert")

NTU60_CLASSES = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", 
    "pickup", "throw", "sitting down", "standing up", "clapping", "reading", 
    "writing", "tear up paper", "wear jacket", "take off jacket", "wear a shoe", 
    "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", 
    "take off a hat/cap", "cheer up", "hand waving", "kicking something", 
    "put/take something from pocket", "hopping", "jump up", "make a phone call", 
    "playing with phone/tablet", "typing on a keyboard", "pointing to something", 
    "taking a selfie", "check time (from watch)", "rub two hands together", 
    "nod head/bow", "shake head", "wipe face", "salute", "put the palms together", 
    "cross hands in front", "sneeze/cough", "staggering", "falling", 
    "touch head (headache)", "touch chest (stomachache/heart pain)", 
    "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition", 
    "use a fan", "punching/slapping other person", "kicking other person", 
    "pushing other person", "pat on back of other person", 
    "point finger at the other person", "hugging other person", 
    "giving something to other person", "touch other person's pocket", 
    "handshaking", "walking towards each other", "walking apart from each other"
]


class MotionBERTActionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # We need the weights file
        self.ckpt_path = cfg.project_root / "weights" / "motionbert_ntu60_xsub.bin"

    def load(self):
        if self.model is not None:
            return
            
        if not self.ckpt_path.exists():
            logger.error(f"MotionBERT weights not found at {self.ckpt_path}")
            return
            
        logger.info(f"Loading MotionBERT (DSTformer) from {self.ckpt_path}...")
        
        # DSTformer configuration for NTU60
        self.model = DSTformer(
            dim_in=3, 
            dim_out=60, # NTU-60 classes
            dim_feat=256, 
            dim_rep=512, 
            depth=5, 
            num_heads=8, 
            mlp_ratio=2, 
            norm_layer=nn.LayerNorm, 
            maxlen=243, 
            num_joints=17
        ).to(self.device)
        
        # Load weights
        state_dict = torch.load(self.ckpt_path, map_location=self.device)
        # Check if the checkpoint has 'model' key (standard MotionBERT format)
        if 'model_pos' in state_dict:
            # Action recognition weights usually have 'model' or 'model_pos'
            self.model.load_state_dict(state_dict['model_pos'], strict=False)
        elif 'model' in state_dict:
            self.model.load_state_dict(state_dict['model'], strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
            
        self.model.eval()
        logger.info("MotionBERT loaded successfully.")

    def unload(self):
        self.model = None

    def predict(self, skeletons: List[Dict]) -> str:
        """
        Predict action from a sequence of skeletons for a SINGLE person.
        skeletons: list of dicts over frames, e.g., [{"keypoints": np.array(17,3)}]
        """
        if self.model is None or not skeletons:
            return "Unknown"

        # 1. Extract and pad sequence to 243 frames (MotionBERT maxlen)
        T_target = 243
        seq_len = len(skeletons)
        
        # Input shape expected by DSTformer: (B, T, V, C) or it processes (B, T, V, C) internally
        # Actually DSTformer for action recognition usually takes (N, 3, T, V, M) in the wrapper
        # But DSTformer itself takes (B, T, V, C) if we bypass the graph wrappers.
        # Let's prepare a (1, 243, 17, 3) tensor
        frames_kpts = []
        for sk in skeletons:
            if isinstance(sk, dict):
                kpts = sk.get("keypoints")
            else:
                kpts = sk # It might be a numpy array directly
                
            if kpts is not None and kpts.shape == (17, 3):
                # We must ZERO-CENTER the skeleton to fix the "Falling" bug!
                # Keypoint 0 (Nose) or Center of gravity (Hip)
                # Let's use the average of hips (idx 11, 12 in COCO) as root
                root = (kpts[11, :2] + kpts[12, :2]) / 2.0
                
                # Copy so we don't modify the original
                norm_kpts = kpts.copy()
                norm_kpts[:, :2] = norm_kpts[:, :2] - root
                
                # Scale by standard resolution to mimic NTU preprocessing
                # Assuming video is around 1920x1080
                norm_kpts[:, 0] = norm_kpts[:, 0] / 1920.0
                norm_kpts[:, 1] = norm_kpts[:, 1] / 1080.0
                
                frames_kpts.append(norm_kpts)
                
        if not frames_kpts:
            return "Unknown"
            
        # Pad or truncate to T_target
        if len(frames_kpts) < T_target:
            # Pad by repeating the last frame
            pad_len = T_target - len(frames_kpts)
            frames_kpts.extend([frames_kpts[-1]] * pad_len)
        elif len(frames_kpts) > T_target:
            # Uniform sampling
            indices = np.linspace(0, len(frames_kpts) - 1, T_target, dtype=int)
            frames_kpts = [frames_kpts[i] for i in indices]
            
        # (243, 17, 3)
        input_data = np.stack(frames_kpts)
        # DSTformer expects (B, T, V, C) -> (1, 243, 17, 3)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # DSTformer forward returns: (B, F, J, num_classes)
            logits = self.model(input_tensor)
            # Pool over frames (F) and joints (J)
            logits = logits.mean(dim=(1, 2))  # -> (B, num_classes)
            
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_idx_tensor = torch.max(probs, dim=1)
            max_prob = max_prob.item()
            pred_idx = pred_idx_tensor.item()
            
        if max_prob < 0.2:
            return "Unknown"
            
        if 0 <= pred_idx < len(NTU60_CLASSES):
            raw_action = NTU60_CLASSES[pred_idx]
            if raw_action == "falling":
                return "躺著"
            elif raw_action in ["make a phone call", "playing with phone/tablet", "point finger at the other person"]:
                return "說話"
            elif raw_action == "sitting down":
                return "坐著"
            elif raw_action == "standing up":
                return "站著"
            elif raw_action in ["walking towards each other", "walking apart from each other", "staggering"]:
                return "走路"
            else:
                return "Unknown"
        return "Unknown"

if __name__ == "__main__":
    with ModelGuard("MotionBERT_Action"):
        model = MotionBERTActionModel()
        model.load()
        dummy_skels = [{"keypoints": np.random.rand(17, 3)} for _ in range(50)]
        res = model.predict(dummy_skels)
        logger.info(f"Test Prediction: {res}")
