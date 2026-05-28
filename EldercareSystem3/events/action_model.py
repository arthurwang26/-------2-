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
        self.ckpt_path = cfg.project_root.parent / "shared_weights" / "motionbert_ntu60_xsub.bin"

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

    def predict(self, skeletons: List[Dict]) -> List[Dict]:
        """
        Predict action from a sequence of skeletons for a SINGLE person using a sliding window.
        """
        if self.model is None or not skeletons:
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": 0}]

        total_frames = len(skeletons)
        if total_frames == 0:
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": 0}]
            
        T_target = 243
        window_size = 60
        stride = 30
        
        raw_predictions = []
        prob_history = []
        history_len = 3
        
        for start_idx in range(0, total_frames, stride):
            end_idx = min(start_idx + window_size, total_frames)
            window_skeletons = skeletons[start_idx:end_idx]
            
            frames_kpts = []
            for sk in window_skeletons:
                if isinstance(sk, dict):
                    kpts = sk.get("keypoints")
                else:
                    kpts = sk
                    
                if kpts is not None and kpts.shape == (17, 3):
                    root = (kpts[11, :2] + kpts[12, :2]) / 2.0
                    norm_kpts = kpts.copy()
                    norm_kpts[:, :2] = norm_kpts[:, :2] - root
                    norm_kpts[:, 0] = norm_kpts[:, 0] / 1920.0
                    norm_kpts[:, 1] = norm_kpts[:, 1] / 1080.0
                    frames_kpts.append(norm_kpts)
                    
            if not frames_kpts:
                raw_predictions.append({"action": "Unknown", "confidence": 0.0, "start_frame": start_idx, "end_frame": end_idx})
                if end_idx == total_frames: break
                continue
                
            # Pad to 243 for MotionBERT
            if len(frames_kpts) < T_target:
                pad_len = T_target - len(frames_kpts)
                frames_kpts.extend([frames_kpts[-1]] * pad_len)
            elif len(frames_kpts) > T_target:
                indices = np.linspace(0, len(frames_kpts) - 1, T_target, dtype=int)
                frames_kpts = [frames_kpts[i] for i in indices]
                
            input_data = np.stack(frames_kpts)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                # If the output is (B, num_classes), do not pool.
                # If it's (B, T, V, C), we would pool, but usually it's (B, num_classes)
                if len(logits.shape) > 2:
                    # Fallback if it didn't pool internally
                    logits = logits.view(logits.shape[0], -1, logits.shape[-1]).mean(dim=1)
                    
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            prob_history.append(probs)
            if len(prob_history) > history_len:
                prob_history.pop(0)
                
            avg_probs = np.mean(prob_history, axis=0)
            pred_idx = int(np.argmax(avg_probs))
            conf = float(avg_probs[pred_idx])
            
            action_name = NTU60_CLASSES[pred_idx] if pred_idx < len(NTU60_CLASSES) else "Unknown"
            mapped_action = action_name
            if action_name in ["standing up", "cheer up"]: mapped_action = "Standing"
            elif action_name in ["walking towards each other", "walking apart from each other", "staggering"]: mapped_action = "Walking"
            elif action_name in ["sitting down"]: mapped_action = "Sitting"
            elif action_name in ["falling"]: mapped_action = "Fall Down"
            else: mapped_action = "Unknown"
            
            if conf < 0.2: mapped_action = "Unknown"
            
            raw_predictions.append({"action": mapped_action, "confidence": conf, "start_frame": start_idx, "end_frame": end_idx})
            
            if end_idx == total_frames:
                break
                
        # Merge consecutive identical actions
        merged = []
        for pred in raw_predictions:
            if not merged:
                merged.append(pred)
            else:
                last = merged[-1]
                if last["action"] == pred["action"]:
                    last["end_frame"] = pred["end_frame"]
                    last["confidence"] = (last["confidence"] + pred["confidence"]) / 2.0
                else:
                    merged.append(pred)
                    
        return merged

if __name__ == "__main__":
    with ModelGuard("MotionBERT_Action"):
        model = MotionBERTActionModel()
        model.load()
        dummy_skels = [{"keypoints": np.random.rand(17, 3)} for _ in range(50)]
        res = model.predict(dummy_skels)
        logger.info(f"Test Prediction: {res}")
