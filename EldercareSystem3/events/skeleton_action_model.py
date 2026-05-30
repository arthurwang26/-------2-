import torch
import numpy as np
from events.custom_action_model import TransformerLSTMActionModel
from utils.logger import get_logger

logger = get_logger("skeleton_action")

class CustomSkeletonActionModel:
    def __init__(self, weights_path="events/weights/custom_action_model.pth", seq_len=60, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        
        logger.info(f"Loading custom Transformer+LSTM Action Model from {weights_path}...")
        self.model = TransformerLSTMActionModel(input_dim=102, hidden_dim=256, num_classes=7, dropout=0.5)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Kaggle dataset labels mapping
        self.idx_to_class = {
            0: "Sitting",
            1: "Falling",
            2: "Walking",
            3: "Standing",
            4: "Standing",
            5: "Falling",
            6: "Sitting"
        }
        logger.info("Custom Action Model loaded successfully.")

    def predict(self, skeleton_frames, frame_w=1920, frame_h=1080):
        features = []
        for skel in skeleton_frames:
            if skel is None:
                continue
            
            kpts = np.array(skel)
            valid = kpts[:, 2] > 0.1
            if not np.any(valid):
                continue
            
            flat_kpts = []
            
            # Step 1: Normalize to [0, 1] range to match YOLOv7 training dataset scaling
            for kp in kpts:
                x_norm = kp[0] / frame_w if kp[2] > 0 else 0.0
                y_norm = kp[1] / frame_h if kp[2] > 0 else 0.0
                conf = kp[2]
                flat_kpts.extend([x_norm, y_norm, conf])
            features.append(flat_kpts)
            
        if not features:
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": len(skeleton_frames)}]
            
        features = np.array(features, dtype=np.float32)
        total_frames = len(features)
        
        # Step 2: Translation Invariance (Subtract nose) EXACLTY as done in train_custom_ai.py
        for f_idx in range(len(features)):
            nose_x = features[f_idx, 0]
            nose_y = features[f_idx, 1]
            if nose_x != 0.0 and nose_y != 0.0:
                for i in range(0, 51, 3):
                    if features[f_idx, i] != 0.0:
                        features[f_idx, i] -= nose_x
                    if features[f_idx, i+1] != 0.0:
                        features[f_idx, i+1] -= nose_y
        
        # Step 3: Compute velocity (Temporal Dynamics)
        velocity = np.zeros_like(features)
        velocity[1:] = features[1:] - features[:-1]
            
        # Combine position and velocity (51 + 51 = 102 features)
        full_features = np.concatenate([features, velocity], axis=1)
        
        window_size = self.seq_len
        stride = 5
        
        if total_frames == 0 or not np.any(full_features):
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": 0}]

        raw_predictions = []
        prob_history = []
        history_len = 5
        
        for start_idx in range(0, total_frames, stride):
            end_idx = min(start_idx + window_size, total_frames)
            window_features = full_features[start_idx:end_idx].copy()
            
            seq = np.zeros((window_size, 102), dtype=np.float32)
            actual_len = len(window_features)
            
            if actual_len < window_size and actual_len > 0:
                pad_len = window_size - actual_len
                last_frame = window_features[-1:]
                padded = np.vstack([window_features, np.repeat(last_frame, pad_len, axis=0)])
                seq = padded
            else:
                seq[:actual_len, :] = window_features
            
            input_tensor = torch.tensor(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
            # Temporal smoothing
            prob_history.append(probabilities)
            if len(prob_history) > history_len:
                prob_history.pop(0)
                
            smoothed_probs = np.mean(prob_history, axis=0)
            predicted_idx = int(np.argmax(smoothed_probs))
            conf = float(smoothed_probs[predicted_idx])
            action_name = self.idx_to_class.get(predicted_idx, "Unknown")
            
            raw_predictions.append({"action": action_name, "confidence": conf, "start_frame": start_idx, "end_frame": end_idx})
            
            if (end_idx - start_idx) < 15:
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
                    boundary = pred["start_frame"] + (pred["end_frame"] - pred["start_frame"]) // 2
                    last["end_frame"] = boundary
                    merged.append({
                        "action": pred["action"], 
                        "start_frame": boundary, 
                        "end_frame": pred["end_frame"], 
                        "confidence": pred["confidence"]
                    })
        
        if merged:
            merged[-1]["end_frame"] = total_frames
                    
        return merged
