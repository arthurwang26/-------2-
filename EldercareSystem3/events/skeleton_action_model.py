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
            0: "Sit down",
            1: "Lying Down",
            2: "Walking",
            3: "Stand up",
            4: "Standing",
            5: "Fall Down",
            6: "Sitting"
        }
        logger.info("Custom Action Model loaded successfully.")

    def predict(self, skeleton_sequence, w=1920, h=1080):
        # skeleton_sequence: list of ndarray (17,3)
        # Ensure we have 51 features per frame.
        if not skeleton_sequence:
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": 0}]
            
        features = []
        for frame_data in skeleton_sequence:
            frame_features = []
            if isinstance(frame_data, np.ndarray) and frame_data.shape == (17, 3):
                # Copy the data to avoid modifying the original array (needed for visualization)
                norm_data = frame_data.copy()
                norm_data[:, 0] /= w  # normalize X
                norm_data[:, 1] /= h  # normalize Y
                frame_features.extend(norm_data.flatten())
            else:
                # pad with zeros if keypoints missing
                frame_features.extend([0.0] * 51)
            features.append(frame_features)
            
        features = np.array(features, dtype=np.float32)
        
        # --- VELOCITY FEATURES (Temporal Dynamics) ---
        # Compute velocity FIRST on the scale-normalized coordinates
        velocity = np.zeros_like(features)
        velocity[1:] = features[1:] - features[:-1]
        
        window_size = self.seq_len
        stride = 30
        total_frames = len(features)
        
        if total_frames == 0:
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": 0}]

        # Sliding Window
        raw_predictions = []
        prob_history = []
        history_len = 3  # Smooth over 3 sliding windows
        
        for start_idx in range(0, total_frames, stride):
            end_idx = min(start_idx + window_size, total_frames)
            
            # Extract window data
            window_features = features[start_idx:end_idx].copy()
            window_velocity = velocity[start_idx:end_idx].copy()
            
            # --- NORMALIZATION (Window-Level Translation Invariance) ---
            # Center this specific window around its first valid nose coordinate
            base_x, base_y = 0.0, 0.0
            for f_idx in range(len(window_features)):
                if window_features[f_idx, 0] != 0.0 and window_features[f_idx, 1] != 0.0:
                    base_x = window_features[f_idx, 0]
                    base_y = window_features[f_idx, 1]
                    break
                    
            if base_x != 0.0 and base_y != 0.0:
                for f_idx in range(len(window_features)):
                    if window_features[f_idx, 0] != 0.0 or window_features[f_idx, 1] != 0.0:
                        for i in range(0, 51, 3):
                            if window_features[f_idx, i] != 0.0:
                                window_features[f_idx, i] -= base_x
                            if window_features[f_idx, i+1] != 0.0:
                                window_features[f_idx, i+1] -= base_y
                                
            # Combine position and velocity for this window
            window_combined = np.concatenate([window_features, window_velocity], axis=1)
            
            # Pad or truncate to seq_len (60)
            seq = np.zeros((window_size, 102), dtype=np.float32)
            actual_len = len(window_combined)
            seq[:actual_len, :] = window_combined
            
            input_tensor = torch.tensor(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
            # --- TEMPORAL PROBABILITY MOVING AVERAGE ---
            prob_history.append(probabilities)
            if len(prob_history) > history_len:
                prob_history.pop(0)
                
            # Compute smoothed probabilities
            avg_probs = np.mean(prob_history, axis=0)
            
            predicted_idx = int(np.argmax(avg_probs))
            conf = float(avg_probs[predicted_idx])
            
            action_name = self.idx_to_class.get(predicted_idx, "Unknown")
            raw_predictions.append({"action": action_name, "confidence": conf, "start_frame": start_idx, "end_frame": end_idx})
            
            if end_idx == total_frames:
                break
                
        # Merge consecutive identical actions (Pure merging, NO rules)
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
