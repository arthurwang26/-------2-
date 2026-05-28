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
                
                # --- FILTER UNCONFIDENT KEYPOINTS ---
                # YOLOv8 pose outputs random x,y for missing keypoints with low conf.
                # This causes massive fake 'velocity' which makes stationary people look like they're walking.
                unconfident_mask = norm_data[:, 2] < 0.3
                norm_data[unconfident_mask, 0] = 0.0
                norm_data[unconfident_mask, 1] = 0.0
                norm_data[unconfident_mask, 2] = 0.0
                
    def predict(self, skeleton_frames):
        # 幾何規則引擎：計算關節角度以打破坐著的幻覺
        def calculate_angle(p1, p2, p3):
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 < 1e-6 or norm_v2 < 1e-6: return 0.0
            cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        avg_hip_angle = 0.0
        avg_knee_angle = 0.0
        valid_frames = 0
        
        for skel in skeleton_frames:
            if skel is None: continue
            
            # skel 是一個包含 17 個點的 list，每個點是 [x, y, conf]
            # 5: LShoulder, 6: RShoulder, 11: LHip, 12: RHip, 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle
            if len(skel) >= 17:
                ls, rs = skel[5], skel[6]
                lh, rh = skel[11], skel[12]
                lk, rk = skel[13], skel[14]
                la, ra = skel[15], skel[16]
                
                # 若信心度 > 0 才計算
                if lh[2] > 0 and lk[2] > 0:
                    l_hip = calculate_angle(ls, lh, lk) if ls[2] > 0 else 0
                    r_hip = calculate_angle(rs, rh, rk) if rs[2] > 0 else 0
                    l_knee = calculate_angle(lh, lk, la) if la[2] > 0 else 0
                    r_knee = calculate_angle(rh, rk, ra) if ra[2] > 0 else 0
                    
                    max_hip = max(l_hip, r_hip)
                    max_knee = max(l_knee, r_knee)
                    if max_hip > 0 or max_knee > 0:
                        avg_hip_angle += max_hip
                        avg_knee_angle += max_knee
                        valid_frames += 1
                        
        if valid_frames > 0:
            avg_hip_angle /= valid_frames
            avg_knee_angle /= valid_frames

        features = []
        for skel in skeleton_frames:
            if skel is None:
                continue
            
            # extract bounding box of the skeleton to normalize
            kpts = np.array(skel)
            
            # --- V3 Strategy: RESTORE LOW CONFIDENCE FILTERING ---
            # Set unconfident points to exactly 0 to eliminate fake velocity
            # The model is trained to handle these 0.0 values as pure dropout.
            unconfident_mask = kpts[:, 2] < 0.3
            kpts[unconfident_mask, 0] = 0.0
            kpts[unconfident_mask, 1] = 0.0
            kpts[unconfident_mask, 2] = 0.0

            valid = kpts[:, 2] > 0.2
            if not np.any(valid):
                continue
            
            xs = kpts[valid, 0]
            ys = kpts[valid, 1]
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            w = max(x_max - x_min, 1)
            h = max(y_max - y_min, 1)
            
            flat_kpts = []
            for kp in skel:
                flat_kpts.extend([kp[0]/w, kp[1]/h, kp[2]])
            features.append(flat_kpts)
            
        if not features:
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": len(skeleton_frames)}]
            
        features = np.array(features, dtype=np.float32)
        
        # --- VELOCITY FEATURES (Temporal Dynamics) ---
        velocity = np.zeros_like(features)
        for i in range(1, len(features)):
            for j in range(0, 51, 3):
                if features[i, j] != 0.0 and features[i-1, j] != 0.0:
                    velocity[i, j] = features[i, j] - features[i-1, j]
                    velocity[i, j+1] = features[i, j+1] - features[i-1, j+1]
        
        velocity[np.abs(velocity) < 0.002] = 0.0
        total_frames = len(features)
        
        # Check if lower body is completely missing
        lower_body_conf = features[:, [35, 38, 41, 44, 47, 50]] # 11, 12, 13, 14, 15, 16
        avg_lower_body_points = np.mean(np.sum(lower_body_conf > 0.0, axis=1))
        
        is_partial_skeleton = False
        
        # V4 Logic: VLM Handover for total occlusion
        if avg_lower_body_points < 1.0: 
            logger.info("Lower body completely missing (< 1.0 points). Returning Unknown for VLM handover.")
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": total_frames}]
        elif avg_lower_body_points < 4.0: 
            logger.info(f"Lower body partially missing ({avg_lower_body_points:.1f} pts). Flagging for VLM double-check.")
            is_partial_skeleton = True

        window_size = self.seq_len
        stride = 5
        
        if total_frames == 0:
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": 0}]
            
        if not np.any(features):
            return [{"action": "Unknown", "confidence": 0.0, "start_frame": 0, "end_frame": 0}]

        # Sliding Window
        raw_predictions = []
        prob_history = []
        history_len = 5  # Smooth over 5 sliding windows (approx 0.8 seconds)
        
        for start_idx in range(0, total_frames, stride):
            end_idx = min(start_idx + window_size, total_frames)
            
            # Extract window data
            window_features = features[start_idx:end_idx].copy()
            window_velocity = velocity[start_idx:end_idx].copy()
            
            # --- NORMALIZATION (Window-Level Translation Invariance) ---
            # Center this specific window around its first valid hip center
            # This PRESERVES absolute motion (walking across the screen) while normalizing screen position.
            anchor_x, anchor_y = 0.0, 0.0
            for f_idx in range(len(window_features)):
                left_hip_x, left_hip_y = window_features[f_idx, 33], window_features[f_idx, 34]
                right_hip_x, right_hip_y = window_features[f_idx, 36], window_features[f_idx, 37]
                
                if left_hip_x != 0.0 and right_hip_x != 0.0:
                    anchor_x = (left_hip_x + right_hip_x) / 2.0
                    anchor_y = (left_hip_y + right_hip_y) / 2.0
                    break
                elif left_hip_x != 0.0:
                    anchor_x, anchor_y = left_hip_x, left_hip_y
                    break
                elif right_hip_x != 0.0:
                    anchor_x, anchor_y = right_hip_x, right_hip_y
                    break
                    
            if anchor_x == 0.0 and anchor_y == 0.0:
                for f_idx in range(len(window_features)):
                    if window_features[f_idx, 0] != 0.0 and window_features[f_idx, 1] != 0.0:
                        anchor_x, anchor_y = window_features[f_idx, 0], window_features[f_idx, 1]
                        break
                        
            if anchor_x != 0.0 and anchor_y != 0.0:
                for f_idx in range(len(window_features)):
                    if window_features[f_idx, 33] != 0.0 or window_features[f_idx, 0] != 0.0:
                        for i in range(0, 51, 3):
                            if window_features[f_idx, i] != 0.0:
                                window_features[f_idx, i] -= anchor_x
                            if window_features[f_idx, i+1] != 0.0:
                                window_features[f_idx, i+1] -= anchor_y
                                
            # --- SCALE NORMALIZATION (Scale Invariance) ---
            max_height = 0.001
            for f_idx in range(len(window_features)):
                valid_y = []
                for i in range(1, 51, 3): # Y coordinates
                    if window_features[f_idx, i+1] != 0.0: # Check confidence
                        valid_y.append(window_features[f_idx, i])
                if len(valid_y) > 0:
                    height = max(valid_y) - min(valid_y)
                    if height > max_height:
                        max_height = height
                        
            for f_idx in range(len(window_features)):
                for i in range(0, 51, 3):
                    if window_features[f_idx, i+2] != 0.0: # Check confidence
                        window_features[f_idx, i] /= max_height
                        window_features[f_idx, i+1] /= max_height
                        
            # Normalize velocity using the SAME scale factor so velocity is relative to body size!
            window_velocity /= max_height
                                
            # Combine position and velocity for this window
            window_combined = np.concatenate([window_features, window_velocity], axis=1)
            
            if start_idx == 0:
                # print(f"DEBUG Window 0 pos sum: {np.sum(np.abs(window_features)):.4f}, vel sum: {np.sum(np.abs(window_velocity)):.4f}, anchor: {anchor_x:.4f}, {anchor_y:.4f}")
                pass
            
            # Pad or truncate to seq_len (60)
            seq = np.zeros((window_size, 102), dtype=np.float32)
            actual_len = len(window_combined)
            
            if actual_len < window_size and actual_len > 0:
                # REPEAT the last frame to fill the window (better than zero-padding for end-of-video actions)
                pad_len = window_size - actual_len
                last_frame = window_combined[-1:]
                padded_combined = np.vstack([window_combined, np.repeat(last_frame, pad_len, axis=0)])
                seq = padded_combined
            else:
                seq[:actual_len, :] = window_combined
            
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
            # print(f"DEBUG Action Model Window {start_idx}-{end_idx}: {action_name} ({conf:.2f})")
            raw_predictions.append({"action": action_name, "confidence": conf, "start_frame": start_idx, "end_frame": end_idx})
            
            if (end_idx - start_idx) < 15: # Stop if we are too close to the end
                break
                
        # Merge consecutive identical actions into a continuous non-overlapping timeline
        merged = []
        for pred in raw_predictions:
            if not merged:
                merged.append({
                    "action": pred["action"], 
                    "start_frame": 0, 
                    "end_frame": pred["end_frame"], 
                    "confidence": min(pred["confidence"], 0.50) if is_partial_skeleton else pred["confidence"],
                    "partial_skeleton": is_partial_skeleton
                })
            else:
                last = merged[-1]
                if last["action"] == pred["action"]:
                    last["end_frame"] = pred["end_frame"]
                    new_conf = (last["confidence"] + (min(pred["confidence"], 0.50) if is_partial_skeleton else pred["confidence"])) / 2.0
                    last["confidence"] = new_conf
                else:
                    # Set non-overlapping boundary around the center of the window
                    boundary = pred["start_frame"] + (pred["end_frame"] - pred["start_frame"]) // 2
                    last["end_frame"] = boundary
                    merged.append({
                        "action": pred["action"], 
                        "start_frame": boundary, 
                        "end_frame": pred["end_frame"], 
                        "confidence": min(pred["confidence"], 0.50) if is_partial_skeleton else pred["confidence"],
                        "partial_skeleton": is_partial_skeleton
                    })
        
        # Ensure the last segment reaches the end
        if merged:
            merged[-1]["end_frame"] = total_frames
                    
        return merged
