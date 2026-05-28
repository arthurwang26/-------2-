import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from events.custom_action_model import TransformerLSTMActionModel
from utils.logger import get_logger

logger = get_logger("train_ai")

# --- Dataset Definition ---
class SkeletonDataset(Dataset):
    def __init__(self, csv_file, seq_len=60, split='train'):
        self.seq_len = seq_len
        self.split = split
        logger.info(f"Loading dataset from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Handle NaN values and ensure data is sorted
        df = df.fillna(0.0)
        df = df.sort_values(by=['video', 'frame'])
        
        self.videos = df['video'].unique()
        self.sequences = []
        self.labels = []
        
        feature_cols = df.columns[2:53] # 51 features
        
        logger.info("Grouping frames into sequences...")
        for vid in tqdm(self.videos, desc="Processing videos"):
            vid_df = df[df['video'] == vid]
            features = vid_df[feature_cols].values # shape (frames, 51)
            label = vid_df['label'].values[0]
            
            # --- NORMALIZATION (Window-Level Translation Invariance) ---
            # Center the ENTIRE sequence relative to the FIRST valid HIP CENTER
            # This PRESERVES absolute motion (walking across the screen) while normalizing screen position.
            anchor_x, anchor_y = 0.0, 0.0
            for f_idx in range(len(features)):
                left_hip_x, left_hip_y = features[f_idx, 33], features[f_idx, 34]
                right_hip_x, right_hip_y = features[f_idx, 36], features[f_idx, 37]
                
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
                # Fallback to first valid nose
                for f_idx in range(len(features)):
                    if features[f_idx, 0] != 0.0 and features[f_idx, 1] != 0.0:
                        anchor_x, anchor_y = features[f_idx, 0], features[f_idx, 1]
                        break
                        
            if anchor_x != 0.0 and anchor_y != 0.0:
                for f_idx in range(len(features)):
                    if features[f_idx, 33] != 0.0 or features[f_idx, 0] != 0.0: # Check if frame is valid
                        for i in range(0, 51, 3):
                            if features[f_idx, i] != 0.0: # X
                                features[f_idx, i] -= anchor_x
                            if features[f_idx, i+1] != 0.0: # Y
                                features[f_idx, i+1] -= anchor_y
                                
            # --- SCALE NORMALIZATION (Scale Invariance) ---
            max_height = 0.001
            for f_idx in range(len(features)):
                valid_y = []
                for i in range(1, 51, 3): # Y coordinates
                    if features[f_idx, i] != 0.0:
                        # Before normalization, non-zero means valid, but now they are translated around 0.
                        # Wait, what if the translated Y is 0.0? It's still valid!
                        # We must check if confidence (i+1) is > 0
                        if features[f_idx, i+1] != 0.0:
                            valid_y.append(features[f_idx, i])
                if len(valid_y) > 0:
                    height = max(valid_y) - min(valid_y)
                    if height > max_height:
                        max_height = height
            
            for f_idx in range(len(features)):
                for i in range(0, 51, 3):
                    if features[f_idx, i+2] != 0.0: # Check confidence to know it's valid
                        features[f_idx, i] /= max_height
                        features[f_idx, i+1] /= max_height
            
            # --- VELOCITY FEATURES (Temporal Dynamics) ---
            # Calculate the difference between consecutive frames (dx/dt, dy/dt)
            velocity = np.zeros_like(features)
            for i in range(1, len(features)):
                for j in range(0, 51, 3):
                    # Only compute velocity if keypoint is present in BOTH frames
                    if features[i, j+2] != 0.0 and features[i-1, j+2] != 0.0:
                        velocity[i, j] = features[i, j] - features[i-1, j]
                        velocity[i, j+1] = features[i, j+1] - features[i-1, j+1]
            
            # Combine position and velocity (51 + 51 = 102 features)
            combined_features = np.concatenate([features, velocity], axis=1)
            
            # Pad or truncate to seq_len
            seq = np.zeros((self.seq_len, 102), dtype=np.float32)
            actual_len = min(len(combined_features), self.seq_len)
            seq[:actual_len, :] = combined_features[:actual_len, :]
            
            self.sequences.append(seq)
            self.labels.append(label)
            
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels, dtype=np.int64)
        logger.info(f"Loaded {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        
        # --- JITTER AUGMENTATION ---
        if self.split == 'train':
            # Add small random noise to positions (first 51 features)
            pos_mask = seq[:, 0:51] != 0.0
            jitter = np.random.normal(0, 0.005, seq[:, 0:51].shape).astype(np.float32)
            seq[:, 0:51][pos_mask] += jitter[pos_mask]

            # Vertical Scale Jitter (simulates people sitting taller/shorter or camera angle changes)
            # Randomly scale the Y coordinates by a factor between 0.85 and 1.15
            y_scale = np.random.uniform(0.85, 1.15)
            for i in range(1, 51, 3):
                y_mask = seq[:, i] != 0.0
                seq[:, i][y_mask] *= y_scale
            
            # Keypoint Left/Right Swap (Simulate Front/Back facing)
            # 50% chance to swap left and right joints to simulate the person turning around
            if np.random.random() < 0.5:
                pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
                for p1, p2 in pairs:
                    temp = seq[:, p1*3:(p1+1)*3].copy()
                    seq[:, p1*3:(p1+1)*3] = seq[:, p2*3:(p2+1)*3]
                    seq[:, p2*3:(p2+1)*3] = temp
            
            # Keypoint Dropout (V4: Partial Occlusion - KEEP AT LEAST ONE THIGH)
            # 1. 15% chance to simulate partial lower body occlusion
            if np.random.random() < 0.15:
                rand_val = np.random.random()
                if rand_val < 0.33:
                    # Case A: Drop both calves/feet, keep thighs
                    seq[:, 13*3:17*3] = 0.0
                elif rand_val < 0.66:
                    # Case B: Drop left leg entirely, keep right leg
                    for kp_idx in [11, 13, 15]:
                        seq[:, kp_idx*3:kp_idx*3+3] = 0.0
                else:
                    # Case C: Drop right leg entirely, keep left leg
                    for kp_idx in [12, 14, 16]:
                        seq[:, kp_idx*3:kp_idx*3+3] = 0.0
                        
            # 2. 5% chance to completely drop any random keypoint
            for kp_idx in range(17):
                if np.random.random() < 0.05:
                    seq[:, kp_idx*3:kp_idx*3+3] = 0.0
            
            # Recompute velocity based on noisy/dropped positions
            vel = np.zeros_like(seq[:, 0:51])
            vel[1:] = seq[1:, 0:51] - seq[:-1, 0:51]
            seq[:, 51:102] = vel
            
        return torch.tensor(seq), torch.tensor(self.labels[idx])

# --- Training Loop ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Paths
    train_csv = r"C:\Users\arthu\Desktop\新增資料夾 (2)\archive\dataset_action_split\train_yolov7.csv"
    test_csv = r"C:\Users\arthu\Desktop\新增資料夾 (2)\archive\dataset_action_split\test_yolov7.csv"
    save_path = "events/weights/custom_action_model.pth"
    
    os.makedirs("events/weights", exist_ok=True)
    
    # Hyperparameters
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.0005
    weight_decay = 1e-4  # L2 Regularization to prevent overfitting
    seq_len = 60
    
    # DataLoaders
    train_dataset = SkeletonDataset(train_csv, seq_len=seq_len, split='train')
    test_dataset = SkeletonDataset(test_csv, seq_len=seq_len, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Add custom Focal Loss
    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma=2.0, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.weight = weight
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            import torch.nn.functional as F
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', label_smoothing=0.1)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            return focal_loss.sum()

    # Model (Now using 102 features and Bidirectional LSTM)
    model = TransformerLSTMActionModel(input_dim=102, hidden_dim=256, num_classes=7, dropout=0.5).to(device)
    
    # Class weights to handle imbalance
    # Base inverse frequency weights: [3.26, 1.09, 0.48, 2.41, 0.64, 1.73, 0.86]
    # Moderate adjustment: slightly boost Sitting(6) and Lying Down(1), slightly suppress Standing(4)
    weights = torch.tensor([3.26, 1.4, 0.48, 2.41, 0.45, 1.73, 1.3], dtype=torch.float32).to(device)
    
    # Remove label smoothing to force confident predictions
    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma=2.0, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.weight = weight
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            import torch.nn.functional as F
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            if self.reduction == 'mean': return focal_loss.mean()
            return focal_loss.sum()
            
    criterion = FocalLoss(weight=weights, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # --- DATA AUGMENTATION ---
            # Add random Gaussian noise to coordinates during training to prevent overfitting
            noise = torch.randn_like(inputs) * 0.01
            inputs_aug = inputs + noise
            
            outputs = model(inputs_aug)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                v_loss = criterion(outputs, labels)
                val_loss += v_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning Rate Scheduler
        scheduler.step(val_acc)
        
        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best model to {save_path} with Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered. Best Val Acc: {best_acc:.2f}%")
                break

if __name__ == "__main__":
    train_model()
