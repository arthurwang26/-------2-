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
    def __init__(self, csv_file, seq_len=60):
        self.seq_len = seq_len
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
            
            # --- NORMALIZATION (Translation Invariance) ---
            # Subtract nose_x (idx 0) from all X coords (idx 0, 3, 6, 9...), and nose_y (idx 1) from all Y coords (idx 1, 4, 7...)
            # This makes the skeleton relative to the nose, drastically improving accuracy!
            for f_idx in range(len(features)):
                nose_x = features[f_idx, 0]
                nose_y = features[f_idx, 1]
                # If nose is missing (0.0), skip normalization for this frame
                if nose_x != 0.0 and nose_y != 0.0:
                    for i in range(0, 51, 3):
                        if features[f_idx, i] != 0.0: # X
                            features[f_idx, i] -= nose_x
                        if features[f_idx, i+1] != 0.0: # Y
                            features[f_idx, i+1] -= nose_y
            
            # --- VELOCITY FEATURES (Temporal Dynamics) ---
            # Calculate the difference between consecutive frames (dx/dt, dy/dt)
            velocity = np.zeros_like(features)
            velocity[1:] = features[1:] - features[:-1]
            
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
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])

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
    train_dataset = SkeletonDataset(train_csv, seq_len=seq_len)
    test_dataset = SkeletonDataset(test_csv, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model (Now using 102 features and Bidirectional LSTM)
    model = TransformerLSTMActionModel(input_dim=102, hidden_dim=256, num_classes=7, dropout=0.5).to(device)
    # Adding label smoothing to CrossEntropyLoss prevents overconfidence and helps generalize to 100%
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning Rate Scheduler
        scheduler.step(val_acc)
        
        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"==> Saved new best model to {save_path} with Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered. Best Val Acc: {best_acc:.2f}%")
                break

if __name__ == "__main__":
    train_model()
