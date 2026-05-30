import os
import cv2
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def main():
    dataset_dir = r"C:\Users\arthu\Desktop\新增資料夾 (2)\archive\dataset_action_split"
    
    # Load YOLOv8-pose
    model = YOLO('yolov8s-pose.pt')
    
    splits = ['train', 'test']
    
    # Keypoint columns
    kp_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Create column headers
    columns = ['video', 'frame']
    for name in kp_names:
        columns.extend([f'{name}_x', f'{name}_y', f'{name}_s'])
    columns.append('label')
    
    class_to_idx = {
        "Sitting": 0,
        "Fall Down": 1,
        "Walking": 2,
        "Standing": 3,
        "Lying Down": 5, # We map falling/lying down to similar indexes
        "Sit down": 6,
        "Stand up": 4
    }
    
    for split in splits:
        print(f"Processing {split} split...")
        data_rows = []
        
        split_dir = os.path.join(dataset_dir, split)
        classes = os.listdir(split_dir)
        
        for cls_name in classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            
            label = class_to_idx.get(cls_name)
            if label is None:
                print(f"Warning: Unknown class {cls_name}, skipping.")
                continue
                
            videos = glob.glob(os.path.join(cls_dir, "*.avi"))
            for vid_path in tqdm(videos, desc=f"{cls_name}"):
                vid_name = f"{cls_name}/{os.path.basename(vid_path)}"
                cap = cv2.VideoCapture(vid_path)
                
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(frame)
                cap.release()
                
                if not frames: continue
                
                # Batch process with YOLO pose in chunks to prevent OOM
                BATCH_SIZE = 16
                for batch_start in range(0, len(frames), BATCH_SIZE):
                    batch_frames = frames[batch_start:batch_start+BATCH_SIZE]
                    results = model(batch_frames, verbose=False)
                    
                    for i_in_batch, res in enumerate(results):
                        f_idx = batch_start + i_in_batch
                        
                        # Get the person with the largest bounding box (or highest conf)
                        if res.keypoints is None or res.keypoints.data is None or len(res.keypoints.data) == 0:
                            continue
                            
                        kps = res.keypoints.data.cpu().numpy()
                        boxes = res.boxes.xywh.cpu().numpy() if res.boxes is not None else []
                        
                        if len(boxes) > 0:
                            # Find largest box area
                            areas = boxes[:, 2] * boxes[:, 3]
                            best_idx = np.argmax(areas)
                        else:
                            best_idx = 0
                            
                        best_kps = kps[best_idx] # (17, 3)
                        
                        row = [vid_name, f_idx + 1]
                        for i in range(17):
                            row.extend([best_kps[i][0], best_kps[i][1], best_kps[i][2]])
                        row.append(label)
                        
                        data_rows.append(row)
                    
        # Save to CSV
        df = pd.DataFrame(data_rows, columns=columns)
        out_path = os.path.join(dataset_dir, f"{split}_rtmpose.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
