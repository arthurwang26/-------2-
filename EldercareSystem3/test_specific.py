import os
import cv2
import json
from pathlib import Path
from config import cfg
from inference.tracker import ByteTrackTracker
from inference.face import FaceIdentityMatcher
from inference.appearance_reid import AppearanceReID
from inference.pose import RTMPoseEstimator
from events.action_model import MotionBERTActionModel
from utils.reset_memory import ModelGuard

def test_video(video_path_str):
    video_path = Path(video_path_str)
    print(f"\n--- Testing {video_path.name} ---")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames.")
    
    # 1. Track
    with ModelGuard("Tracker"):
        tracker = ByteTrackTracker()
        tracker.load()
        tracks, objects = tracker.track(frames)
    
    # 2. Face
    with ModelGuard("Face Matcher"):
        face = FaceIdentityMatcher()
        face.load()
        identities = face.match_identities(frames, tracks)
        print("Initial Face Identities:", identities)
        
    # 3. ReID with Exclusion Logic
    from inference.appearance_reid import AppearanceReID
    reid_app = AppearanceReID()
    # First, enroll confidently matched faces
    face_matched_names = set()
    for tid, name in identities.items():
        if name != "Unknown":
            face_matched_names.add(name)
            for f_idx in range(0, len(frames), 5):
                frame_tracks = tracks[f_idx] if f_idx < len(tracks) else []
                t_obj = next((t for t in frame_tracks if t["track_id"] == tid), None)
                if t_obj:
                    w_box = t_obj["bbox"][2] - t_obj["bbox"][0]
                    h_box = t_obj["bbox"][3] - t_obj["bbox"][1]
                    if w_box > 0 and 0.5 < (h_box / w_box) < 4.0:
                        hist = reid_app.extract_histogram(frames[f_idx], t_obj["bbox"])
                        reid_app.update_gallery(name, hist)
                        
    # Then, identify Unknown tracks using ONLY remaining names!
    available_names = set(cfg.target_names) - face_matched_names
    if available_names:
        for tid in set(t["track_id"] for frame_tracks in tracks for t in frame_tracks):
            if identities.get(tid, "Unknown") == "Unknown":
                reid_votes = []
                for f_idx in range(0, len(frames), 5):
                    frame_tracks = tracks[f_idx] if f_idx < len(tracks) else []
                    t_obj = next((t for t in frame_tracks if t["track_id"] == tid), None)
                    if t_obj:
                        hist = reid_app.extract_histogram(frames[f_idx], t_obj["bbox"])
                        # Identify, but restrict to available_names
                        best_name = "Unknown"
                        best_sim = 0.5
                        for name in available_names:
                            if name in reid_app.gallery:
                                hists = reid_app.gallery[name]
                                sims = [cv2.compareHist(hist, h, cv2.HISTCMP_CORREL) for h in hists]
                                if sims:
                                    avg_sim = sum(sorted(sims, reverse=True)[:3]) / min(3, len(sims))
                                    if avg_sim > best_sim:
                                        best_sim = avg_sim
                                        best_name = name
                        if best_name != "Unknown":
                            reid_votes.append(best_name)
                
                if reid_votes:
                    from collections import Counter
                    most_common = Counter(reid_votes).most_common(1)[0]
                    if most_common[1] >= len(reid_votes) * 0.5:
                        identities[tid] = most_common[0]
                        # Once assigned, remove from available to prevent duplicates
                        available_names.discard(most_common[0])
    print("Final Identities:", identities)
    
    # 4. Pose
    with ModelGuard("Pose"):
        pose = RTMPoseEstimator()
        pose.load()
        skeletons = pose.estimate(frames, tracks)
        
    # Merge skeletons by person (including Unknown)
    person_skeletons = {name: [None]*len(frames) for name in list(cfg.target_names) + ["Unknown"]}
    for tid, name in identities.items():
        if name in person_skeletons:
            seq = skeletons.get(tid, [None]*len(frames))
            for i, kps in enumerate(seq):
                if kps is not None:
                    person_skeletons[name][i] = kps
                    
    # 5. Action
    with ModelGuard("Action"):
        from events.skeleton_action_model import CustomSkeletonActionModel
        action_model = CustomSkeletonActionModel()
    
    for name, seq in person_skeletons.items():
        print(f"\n{name} Actions:")
        valid = sum(1 for x in seq if x is not None)
        print(f"{name} has {valid} valid skeleton frames out of {len(frames)}")
        actions = action_model.predict(seq)
        for a in actions:
            print(f"  {a['start_frame']}->{a['end_frame']} | {a['action']} (conf: {a['confidence']:.2f})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_video(sys.argv[1])
    else:
        shared_dir = Path(r"C:\Users\arthu\Desktop\新增資料夾 (2)\shared_data")
        for video_file in shared_dir.glob("*.mp4"):
            test_video(str(video_file))
