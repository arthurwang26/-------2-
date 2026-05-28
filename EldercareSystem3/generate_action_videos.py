import os
import sys
import glob
import cv2
import json
import numpy as np
from pathlib import Path

from config import cfg
from events.skeleton_action_model import CustomSkeletonActionModel
from utils.visualizer import draw_frame
from utils.logger import get_logger

logger = get_logger("action_vis")

def save_video(frames, path, fps=8):
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

def run():
    out_dir = cfg.output_dir / "action_videos_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(glob.glob(str(cfg.raw_dir / "*.mp4")))
    
    act_model = CustomSkeletonActionModel()

    for vid_path in video_files:
        vid_name = Path(vid_path).stem
        logger.info(f"Processing {vid_name}...")

        # Load video frames
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 8
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames: continue
        frame_h, frame_w = frames[0].shape[:2]

        # Load debug JSONs
        debug_dir = cfg.output_dir / "debug" / vid_name
        with open(debug_dir / "tracks.json", "r", encoding="utf-8") as f:
            tracks = json.load(f)
        with open(debug_dir / "skeletons.json", "r", encoding="utf-8") as f:
            skeletons = json.load(f)
        
        # Load identities from tracks (ByteTrack + Face Matcher merged tracks are saved in tracks.json? No, identities aren't saved explicitly, but tracks have them? Wait, tracks.json just has track_id. Wait, how do I get identities?)
        # Let's extract identities from tracks or clip_hoi.json?
        # clip_hoi.json has "track_id" and "person".
        with open(debug_dir / "clip_hoi.json", "r", encoding="utf-8") as f:
            clip_events = json.load(f)
        
        identities = {}
        for ev in clip_events:
            tid = ev.get("track_id")
            person = ev.get("person")
            if tid is not None and person:
                # Need integer tid
                identities[int(tid)] = person

        # Fix skeletons keys to int
        skeletons = {int(k): v for k, v in skeletons.items()}

        # Predict actions using the LATEST rule engine
        actions = {}
        for tid_str, skel_seq in skeletons.items():
            tid = int(tid_str)
            if tid not in identities or identities[tid] == "Unknown":
                continue
            act_list = act_model.predict(skel_seq)
            actions[tid] = act_list

        vis_frames = []
        for i, frame in enumerate(frames):
            frame_tracks_i = tracks[i] if i < len(tracks) else []
            frame_skels = {}
            for tid, skels in skeletons.items():
                if i < len(skels) and skels[i] is not None:
                    # JSON stores as list, visualizer expects ndarray
                    frame_skels[tid] = np.array(skels[i])
            
            frame_actions = {}
            for tid, act_list in actions.items():
                for act in act_list:
                    if act["start_frame"] <= i < act["end_frame"]:
                        # Append Geometric rule flag to action text for visualization
                        action_text = act["action"]
                        if act.get("partial_skeleton", False):
                            action_text += " (RULE)"
                        frame_actions[tid] = action_text
                        break

            v = draw_frame(
                frame,
                detections=frame_tracks_i,
                skeletons=frame_skels,
                identities=identities,
                actions=frame_actions,
            )
            vis_frames.append(v)

        vis_path = str(out_dir / f"{vid_name}_vis.mp4")
        save_video(vis_frames, vis_path, fps=int(fps))
        logger.info(f"Saved: {vis_path}")

if __name__ == "__main__":
    run()
