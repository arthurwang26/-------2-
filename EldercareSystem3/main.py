"""
ElderCare 多模態影片行為分析系統 — 主管線
所有模型均為真實 DL 推理，無硬編碼覆蓋。

v2.1: Track Merging — 將 ByteTrack 碎片化的 track_id 合併為單一身份
"""
import os
import sys
import glob
import cv2
import json

import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent))

import torch
from functools import partial
torch.load = partial(torch.load, weights_only=False)

from config import cfg
from utils.logger import get_logger
from utils.reset_memory import ModelGuard, enforce_vram_clear, clear_all_cache

logger = get_logger("main")

EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']


def extract_frames(video_path: str) -> tuple:
    """Extract frames from video and return original fps."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 8
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, int(fps)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def save_debug_json(vid_name, file_name, data):
    debug_dir = cfg.output_dir / "debug" / vid_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def save_video(frames: List[np.ndarray], path: str, fps: int = 8):
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def merge_tracks_by_identity(identities: Dict[int, str],
                              skeletons: Dict[int, List[np.ndarray]],
                              tracks: List[List[Dict]]) -> tuple:
    """
    CRITICAL FIX: Merge fragmented track_ids belonging to the same person.
    
    ByteTrack assigns 10+ track_ids to the same person due to occlusions/turns.
    This function merges all data (skeletons, track detections) into
    a single canonical track_id per person.
    
    Returns: (merged_identities, merged_skeletons, merged_tracks, canonical_map)
    """
    # Group track_ids by person name
    name_to_tids = defaultdict(list)
    for tid, name in identities.items():
        if name != "Unknown":
            name_to_tids[name].append(tid)
    
    # For each person, pick the canonical track_id (the one with most skeleton frames)
    canonical_map = {}  # old_tid -> canonical_tid
    canonical_to_name = {}
    
    for name, tids in name_to_tids.items():
        # Pick the tid with the most skeleton data as canonical
        tid_skel_len = [(tid, len(skeletons.get(tid, []))) for tid in tids]
        tid_skel_len.sort(key=lambda x: x[1], reverse=True)
        canonical_tid = tid_skel_len[0][0]
        
        for tid in tids:
            canonical_map[tid] = canonical_tid
        canonical_to_name[canonical_tid] = name
        
        if len(tids) > 1:
            logger.info(f"Merging {name}: {len(tids)} track_ids {tids} -> canonical={canonical_tid}")
    
    # Merge skeletons
    merged_skeletons = {}
    for name, tids in name_to_tids.items():
        canonical_tid = canonical_map[tids[0]]
        max_len = max([len(skeletons.get(t, [])) for t in tids] + [0])
        merged_seq = [None] * max_len
        for tid in tids:
            seq = skeletons.get(tid, [])
            for i, kps in enumerate(seq):
                if kps is not None and merged_seq[i] is None:
                    merged_seq[i] = kps
        if any(x is not None for x in merged_seq):
            merged_skeletons[canonical_tid] = merged_seq
    
    # Build merged identities
    merged_identities = {}
    for tid, name in identities.items():
        if name != "Unknown" and tid in canonical_map:
            merged_identities[canonical_map[tid]] = name
        elif name == "Unknown" and tid not in canonical_map:
            merged_identities[tid] = "Unknown"
    
    # Update tracks for visualization: remap track_ids
    merged_tracks = []
    for frame_tracks in tracks:
        seen_canonical = set()
        new_frame_tracks = []
        for t in frame_tracks:
            tid = t["track_id"]
            if tid in canonical_map:
                new_tid = canonical_map[tid]
                if new_tid not in seen_canonical:
                    seen_canonical.add(new_tid)
                    new_t = t.copy()
                    new_t["track_id"] = new_tid
                    new_frame_tracks.append(new_t)
            else:
                # Unknown person: only keep if it has some skeleton data
                # FIX: Check if name is Unknown OR it is not in identities yet
                new_frame_tracks.append(t)
        merged_tracks.append(new_frame_tracks)
    
    # Get the canonical track_ids for known persons
    canonical_tids = set(canonical_to_name.keys())
    
    logger.info(f"Track merge complete: {sum(len(v) for v in name_to_tids.values())} track_ids -> {len(canonical_tids)} persons")
    
    return merged_identities, merged_skeletons, merged_tracks, canonical_map


def process_pipeline():
    import shutil
    logger.info("=" * 60)
    logger.info("ElderCare Multimodal Behavior Analysis System v2.1 (T600)")
    logger.info("=" * 60)

    # Clean up output directories to ensure fresh start
    dirs_to_clean = ["visuals", "llm_reports", "system_docs", "debug", "database"]
    for d in dirs_to_clean:
        dir_path = cfg.output_dir / d
        if dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)
            logger.info(f"Cleaned up {dir_path}")

    # Clear all caches
    clear_all_cache(cfg.project_root)
    enforce_vram_clear()

    # Import modules
    from inference.tracker import ByteTrackTracker
    from inference.face import FaceIdentityMatcher
    from inference.pose import RTMPoseEstimator
    from inference.appearance_reid import AppearanceReID
    # Removed RGBActionModel
    from events.clip_hoi import CLIPHOIPredictor
    from database.kg_exporter import KnowledgeGraphExporter
    from database.ts_db import TimeSeriesDB
    from report.blip_caption import BLIPCaptioner
    from report.vlm_caption import SmolVLMCaptioner

    from report.llm_report import QwenReporter
    from utils.visualizer import draw_frame

    # Init Time-Series DB
    ts_db = TimeSeriesDB()

    # Find videos
    all_video_files = sorted(glob.glob(str(cfg.raw_dir / "*.mp4")))
    video_files = all_video_files
    if not video_files:
        logger.error(f"No videos found in {cfg.raw_dir}")
        return
    logger.info(f"Found {len(video_files)} videos.")

    # Global state    
    events_by_clip = {}
    objects_by_clip = {}
    skeletons_by_clip = {}
    captions_by_clip = {}
    vlm_captions_dict = {}

    all_baseline_skeletons = {}

    # Global Appearance ReID to handle back-to-camera tracking
    reid_app = AppearanceReID()

    for vid_idx, video_path in enumerate(video_files):
        vid_name = Path(video_path).stem
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing [{vid_idx+1}/{len(video_files)}]: {vid_name}")
        logger.info(f"{'='*50}")

        frames, fps = extract_frames(video_path)
        if not frames:
            logger.warning(f"No frames from {video_path}")
            continue
        logger.info(f"Extracted {len(frames)} frames.")

        clip_events = []

        # ===== PERCEPTION LAYER =====

        # 1. Tracking (YOLO + ByteTrack) — fresh tracker per clip for clean IDs
        tracker = ByteTrackTracker()
        tracker.load()
        tracks, objects = tracker.track(frames)
        tracker.unload()
        del tracker
        enforce_vram_clear()

        # Log detection summary
        all_tids = set()
        for frame_tracks in tracks:
            for t in frame_tracks:
                all_tids.add(t["track_id"])
        logger.info(f"Tracked {len(all_tids)} unique track_ids (before merge).")

        all_obj_classes = set()
        for frame_objs in objects:
            for o in frame_objs:
                all_obj_classes.add(o["class_name"])
        logger.info(f"Detected objects: {all_obj_classes}")

        # 2. Face Identity Matching
        with ModelGuard("Face Matcher"):
            face = FaceIdentityMatcher()
            face.load()
            identities = face.match_identities(frames, tracks)
            face.unload() if hasattr(face, 'unload') else None
        del face
        enforce_vram_clear()
        
        # Cross-clip Appearance ReID (fixes back-to-camera without assumptions)
        # 1. Update gallery ONLY using confidently Face-Matched tracks to prevent poisoning
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
                        # ML Anti-Poisoning Constraint: Only extract features from reasonable bounding boxes
                        if w_box > 0 and 0.5 < (h_box / w_box) < 4.0:
                            hist = reid_app.extract_histogram(frames[f_idx], t_obj["bbox"])
                            reid_app.update_gallery(name, hist)

        # 2. Identify unknown tracks using the clean gallery via majority voting, EXCLUDING assigned names
        available_names = set(cfg.target_names) - face_matched_names
        if available_names:
            for tid in all_tids:
                if identities.get(tid, "Unknown") == "Unknown":
                    reid_votes = []
                    for f_idx in range(0, len(frames), 5):
                        frame_tracks = tracks[f_idx] if f_idx < len(tracks) else []
                        t_obj = next((t for t in frame_tracks if t["track_id"] == tid), None)
                        if t_obj:
                            hist = reid_app.extract_histogram(frames[f_idx], t_obj["bbox"])
                            
                            best_name = "Unknown"
                            best_sim = 0.60
                            for name in available_names:
                                if name in reid_app.gallery:
                                    import cv2
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
                        # Only assign if the majority vote is strong enough
                        if most_common[1] >= len(reid_votes) * 0.5:
                            identities[tid] = most_common[0]
                            logger.info(f"ReID Match: Track {tid} -> {most_common[0]} based on majority clothing votes")
                            available_names.discard(most_common[0])

        logger.info(f"Identities before merge: {identities}")

        # 3. Pose Estimation
        with ModelGuard("Pose"):
            pose = RTMPoseEstimator()
            pose.load()
            skeletons = pose.estimate(frames, tracks)
            pose.unload() if hasattr(pose, 'unload') else None
        del pose
        enforce_vram_clear()
        logger.info(f"Skeletons extracted for tracks: {list(skeletons.keys())}")
        
        # Filter identities by MIN_PRESENCE_RATIO to remove phantom tracks (false positives)
        MIN_PRESENCE_RATIO = 0.05
        for tid in list(identities.keys()):
            # A track must have bounding boxes in at least 5% of frames
            # to be considered a valid person, else downgrade to Unknown
            bbox_count = sum(1 for frame_t in tracks if any(t['track_id'] == tid for t in frame_t))
            if bbox_count < len(frames) * MIN_PRESENCE_RATIO:
                if identities.get(tid, "Unknown") != "Unknown":
                    logger.info(f"Downgrading track {tid} ({identities[tid]}) to Unknown due to low presence ({bbox_count}/{len(frames)} frames)")
                    identities[tid] = "Unknown"

        # ===== v2.1 CRITICAL FIX: TRACK MERGING =====
        # Merge fragmented track_ids for the same person
        (merged_identities, merged_skeletons, 
         merged_tracks, canonical_map) = merge_tracks_by_identity(
            identities, skeletons, tracks)
        
        # Use merged data from here on
        identities = merged_identities
        skeletons = merged_skeletons
        tracks = merged_tracks

        # Store skeletons for anomaly detection
        skeletons_by_clip[vid_name] = skeletons
        if vid_idx == 0:
            all_baseline_skeletons = skeletons.copy()

        # Count actual persons (known identities only)
        known_persons = set(n for n in identities.values() if n != "Unknown")
        num_persons = max(1, len(known_persons))
        logger.info(f"After merge: {num_persons} known persons: {known_persons}")

        # ===== EVENT GENERATION LAYER =====

        # 5. Action Recognition (Custom Transformer+LSTM with Skeletons)
        with ModelGuard("TransformerLSTM"):
            from events.skeleton_action_model import CustomSkeletonActionModel
            act_model = CustomSkeletonActionModel()
            
            actions = {}
            frame_h, frame_w = frames[0].shape[:2]
            for tid in identities.keys():
                skel_seq = skeletons.get(tid, [])
                act_list = act_model.predict(skel_seq, w=frame_w, h=frame_h)
                actions[tid] = act_list

        for tid, act_list in actions.items():
            name = identities.get(tid, f"ID:{tid}")
            if name == "Unknown":
                continue
            for act in act_list:
                clip_events.append({
                    "video": vid_name, "track_id": tid, "person": name,
                    "type": "Action", "action": act["action"],
                    "confidence": act["confidence"],
                    "start_frame": act["start_frame"],
                    "end_frame": act["end_frame"]
                })

        # 6. HOI Prediction (Dynamic CLIP Zero-Shot using YOLO objects)
        with ModelGuard("HOI CLIP"):
            clip_model = CLIPHOIPredictor()
            clip_model.load()
            # Pass YOLO objects to dynamically build CLIP prompts
            hoi_events_clip = clip_model.predict(frames, tracks, objects_per_frame=objects, sample_rate=30)
            clip_model.unload() if hasattr(clip_model, 'unload') else None
        del clip_model
        enforce_vram_clear()

        for h in hoi_events_clip:
            name = identities.get(h["track_id"], "Unknown")
            if name == "Unknown":
                continue
            h["video"] = vid_name
            h["person"] = name
            clip_events.append(h)

        # ===== BLIP CAPTION (Keyframe captioning only, verification moved to VLM) =====
        with ModelGuard("BLIP"):
            blip = BLIPCaptioner()
            blip.load()
            
            # Keyframe Captioning only (BLIP is too weak for verification)
            captions = []
            for i in range(0, len(frames), 30):
                cap_text = blip.generate_caption(frames[i], "This is a photo of ")
                captions.append({
                    "frame_idx": i,
                    "caption": cap_text
                })
            
            blip.unload() if hasattr(blip, 'unload') else None
        
        del blip
        enforce_vram_clear()
        
        for c in captions:
            c["video"] = vid_name
        captions_by_clip[vid_name] = captions

        # ===== VLM CAPTION & CROSS-VALIDATION =====
        with ModelGuard("VLM"):
            vlm = SmolVLMCaptioner()
            vlm.load()
            
            # 1. VLM Fallback for Unknown AND low-confidence skeleton actions
            LOW_CONF_THRESHOLD = 0.6
            for tid, act_list in actions.items():
                name = identities.get(tid, f"ID:{tid}")
                if name == "Unknown": continue
                for act in act_list:
                    needs_vlm = (
                        act["action"] == "Unknown" or 
                        act.get("partial_skeleton", False) or
                        (act["confidence"] < LOW_CONF_THRESHOLD and act["confidence"] > 0 and act["action"] not in ["Walking", "Sit down", "Standing"]) or
                        act["action"] in ["Lying Down", "Fall Down", "Stand up"]
                    )
                    if needs_vlm:
                        original_act = act["action"]
                        resolved_act = vlm.resolve_unknown_action(frames, act["start_frame"], act["end_frame"], name)
                        if resolved_act != "Unknown":
                            logger.info(f"VLM override: {name} {original_act}(conf={act['confidence']:.2f}) -> {resolved_act}")
                            act["action"] = resolved_act
                            act["confidence"] = 0.80
                            
                            # Also update the clip_events that was already populated
                            for ev in clip_events:
                                if ev.get("track_id") == tid and ev.get("type") == "Action" and ev.get("start_frame") == act["start_frame"]:
                                    ev["action"] = resolved_act
                                    ev["confidence"] = 0.80
            
            # 2. VLM verification of HOI events (replaces BLIP which was too weak)
            hoi_candidates = [h for h in clip_events if h.get("type") in ("HOI", "HOI-CLIP")]
            for h in hoi_candidates:
                f_idx = h.get("frame_idx", len(frames)//2)
                f_idx = min(f_idx, len(frames)-1)
                action_str = h.get("action", "")
                obj_str = h.get("object", "")
                
                image = __import__('PIL').Image.fromarray(frames[f_idx][:, :, ::-1])
                prompt = f"Is the person in this image {action_str.lower().replace('_', ' ')} a {obj_str.lower()}? Answer Yes or No."
                
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                text = vlm.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = vlm.processor(text=text, images=[image], return_tensors="pt")
                inputs = inputs.to(vlm.device)
                
                with torch.no_grad():
                    gen = vlm.model.generate(**inputs, max_new_tokens=10)
                decoded = vlm.processor.batch_decode(gen, skip_special_tokens=True)
                response = decoded[0].split("Assistant:")[-1].strip().lower()
                
                h["vlm_verified"] = "yes" in response
                h["vlm_answer"] = response
                logger.info(f"VLM verified HOI {action_str}/{obj_str}: {h['vlm_verified']} ({response})")
            
            # 3. VLM Keyframe Captioning
            vlm_captions = vlm.caption_keyframes(frames, objects, identities, interval=30)
            vlm_captions_dict[vid_name] = vlm_captions
            vlm.unload() if hasattr(vlm, 'unload') else None
        
        del vlm
        enforce_vram_clear()
        events_by_clip[vid_name] = clip_events
        # Save to Time-Series DB
        ts_db.insert_events({vid_name: clip_events})

        # ===== VISUALIZATION (using merged tracks) =====
        vis_frames = []
        for i, frame in enumerate(frames):
            frame_tracks_i = tracks[i] if i < len(tracks) else []
            # Restore skeleton drawing
            frame_skels = {}
            for tid, skels in skeletons.items():
                if i < len(skels) and skels[i] is not None:
                    frame_skels[tid] = skels[i]
            frame_actions = {}
            for tid, act_list in actions.items():
                for act in act_list:
                    if act["start_frame"] <= i < act["end_frame"]:
                        frame_actions[tid] = act["action"]
                        break
            
            frame_emotions = {}
            frame_objs = objects[i] if i < len(objects) else []

            v = draw_frame(
                frame,
                detections=frame_tracks_i,
                skeletons=frame_skels,
                identities=identities,
                actions=frame_actions,
                emotions=frame_emotions,
                objects=frame_objs
            )
            vis_frames.append(v)

        vis_path = str(cfg.output_dir / "visuals" / f"{vid_name}_vis.mp4")
        save_video(vis_frames, vis_path, fps=fps)
        logger.info(f"Visualization saved: {vis_path}")

        # Log clip summary
        logger.info(f"--- {vid_name} Summary ---")
        for ev in clip_events:
            logger.info(f"  {ev['person']}: {ev['type']}={ev.get('action', '')}")
            
        # V5.1 Module-level comprehensive debug outputs
        save_debug_json(vid_name, "tracks.json", tracks)
        save_debug_json(vid_name, "objects.json", objects)
        save_debug_json(vid_name, "skeletons.json", skeletons)
        save_debug_json(vid_name, "actions.json", actions)
        save_debug_json(vid_name, "clip_hoi.json", clip_events)
        objects_by_clip[vid_name] = objects
    # ===== KNOWLEDGE GRAPH GENERATION =====
    logger.info("\n--- Phase: Knowledge Graph Generation ---")
    kg_text = ""
    sys_dir = cfg.output_dir / "system_docs"
    sys_dir.mkdir(parents=True, exist_ok=True)
    try:
        from report.kg_generator import generate_mermaid_graph, generate_neo4j_cypher
        kg_md_path = sys_dir / "knowledge_graph.md"
        kg_text = generate_mermaid_graph(events_by_clip, kg_md_path)
        logger.info(f"Knowledge Graph (Markdown) saved to {kg_md_path}")
        
        kg_cypher_path = sys_dir / "knowledge_graph.cypher"
        generate_neo4j_cypher(events_by_clip, kg_cypher_path)
        logger.info(f"Knowledge Graph (Neo4j Cypher) saved to {kg_cypher_path}")
    except Exception as e:
        logger.error(f"Failed to generate KG: {e}")

    # ===== LLM REPORT GENERATION =====
    logger.info("\n--- Phase: LLM Report Generation ---")
    with ModelGuard("Qwen3"):
        reporter = QwenReporter()
        reporter.load()
        report = reporter.generate_report(
            events_by_clip=events_by_clip,
            captions_by_clip=captions_by_clip,
            vlm_captions_by_clip=vlm_captions_dict,
            kg_text=kg_text,
            objects_by_clip=objects_by_clip
        )
    
    # Save outputs
    sys_dir = cfg.output_dir / "system_docs"
    sys_dir.mkdir(parents=True, exist_ok=True)
    report_path = sys_dir / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Save JSON
    debug_dir = cfg.output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "events.json", "w", encoding="utf-8") as f:
        json.dump(events_by_clip, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("Pipeline Complete!")
    logger.info(f"{'='*60}")
    enforce_vram_clear()


if __name__ == "__main__":
    process_pipeline()
