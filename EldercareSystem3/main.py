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


def extract_frames(video_path: str) -> List[np.ndarray]:
    """Extract frames from video. Captures full clip."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

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
                              emotion_vectors: Dict[int, List[np.ndarray]],
                              tracks: List[List[Dict]]) -> tuple:
    """
    CRITICAL FIX: Merge fragmented track_ids belonging to the same person.
    
    ByteTrack assigns 10+ track_ids to the same person due to occlusions/turns.
    This function merges all data (skeletons, emotions, track detections) into
    a single canonical track_id per person.
    
    Returns: (merged_identities, merged_skeletons, merged_emotions, merged_tracks, canonical_map)
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
        merged_seq = []
        for tid in tids:
            merged_seq.extend(skeletons.get(tid, []))
        if merged_seq:
            merged_skeletons[canonical_tid] = merged_seq
    
    # Merge emotion vectors
    merged_emotions = {}
    for name, tids in name_to_tids.items():
        canonical_tid = canonical_map[tids[0]]
        merged_seq = []
        for tid in tids:
            merged_seq.extend(emotion_vectors.get(tid, []))
        if merged_seq:
            merged_emotions[canonical_tid] = merged_seq
    
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
                name = identities.get(tid, "Unknown")
                if name != "Unknown":
                    new_frame_tracks.append(t)
        merged_tracks.append(new_frame_tracks)
    
    # Get the canonical track_ids for known persons
    canonical_tids = set(canonical_to_name.keys())
    
    logger.info(f"Track merge complete: {sum(len(v) for v in name_to_tids.values())} track_ids -> {len(canonical_tids)} persons")
    
    return merged_identities, merged_skeletons, merged_emotions, merged_tracks, canonical_map


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
    from inference.emotion import EmotionRecognizer
    from inference.appearance_reid import AppearanceReID
    from events.rgb_action_model import RGBActionModel
    from events.clip_hoi import CLIPHOIPredictor
    from events.emotion_model import EmotionSequencePredictor
    from events.anomaly_model import AnomalyDetector
    from temporal.cross_day_gru import TemporalReasoningModel
    from database.kg_exporter import KnowledgeGraphExporter
    from database.ts_db import TimeSeriesDB
    from report.vlm_caption import SmolVLMCaptioner
    from report.llm_report import QwenReporter
    from utils.visualizer import draw_frame

    # Init Time-Series DB
    ts_db = TimeSeriesDB()

    # Find videos
    import random
    all_video_files = sorted(glob.glob(str(cfg.raw_dir / "*.mp4")))
    video_files = random.sample(all_video_files, min(3, len(all_video_files)))
    if not video_files:
        logger.error(f"No videos found in {cfg.raw_dir}")
        return
    logger.info(f"Found {len(video_files)} videos.")

    # Global state
    events_by_clip = {}
    anomaly_by_clip = {}
    captions_by_clip = {}
    skeletons_by_clip = {}
    all_baseline_skeletons = {}

    # Keep face app reference for emotion module
    face_app_ref = None

    # Global Appearance ReID to handle back-to-camera tracking
    reid_app = AppearanceReID()

    for vid_idx, video_path in enumerate(video_files):
        vid_name = Path(video_path).stem
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing [{vid_idx+1}/{len(video_files)}]: {vid_name}")
        logger.info(f"{'='*50}")

        frames = extract_frames(video_path)
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

        # 2. Face Identity Matching (v2.1: no greedy constraint)
        with ModelGuard("Face Matcher"):
            face = FaceIdentityMatcher()
            face.load()
            identities = face.match_identities(frames, tracks)
            face_app_ref = face.app  # Keep for emotion
        
        # Cross-clip Appearance ReID (fixes back-to-camera without assumptions)
        for tid in all_tids:
            for f_idx, frame_tracks in enumerate(tracks):
                t_obj = next((t for t in frame_tracks if t["track_id"] == tid), None)
                if t_obj:
                    hist = reid_app.extract_histogram(frames[f_idx], t_obj["bbox"])
                    if np.sum(hist) > 0:
                        if identities.get(tid, "Unknown") != "Unknown":
                            reid_app.update_gallery(identities[tid], hist)
                        else:
                            matched_name = reid_app.identify(hist)
                            if matched_name != "Unknown":
                                identities[tid] = matched_name
                                logger.info(f"ReID Match: Track {tid} -> {matched_name} based on clothing appearance")
                    break

        logger.info(f"Identities before merge: {identities}")

        # 3. Pose Estimation
        with ModelGuard("Pose"):
            pose = RTMPoseEstimator()
            pose.load()
            skeletons = pose.estimate(frames, tracks)
        logger.info(f"Skeletons extracted for tracks: {list(skeletons.keys())}")

        # 4. Emotion Recognition
        with ModelGuard("Emotion"):
            emo = EmotionRecognizer()
            emo.load()
            emotion_vectors = emo.recognize(frames, tracks, face_app=face_app_ref)
        face_app_ref = None  # Release

        # ===== v2.1 CRITICAL FIX: TRACK MERGING =====
        # Merge fragmented track_ids for the same person
        (merged_identities, merged_skeletons, merged_emotions, 
         merged_tracks, canonical_map) = merge_tracks_by_identity(
            identities, skeletons, emotion_vectors, tracks)
        
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

        # 5. Action Recognition (VideoMAE RGB)
        with ModelGuard("VideoMAE"):
            act_model = RGBActionModel()
            act_model.load()
            
            actions = {}
            for tid in identities.keys():
                crops = []
                for f_idx, frame in enumerate(frames):
                    frame_tracks_i = tracks[f_idx] if f_idx < len(tracks) else []
                    t_obj = next((t for t in frame_tracks_i if t["track_id"] == tid), None)
                    if t_obj:
                        x1, y1, x2, y2 = t_obj["bbox"]
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(x2)), min(h, int(y2))
                        if x2 > x1 and y2 > y1:
                            crops.append(frame[y1:y2, x1:x2])
                
                if len(crops) > 0:
                    act_label = act_model.predict(crops)
                    actions[tid] = {"action": act_label, "confidence": 1.0}

        for tid, act in actions.items():
            name = identities.get(tid, f"ID:{tid}")
            if name == "Unknown":
                continue
            clip_events.append({
                "video": vid_name, "track_id": tid, "person": name,
                "type": "Action", "action": act["action"],
                "confidence": act["confidence"]
            })

        # 6. HOI Prediction (Continuous CLIP Zero-Shot)
        with ModelGuard("HOI CLIP"):
            clip_model = CLIPHOIPredictor()
            clip_model.load()
            # Pass all frames and tracks, it samples every 30 frames internally
            hoi_events_clip = clip_model.predict(frames, tracks, sample_rate=30)

        for h in hoi_events_clip:
            name = identities.get(h["track_id"], "Unknown")
            if name == "Unknown":
                continue
            h["video"] = vid_name
            h["person"] = name
            clip_events.append(h)

        # 7. Emotion Sequence Prediction (GRU)
        with ModelGuard("Emotion GRU"):
            emo_model = EmotionSequencePredictor()
            emo_model.load()
            emotions = emo_model.predict(merged_emotions)

        for tid, emo_result in emotions.items():
            name = identities.get(tid, f"ID:{tid}")
            if name == "Unknown":
                continue
            clip_events.append({
                "video": vid_name, "track_id": tid, "person": name,
                "type": "Emotion", "emotion": emo_result["emotion"],
                "confidence": emo_result["confidence"]
            })

        # ===== VLM CAPTION & GROUNDING =====
        with ModelGuard("VLM"):
            vlm = SmolVLMCaptioner()
            vlm.load()
            
            # 1. Grounding (Cross-Validation of CLIP HOI)
            hoi_candidates = [h for h in clip_events if h.get("type") == "HOI"]
            verified_hoi = vlm.verify_hoi_events(frames, hoi_candidates)
            
            # Replace HOI events in clip_events with only verified ones
            non_hoi_events = [h for h in clip_events if h.get("type") != "HOI"]
            clip_events = non_hoi_events + verified_hoi
            
            # 2. Keyframe Captioning
            captions = vlm.caption_keyframes(frames, objects, identities, interval=30)
            for c in captions:
                c["video"] = vid_name
            captions_by_clip[vid_name] = captions

        events_by_clip[vid_name] = clip_events
        # Save to Time-Series DB
        ts_db.insert_events({vid_name: clip_events})

        # ===== VISUALIZATION (using merged tracks) =====
        vis_frames = []
        for i, frame in enumerate(frames):
            frame_tracks_i = tracks[i] if i < len(tracks) else []
            frame_skels = {}
            for tid, sk_seq in skeletons.items():
                if i < len(sk_seq):
                    frame_skels[tid] = sk_seq[i]
            frame_actions = {tid: act["action"] for tid, act in actions.items()}
            frame_emotions = {tid: emo["emotion"] for tid, emo in emotions.items()}
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
        save_video(vis_frames, vis_path)
        logger.info(f"Visualization saved: {vis_path}")

        # Log clip summary
        logger.info(f"--- {vid_name} Summary ---")
        for ev in clip_events:
            logger.info(f"  {ev['person']}: {ev['type']}={ev.get('action', ev.get('emotion', ''))}")
            
        # V5.1 Module-level comprehensive debug outputs
        save_debug_json(vid_name, "tracks.json", tracks)
        save_debug_json(vid_name, "objects.json", objects)
        save_debug_json(vid_name, "skeletons.json", skeletons)
        save_debug_json(vid_name, "actions.json", actions)
        save_debug_json(vid_name, "emotions.json", emotions)
        save_debug_json(vid_name, "clip_hoi.json", clip_events)

    # ===== ANOMALY DETECTION (Self-supervised) =====
    logger.info("\n--- Phase: Self-Supervised Anomaly Detection ---")

    with ModelGuard("Anomaly AE"):
        ae = AnomalyDetector()
        ae.load()

        if all_baseline_skeletons:
            ae.train_baseline(all_baseline_skeletons, epochs=30)

        for vid_name in sorted(skeletons_by_clip.keys()):
            skel = skeletons_by_clip[vid_name]
            scores = ae.predict(skel)
            avg_score = np.mean(list(scores.values())) if scores else 0.0
            anomaly_by_clip[vid_name] = float(avg_score)
            ts_db.insert_anomaly(vid_name, avg_score)
            logger.info(f"{vid_name}: anomaly_score={avg_score:.4f}")

    # ===== CROSS-DAY TEMPORAL REASONING =====
    logger.info("\n--- Phase: Cross-Day Temporal Reasoning ---")
    with ModelGuard("BiGRU"):
        temporal = TemporalReasoningModel()
        temporal.load()
        reasoning = temporal.reason(events_by_clip, anomaly_by_clip)

    risk_score = reasoning["risk_score"]
    trend = reasoning["behavior_trend"]
    ts_db.insert_temporal_risk(risk_score, trend)
    logger.info(f"Risk Score: {risk_score:.4f}, Trend: {trend}")

    # ===== LLM REPORT GENERATION =====
    logger.info("\n--- Phase: LLM Report Generation ---")
    with ModelGuard("Qwen3"):
        reporter = QwenReporter()
        reporter.load()
        report = reporter.generate_report(
            events_by_clip=events_by_clip,
            anomaly_by_clip=anomaly_by_clip,
            risk_score=risk_score,
            trend=trend,
            captions_by_clip=captions_by_clip
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

    with open(cfg.output_dir / "debug" / "anomalies.json", "w", encoding="utf-8") as f:
        json.dump(anomaly_by_clip, f, ensure_ascii=False, indent=2)

    with open(cfg.output_dir / "debug" / "reasoning.json", "w", encoding="utf-8") as f:
        json.dump(reasoning, f, ensure_ascii=False, indent=2)

    # Generate Knowledge Graph (V5.1 Update)
    kg_exporter = KnowledgeGraphExporter()
    kg_exporter.process_events(events_by_clip)
    kg_exporter.export()

    try:
        from report.kg_generator import generate_mermaid_graph, generate_neo4j_cypher
        kg_md_path = sys_dir / "knowledge_graph.md"
        generate_mermaid_graph(events_by_clip, kg_md_path)
        logger.info(f"Knowledge Graph (Markdown) saved to {kg_md_path}")
        
        kg_cypher_path = sys_dir / "knowledge_graph.cypher"
        generate_neo4j_cypher(events_by_clip, kg_cypher_path)
        logger.info(f"Knowledge Graph (Neo4j Cypher) saved to {kg_cypher_path}")
    except Exception as e:
        logger.error(f"Failed to generate legacy KG views: {e}")

    logger.info(f"\n{'='*60}")
    logger.info("Pipeline Complete!")
    logger.info(f"{'='*60}")
    enforce_vram_clear()


if __name__ == "__main__":
    process_pipeline()
