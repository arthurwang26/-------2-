import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import torch

from config import cfg
from events.skeleton_action_model import CustomSkeletonActionModel
from events.clip_hoi import CLIPHOIPredictor
from report.blip_caption import BLIPCaptioner
from report.vlm_caption import SmolVLMCaptioner
from report.llm_report import QwenReporter
from database.ts_db import TimeSeriesDB
from utils.reset_memory import ModelGuard, enforce_vram_clear

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)-10s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("resume")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_debug_json(vid_name, file_name, data):
    debug_dir = cfg.output_dir / "debug" / vid_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def load_debug_json(vid_name, file_name):
    debug_dir = cfg.output_dir / "debug" / vid_name
    path = debug_dir / file_name
    if not path.exists(): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_resume():
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ts_db = TimeSeriesDB(str(cfg.output_dir / "timeseries.db"))
    
    events_by_clip = {}
    captions_by_clip = {}
    vlm_captions_dict = {}
    objects_by_clip = {}
    video_files = []
    for ext in ("*.mp4", "*.avi"):
        video_files.extend(list(cfg.raw_dir.glob(ext)))
    video_files = sorted(video_files)
    
    if not video_files:
        logger.error(f"No videos found in {cfg.raw_dir}.")
        return

    for vid_path in video_files:
        vid_name = vid_path.stem
        logger.info(f"\n{'='*60}\nResuming Video: {vid_name}\n{'='*60}")
        
        # Load saved data
        tracks = load_debug_json(vid_name, "tracks.json")
        objects = load_debug_json(vid_name, "objects.json")
        skeletons = load_debug_json(vid_name, "skeletons.json")
        clip_events = load_debug_json(vid_name, "clip_hoi.json")
        
        if tracks is None or objects is None or skeletons is None or clip_events is None:
            logger.warning(f"Missing debug data for {vid_name}, skipping.")
            continue
            
        objects_by_clip[vid_name] = objects
        
        # Extract frames
        cap = cv2.VideoCapture(str(vid_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0: fps = 24
        
        frames = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= 150: break
            frames.append(frame)
            frame_idx += 1
        cap.release()
        
        # Extract identities from old clip_events
        identities = {}
        for ev in clip_events:
            tid = ev.get("track_id")
            person = ev.get("person")
            if tid is not None and person is not None and person != "Unknown":
                identities[str(tid)] = person
                
        # Fill unknowns
        for tid in skeletons.keys():
            if str(tid) not in identities:
                identities[str(tid)] = "Unknown"
                
        # Re-run Skeleton Action Predictor to apply geometric rules
        actions = {}
        clip_events = [e for e in clip_events if e.get("type") != "Action"] # Remove old actions
        
        with ModelGuard("TransformerLSTM"):
            action_model = CustomSkeletonActionModel()
            for tid, skels in skeletons.items():
                if identities.get(tid) == "Unknown": continue
                acts = action_model.predict(skels)
                actions[tid] = acts
                for act in acts:
                    clip_events.append({
                        "video": vid_name,
                        "frame_idx": act["start_frame"],
                        "track_id": int(tid),
                        "person": identities.get(tid, "Unknown"),
                        "type": "Action",
                        "action": act["action"],
                        "confidence": act["confidence"],
                        "start_frame": act["start_frame"],
                        "end_frame": act["end_frame"],
                        "partial_skeleton": act.get("partial_skeleton", False),
                        "geometric_override": act.get("geometric_override", False)
                    })
            
        del action_model
        enforce_vram_clear()

        # BLIP Captioning
        with ModelGuard("BLIP"):
            blip = BLIPCaptioner()
            blip.load()
            captions = []
            for i in range(0, len(frames), 30):
                cap_text = blip.generate_caption(frames[i], "This is a photo of ")
                captions.append({
                    "frame_idx": i,
                    "caption": cap_text,
                    "video": vid_name
                })
            blip.unload() if hasattr(blip, 'unload') else None
        del blip
        enforce_vram_clear()
        captions_by_clip[vid_name] = captions

        # VLM Captioning & Override
        with ModelGuard("VLM"):
            vlm = SmolVLMCaptioner()
            vlm.load()
            
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
                            
                            for ev in clip_events:
                                if ev.get("track_id") == tid and ev.get("type") == "Action" and ev.get("start_frame") == act["start_frame"]:
                                    ev["action"] = resolved_act
                                    ev["confidence"] = 0.80
                                    
            # Verify HOI
            hoi_candidates = [h for h in clip_events if h.get("type") in ("HOI", "HOI-CLIP")]
            for h in hoi_candidates:
                f_idx = min(h.get("frame_idx", len(frames)//2), len(frames)-1)
                action_str = h.get("action", "")
                obj_str = h.get("object", "")
                image = __import__('PIL').Image.fromarray(frames[f_idx][:, :, ::-1])
                prompt = f"Is the person in this image {action_str.lower().replace('_', ' ')} a {obj_str.lower()}? Answer Yes or No."
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                text = vlm.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = vlm.processor(text=text, images=[image], return_tensors="pt").to(vlm.device)
                with torch.no_grad():
                    gen = vlm.model.generate(**inputs, max_new_tokens=10)
                decoded = vlm.processor.batch_decode(gen, skip_special_tokens=True)
                response = decoded[0].split("Assistant:")[-1].strip().lower()
                h["vlm_verified"] = "yes" in response
                h["vlm_answer"] = response
                
            vlm_captions = vlm.caption_keyframes(frames, objects, identities, interval=30)
            vlm_captions_dict[vid_name] = vlm_captions
            vlm.unload() if hasattr(vlm, 'unload') else None
            
        del vlm
        enforce_vram_clear()
        
        events_by_clip[vid_name] = clip_events
        ts_db.insert_events({vid_name: clip_events})
        
        # Vis skipped to save time in resume
        
        # Save updated debug json
        save_debug_json(vid_name, "actions.json", actions)
        save_debug_json(vid_name, "clip_hoi.json", clip_events)
        
    # KG
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

    # LLM
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
        
    sys_dir = cfg.output_dir / "system_docs"
    sys_dir.mkdir(parents=True, exist_ok=True)
    report_path = sys_dir / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    logger.info("Resume Pipeline Complete!")

if __name__ == "__main__":
    run_resume()
