import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from config import cfg
from utils.logger import get_logger
from utils.reset_memory import ModelGuard

logger = get_logger("tracker")


class ByteTrackTracker:
    """
    Combined YOLOv8s + ByteTrack tracker.
    Returns both person tracks (with IDs) and object detections.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path or cfg.yolo_model
        self.model = None

    def load(self):
        if self.model is None:
            self.model = YOLO(self.model_path)
            if "world" in str(self.model_path).lower():
                classes = ["person"] + list(cfg.relevant_objects)
                self.model.set_classes(classes)
                logger.info(f"YOLO-World set open-vocabulary classes: {classes}")
            logger.info(f"Loaded YOLO tracker from {self.model_path}")

    def unload(self):
        self.model = None

    def track(self, frames: List[np.ndarray]) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Run ByteTrack on frames.
        Returns:
            person_tracks: List[frame] of List[dict] with track_id, bbox, confidence
            object_detections: List[frame] of List[dict] with class_name, bbox, confidence
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        logger.info(f"Running tracking on {len(frames)} frames...")
        results = self.model.track(frames, tracker="bytetrack.yaml", persist=True,
                                   verbose=False, conf=0.2)

        all_tracks = []
        all_objects = []

        for i, res in enumerate(results):
            frame_tracks = []
            frame_objects = []

            if res.boxes is not None:
                has_ids = res.boxes.id is not None
                for j, box in enumerate(res.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    name = res.names[cls_id]

                    if name == "person":
                        tid = int(res.boxes.id[j].item()) if has_ids else j
                        frame_tracks.append({
                            "track_id": tid,
                            "class_name": "person",
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        })
                    elif name in cfg.relevant_objects and conf > 0.25:
                        frame_objects.append({
                            "class_name": name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        })

            all_tracks.append(frame_tracks)
            all_objects.append(frame_objects)
            logger.debug(f"Frame {i}: {len(frame_tracks)} persons, {len(frame_objects)} objects. "
                         f"IDs: {[t['track_id'] for t in frame_tracks]}")

        return all_tracks, all_objects


if __name__ == "__main__":
    import cv2
    video_files = sorted(cfg.raw_dir.glob("*.mp4"))
    if video_files:
        cap = cv2.VideoCapture(str(video_files[0]))
        frames = []
        for _ in range(10):
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()

        with ModelGuard("ByteTrack"):
            tracker = ByteTrackTracker()
            tracker.load()
            tracks, objects = tracker.track(frames)
            for i, (t, o) in enumerate(zip(tracks, objects)):
                logger.info(f"Frame {i}: persons={len(t)}, objects={[x['class_name'] for x in o]}")
