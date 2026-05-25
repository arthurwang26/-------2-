import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from config import cfg
from utils.logger import get_logger
from utils.reset_memory import ModelGuard

logger = get_logger("pose")


class RTMPoseEstimator:
    """Skeleton extraction using YOLOv8s-pose (COCO 17 keypoints)."""

    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            model_path = cfg.pose_model_path
            if not model_path.exists():
                # Fallback to yolov8s-pose from ultralytics hub
                logger.warning(f"Pose model not found at {model_path}, using yolov8s-pose.pt")
                model_path = "yolov8s-pose.pt"
            self.model = YOLO(str(model_path))
            logger.info(f"Pose model loaded from {model_path}")

    def unload(self):
        self.model = None

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / (areaA + areaB - inter + 1e-8)

    def estimate(self, frames: List[np.ndarray],
                 tracks_batch: List[List[Dict]]) -> Dict[int, List[np.ndarray]]:
        """
        Extract 2D skeletons matched to tracked persons.
        Returns: track_id -> List[np.ndarray(17,3)] (x, y, confidence per frame)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        logger.info(f"Extracting skeletons for {len(frames)} frames...")
        skeleton_seqs = {}

        results = self.model(frames, verbose=False)

        # Initialize all tracked persons with a list of None matching frame length
        all_tids = set()
        for frame_tracks in tracks_batch:
            for t in frame_tracks:
                all_tids.add(t["track_id"])
        
        for tid in all_tids:
            skeleton_seqs[tid] = [None] * len(frames)

        for i, (res, tracks) in enumerate(zip(results, tracks_batch)):
            if res.keypoints is None or res.keypoints.data is None or not tracks:
                continue

            kps_all = res.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

            if len(kps_all) == 0 or len(boxes) == 0:
                continue

            # Match each pose detection to a tracked person via IoU
            for kps, box in zip(kps_all, boxes):
                best_tid = -1
                best_iou = 0.3  # minimum IoU threshold

                for t in tracks:
                    iou = self._iou(box, t["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = t["track_id"]

                if best_tid != -1:
                    skeleton_seqs[best_tid][i] = kps

        for tid, seq in skeleton_seqs.items():
            valid_frames = sum(1 for x in seq if x is not None)
            logger.debug(f"Track {tid}: extracted {valid_frames} skeleton frames.")

        return skeleton_seqs


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

        dummy_tracks = [[{"track_id": 1, "bbox": [0, 0, 640, 480]}] for _ in frames]
        with ModelGuard("Pose"):
            pose = RTMPoseEstimator()
            pose.load()
            skeletons = pose.estimate(frames, dummy_tracks)
            for tid, seq in skeletons.items():
                logger.info(f"Track {tid}: {len(seq)} frames, shape={seq[0].shape}")
