import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import cfg
from utils.logger import get_logger
from utils.reset_memory import ModelGuard

logger = get_logger("face_matcher")


class FaceIdentityMatcher:
    """SCRFD (via InsightFace) + ArcFace for face recognition against 2 target persons.
    
    v2.1 FIX: Removed greedy one-name-per-track constraint.
    ByteTrack fragments one person into many track_ids, so the same name
    MUST be assignable to multiple tracks. We match each track independently.
    """

    def __init__(self):
        self.app = None
        self.target_embeddings = {}  # name -> 512d normalized vector

    def load(self):
        if self.app is not None:
            return
        import insightface
        logger.info("Loading InsightFace (SCRFD + ArcFace)...")
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace loaded.")
        self._load_target_faces()

    def _load_target_faces(self):
        """Load the 2 target face images and extract embeddings."""
        enrollment_dir = cfg.enrollment_dir
        for name in cfg.target_names:
            # Try common image extensions
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                p = enrollment_dir / f"{name}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                logger.error(f"Target face image not found for {name} in {enrollment_dir}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Failed to read image: {img_path}")
                continue

            faces = self.app.get(img)
            if not faces:
                logger.error(f"No face detected in {img_path}")
                continue

            # Use the largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            emb = face.normed_embedding
            self.target_embeddings[name] = emb
            logger.info(f"Enrolled target: {name} (embedding shape: {emb.shape})")

        logger.info(f"Target lock configured: {list(self.target_embeddings.keys())}")

    def unload(self):
        self.app = None
        self.target_embeddings = {}

    def match_identities(self, frames: List[np.ndarray],
                         tracks_batch: List[List[Dict]]) -> Dict[int, str]:
        """
        Match detected faces to target persons across all frames.
        Returns mapping: track_id -> identity name ("王奶奶" / "陳爺爺" / "Unknown")
        
        v2.1: Each track_id is independently matched. The same name CAN be assigned
        to multiple track_ids since ByteTrack fragments one person into many IDs.
        """
        if self.app is None:
            raise RuntimeError("Model is not loaded.")
        if not self.target_embeddings:
            logger.warning("No target embeddings loaded! All identities will be Unknown.")
            return {}

        logger.info(f"Running face matching on {len(frames)} frames...")
        identity_votes = {}  # track_id -> list of (name, similarity)

        # Sample frames for efficiency (every 3rd frame)
        sample_indices = list(range(0, len(frames), 3))

        for idx in sample_indices:
            frame = frames[idx]
            tracks = tracks_batch[idx] if idx < len(tracks_batch) else []
            if not tracks:
                continue

            faces = self.app.get(frame)
            if not faces:
                continue

            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox
                fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2

                # Find which track this face belongs to
                best_tid = -1
                best_iou = 0
                for t in tracks:
                    tx1, ty1, tx2, ty2 = t["bbox"]
                    if tx1 <= fcx <= tx2 and ty1 <= fcy <= ty2:
                        # Instead of IoU which penalizes larger person boxes,
                        # we use the distance from the face center to the TOP-CENTER of the person box.
                        # The face should be near the top-center of the person.
                        pxc = (tx1 + tx2) / 2
                        pyc = ty1 + (ty2 - ty1) * 0.15  # Estimate face is roughly at 15% from top
                        
                        dist = ((fcx - pxc) ** 2 + (fcy - pyc) ** 2) ** 0.5
                        
                        # We want the minimum distance
                        if best_tid == -1 or dist < best_iou:  # using best_iou to store min dist
                            best_iou = dist
                            best_tid = t["track_id"]

                if best_tid == -1:
                    continue

                emb = face.normed_embedding
                best_sim = 0.40
                best_name = "Unknown"

                for name, target_emb in self.target_embeddings.items():
                    sim = float(np.dot(emb, target_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

                if best_tid not in identity_votes:
                    identity_votes[best_tid] = []
                identity_votes[best_tid].append((best_name, best_sim))

        # v2.1: Independent assignment - NO greedy constraint
        # Each track gets its best match independently.
        # The same person name CAN appear on multiple track_ids.
        final_map = {}

        for tid, votes in identity_votes.items():
            known = [(n, s) for n, s in votes if n != "Unknown"]
            if known:
                # Majority vote: pick the name with the most votes
                from collections import Counter
                name_counts = Counter(n for n, s in known)
                best_name = name_counts.most_common(1)[0][0]
                # Average similarity for that name
                avg_sim = np.mean([s for n, s in known if n == best_name])
                if avg_sim > 0.3:
                    final_map[tid] = best_name
                    logger.info(f"Track {tid} -> {best_name} (avg_sim={avg_sim:.3f}, votes={name_counts[best_name]})")
                else:
                    final_map[tid] = "Unknown"
            else:
                final_map[tid] = "Unknown"

        # Assign remaining tracks that had no face detections
        for tracks_frame in tracks_batch:
            for t in tracks_frame:
                if t["track_id"] not in final_map:
                    final_map[t["track_id"]] = "Unknown"

        # Log summary
        name_counts = {}
        for tid, name in final_map.items():
            if name != "Unknown":
                name_counts.setdefault(name, []).append(tid)
        for name, tids in name_counts.items():
            logger.info(f"  {name}: assigned to {len(tids)} track_ids = {tids}")

        return final_map


if __name__ == "__main__":
    video_files = sorted(cfg.raw_dir.glob("*.mp4"))
    if video_files:
        cap = cv2.VideoCapture(str(video_files[0]))
        frames = []
        for _ in range(15):
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()

        dummy_tracks = [[{"track_id": 1, "bbox": [0, 0, f.shape[1], f.shape[0]]}] for f in frames]
        with ModelGuard("Face Matcher"):
            matcher = FaceIdentityMatcher()
            matcher.load()
            ids = matcher.match_identities(frames, dummy_tracks)
            logger.info(f"Identity map: {ids}")
