# Eldercare System - Global Memory & Architecture Decisions

This document serves as the permanent memory for the Eldercare project. All future AI agents MUST read this file to understand the system's history, user preferences, and critical architectural decisions.

## 🔴 CRITICAL RULES
1. **NO EMOTION INFERENCE (絕對禁止猜測情緒)**
   - The system is strictly forbidden from describing or guessing emotional states, moods, or psychological feelings.
   - It must only describe **confirmed, physical actions**.
   - If the Ground Truth (GT) evaluation complains about missing emotions (like "sad" or "happy"), ignore it. Our system design explicitly outlaws emotion tracking.

## 🏗️ ARCHITECTURE DECISIONS
1. **Skeleton Action Model (Custom Transformer + Bi-LSTM)**
   - **Why not ST-GCN (NTU-60)?**: ST-GCN was previously tested and its performance was **POOR**. The NTU-60 pre-trained behaviors do not align well with our specific eldercare CCTV angles or nuanced actions. Do not suggest reverting to ST-GCN.
   - **Why not MotionBERT?**: Previously used in System 2, but we migrated to the custom Transformer+LSTM to have a lightweight, tailored solution that trains specifically on the Kaggle eldercare dataset.

## 🐛 CRITICAL BUG FIXES (Do Not Revert)
1. **Scale Normalization (空間維度常規化)**
   - The custom Kaggle dataset trained the LSTM using coordinates normalized between `0.0 - 1.0`.
   - YOLOv8 outputs raw pixel coordinates (e.g., 1920x1080).
   - **Fix**: All keypoint coordinates MUST be divided by the frame width and height before passing to the LSTM. Failing to do this causes the model to perceive massive velocity (dx/dt) and misclassify simple movements as "Fall Down".

2. **Temporal Alignment / Velocity Distortion (時間軸對齊)**
   - When extracting skeletons, frames without a detected skeleton MUST be padded with `None` or zeros to maintain the original sequence length (`len(frames)`).
   - **Fix**: Do not skip empty frames. If empty frames are skipped, the LSTM calculates velocity between non-adjacent frames as if they were adjacent, resulting in distorted "supersonic" movements.

3. **ReID Poisoning Loop (防毒化衣服追蹤)**
   - If `AppearanceReID` misidentifies a track on the first frame and immediately updates its gallery, it poisons the database with the wrong person's clothes.
   - **Fix**: 
     - Only update the `AppearanceReID` gallery from tracks that have been **confidently identified by FaceIdentityMatcher**.
     - When identifying an "Unknown" track, use **Majority Voting** (sample every 5 frames across the whole track) instead of relying on the first frame.
