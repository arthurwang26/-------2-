import json
import numpy as np
import math

def calculate_angle(p1, p2, p3):
    # p2 is the vertex
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    # Add a small epsilon to avoid division by zero
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 0.0
    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

try:
    data = json.load(open('output3/debug/day1_clip02_11點/skeletons.json', encoding='utf-8'))
    for tid, frames in data.items():
        print(f'Track {tid}:')
        for frame_idx, frame in enumerate(frames[:5]): # Check first 5 frames
            # 5: LShoulder, 11: LHip, 13: LKnee, 15: LAnkle
            ls = frame[5]
            lh = frame[11]
            lk = frame[13]
            la = frame[15]
            
            # 6: RShoulder, 12: RHip, 14: RKnee, 16: RAnkle
            rs = frame[6]
            rh = frame[12]
            rk = frame[14]
            ra = frame[16]
            
            hip_angle_l = calculate_angle(ls, lh, lk)
            hip_angle_r = calculate_angle(rs, rh, rk)
            
            knee_angle_l = calculate_angle(lh, lk, la)
            knee_angle_r = calculate_angle(rh, rk, ra)
            
            print(f'  Frame {frame_idx}: LHipAngle={hip_angle_l:.1f}, RHipAngle={hip_angle_r:.1f}, LKneeAngle={knee_angle_l:.1f}, RKneeAngle={knee_angle_r:.1f}')
except Exception as e:
    print(f"Error: {e}")
