# 模組輸出規格書 (V4.4 雙模態版)

本文件定義了 `EldercareSystem` 中核心感測模組的 I/O 規格。在 V4.4 版本中，動作辨識全面升級為 MMAction2 預訓練模型，人機互動則採用 `HOI-MLP` 與 `HOI-CLIP` 雙軌架構。

## 1. 動作辨識 (ST-GCN)
**負責模組**: `events.action_model.STGCNActionModel`
**技術堆疊**: MMAction2 (NTU-RGB+D 60) `stgcn_ntu60_xsub_coco17.pth`
*   **Input**:
    *   `skeleton_seqs`: 字典 `Dict[track_id, List[np.ndarray]]`，陣列形狀為 `(T, 17, 3)`
*   **Output**:
    ```json
    {
        4: {
            "action": "Sitting", // 從 NTU-60 對應而來，包含 20% 幾何 Fallback 修正
            "confidence": 0.85
        }
    }
    ```

## 2. 人機互動 (Dual-HOI)
**負責模組**: `events.hoi_model.HOIPredictor` (MLP) 與 `events.clip_hoi.CLIPHOIPredictor` (CLIP)
**技術堆疊**: 自監督多層感知機 (Online Learning) + OpenAI `clip-vit-base-patch32` (Zero-Shot Inference)
*   **Input (MLP)**: 
    *   `skeleton_seqs` (骨架), `object_detections` (YOLO BBox)
*   **Input (CLIP)**:
    *   `sampled_frame` (中段影像抽樣), `sampled_skeletons` (單幀骨架), `sampled_objects` (單幀物件)
*   **Output**: 陣列合併
    ```json
    [
        {
            "track_id": 4,
            "action": "Holding",
            "object": "cup",
            "confidence": 0.95,
            "type": "HOI-MLP"
        },
        {
            "track_id": 4,
            "action": "Looking_At",
            "object": "tv",
            "confidence": 0.88,
            "type": "HOI-CLIP"
        }
    ]
    ```

## 3. 情緒辨識 (HSEmotion)
**負責模組**: `inference.emotion.EmotionRecognizer`
*   **Input**:
    *   `frames`: 影像陣列
    *   `tracks`: 該幀的 ByteTrack 追蹤結果
*   **Output**:
    ```json
    {
        4: [
            {"emotion": "Neutral", "score": 0.71},
            {"emotion": "Sadness", "score": 0.29}
        ]
    }
    ```

## 4. 異常偵測 (Autoencoder)
**負責模組**: `events.anomaly_model.AnomalyDetector`
*   **Output**:
    ```json
    {
        4: {
            "is_anomaly": false,
            "anomaly_score": 1.25 // 基於 MSE 重構誤差
        }
    }
    ```
