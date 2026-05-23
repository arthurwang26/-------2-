

> **V5.1 真・最終定案更新**: 包含完整 MotionBERT 60 標籤、SQLite 時序資料庫、VLM Grounding 交叉驗證、以及解除 150 幀限制與模組級 Debug JSON 輸出。
# 給下一代開發者的交接文件 (AI Handover Document) - V5.1

你好，接手的工程師。
這個系統已經進化到 **V5.1 終極版**，請在修改任何程式碼之前，詳讀本交接文件，這會為你省下無數個熬夜除錯的夜晚。

## 一、 系統目前的極限在哪裡？

這個系統最核心的靈魂不是模型多強大，而是**它能在 NVIDIA T600 (4GB VRAM) 上把 9 個神經網路無縫接力跑完**。
1. **千萬不要拔掉 `ModelGuard`**。只要拔掉，系統絕對 OOM (Out Of Memory)。
2. **影片必須切割處理**。`main.py` 是設計成一個 clip 一個 clip 處理，且處理完後記憶體會強制 GC (`gc.collect()`)。

## 二、 各模組踩雷指南

### 1. `events/clip_hoi.py` (V5.1 新增的 CLIP HOI)
*   **雷區**：CLIP 很吃顯存，雖然我們用了 `vit-base-patch32`，但請確保傳給它的 `images` 不要過大。
*   **架構**：目前的設計是只在每支影片的中間擷取一張 `sampled_frame`，去裁切 Union BBox 進行推論。如果你想要每一幀都跑 CLIP，速度會慢到無法接受。

### 2. `events/action_model.py` (MotionBERT (DSTformer))
*   **現狀**：已載入 MMAction2 `stgcn_ntu60_xsub_coco17.pth`。
*   **雷區**：如果你想換模型，請務必確定你的 MotionBERT (DSTformer) 輸入通道對齊 COCO-17 (17 個關節, `x, y, conf`)。這支程式碼裡面有寫死的 `NTU60_MAPPING`，如果你的新權重不是在 NTU-60 上訓練的，這個字典會爆炸。

### 3. `events/hoi_model.py` (自監督 MLP)
*   **現狀**：為了跟 CLIP 做對比，我們保留了這個基於幾何的 MLP。
*   **設計概念**：它利用當下的距離特徵產生 Pseudo-labels 即時訓練。不要覺得奇怪為什麼每次推論前它都要跑 epochs，這是故意的（Online Learning）。

### 4. `report/vlm_caption.py` (視覺語言模型)
*   **現狀**：使用 `SmolVLM2-256M`。
*   **相容性警告**：請務必使用 `transformers >= 4.45`。舊版的 Hugging Face 架構會跳 `ImportError: cannot import name 'SmolVLMForCausalLM'`，必須使用 `AutoModelForImageTextToText`。

### 5. `report/llm_report.py` (最終報表生成)
*   **角色定位**：我們讓 Qwen 同時吃「Ground Truth」與「Sensor Nodes」。
*   **Prompt 修改警告**：請絕對不要修改提示詞裡的 `VLM 獨立原則` 與 `最後才判斷 Ground Truth`。如果你讓 LLM 提早看到答案，它就不會去認真分析前端感測器為什麼算錯，這會毀掉我們這套系統「自動診斷」的核心價值。

## 三、 V5.1 殘留的已知問題與 V5.0 升級路線圖 (Known Issues & Roadmap)

目前系統已達到 4GB VRAM 的架構極限，但仍存在以下幾個領域特有的工程盲區，請接手者優先解決：

### 1. 動作辨識的「動靜態盲區 (Domain Gap)」
*   **問題點**：預訓練的 MotionBERT (DSTformer) (NTU-60) 是時間序列模型 (Temporal)，它只能辨識「動態動作 (如坐下、跌倒)」。當長者處於「靜態姿勢 (維持坐著、站著)」時，骨架沒有時間變化，模型會完全瞎掉並輸出 0 分，導致系統只能退回舊版的物理幾何公式 (20% 權重)。
*   **V5.0 解法**：**自行訓練微型 HAR (Human Activity Recognition) 姿勢分類器**。
    *   仿造 `HOI-MLP` 的自監督邏輯，寫一個只有 3 層的 `Posture-MLP`。
    *   直接吃前端抓出來的 RTMPose (COCO 17點) 座標。
    *   系統開機時利用前幾十秒的畫面進行線上微調 (Online Fine-tuning)，將「靜態」的判定從物理公式徹底升級為 AI 預測。

### 2. 異常跌倒偵測 (Autoencoder) 的假警報
*   **問題點**：目前 `AnomalyDetector` 只能在系統啟動的第一支短片 (8秒鐘) 內學習「正常狀態」。訓練基準線過短，導致只要長者換個姿勢，重構誤差就會飆高，引發大量假陽性 (False Positives)。
*   **V5.0 解法**：**延長基線緩衝區 (Baseline Buffer)**。將訓練資料擴展到「過去一週的滑動區間 (Sliding Window)」，或是讓系統連續開機學習一整天後，再正式開啟異常警報機制。

### 3. 外觀重識別 (ReID) 的服裝混淆
*   **問題點**：目前的 ReID 使用 OpenCV 的 HSV 色彩直方圖。當兩位長者剛好都穿「深色衣服」且同時背對鏡頭時，色彩直方圖會無法區分兩人，導致 ByteTrack 的 ID 發生 Swap 錯亂。
*   **V5.0 解法**：如果 VRAM 還有空間，將 HSV 替換為輕量級的深度特徵提取器（如 **OSNet** 或 **MobileNet-V2** 提取 ReID 特徵），利用服裝紋理而非純顏色來區分。

### 4. 視覺語言模型 (VLM) 的效能瓶頸
*   **問題點**：生成式 AI 非常耗時。雖然已將抽幀間距 (interval) 調降為 30（每秒一張），但在 T600 上仍需耗費近一半的執行時間。
*   **V5.0 解法**：既然已有 Dual-HOI 雙軌防呆，基礎的人機互動已無需 VLM 插手。未來可考慮**移除 VLM 旁路**，或僅在「發生異常分數大於 0.8」的關鍵幀才觸發 VLM 進行緊急影像截圖敘述。

## 四、 給你的最後一句話

這個專案證明了**「演算法架構的設計，遠比無腦堆疊算力更重要」**。
請善用 `ModelGuard`，祝你好運！
— 系統開發與維護 AI Agent (2026 留)
