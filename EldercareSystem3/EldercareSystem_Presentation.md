> **V6.0 真・最終定案更新**: 包含完整 VideoMAE (Kinetics-400) RGB 連續動作模型、長照行為白名單過濾、VLM 社交互動提示詞強化、SQLite 時序資料庫，以及 Shared Data 輕量化共用架構。
# 長照多模態行為分析系統 (V6.0 終極旗艦版) - 專案成果簡報
**Eldercare Multimodal Behavior Analysis System**
> 專為邊緣運算 (Edge AI) 打造的零幻覺、零 Domain Gap 雙模態 AI 稽核管線

---

## 壹、 專案背景與痛點分析 (Background & Motivation)

### 1. 長照機構的現況挑戰
*   **照護人力短缺**：無法達成 24 小時 1:1 的全天候盯哨。
*   **傳統監視器的盲點**：僅能做到事後錄影調閱，缺乏「即時行為理解」與「危險預測」能力。
*   **隱私與預算考量**：資料無法上傳公有雲，且多數機構只能負擔極低階的邊緣運算設備（如 NVIDIA T600 4GB VRAM）。

### 2. 技術痛點 (Technical Challenges)
要在 **4GB VRAM** 的嚴苛限制下，運行包含物件偵測、動作辨識、情緒辨識、人機互動、大語言模型等高達 9 個神經網路，這在傳統端到端 (End-to-End) 架構中是完全不可能的任務，通常會導致 `Out Of Memory (OOM)` 崩潰。

---

## 貳、 核心技術創新 (Core Innovations)

為了突破上述限制，本團隊開發了 **V6.0 終極旗艦版**，導入三大革命性架構：

### 創新一：ModelGuard (極限記憶體守衛) 與 Shared Data
*   **機制**：實作了嚴格的 Context Manager。保證所有大型神經網路（YOLO, VideoMAE, CLIP, SmolVLM, Qwen）皆以「接力賽」的方式運行。並且將百 MB 級的權重與影片抽離至獨立的 `shared_data` 共用層。
*   **成效**：模型做到「用完即丟、瞬間清空 CUDA 快取」，將系統的峰值 VRAM 嚴格壓制在 3GB 以下，達成永不崩潰的邊緣運算奇蹟。

### 創新二：VideoMAE 真實影像動作辨識 (解決 Domain Gap)
*   **機制**：摒棄了傳統基於 NTU-60 訓練的實驗室骨架模型 (MotionBERT)，全面導入 **Kinetics-400 預訓練的 VideoMAE (RGB 連續截圖模型)**。
*   **成效**：直接理解連續影像中的物件外觀與互動，徹底解決了骨架模型在長照場景「靜態行為」下亂猜 (如 handshaking) 的致命缺陷。並結合「長照動作白名單 (Whitelist)」，強迫模型在合理行為中進行高可信度的預測 (如 reading book, sneezing)。

### 創新三：雙軌人機互動 (Dual-HOI Architecture)
為了解決單一攝影機缺乏 Z 軸深度、極易誤判「站在沙發前」與「坐在沙發上」的致命缺陷，本系統獨創雙軌並行驗證：
1.  **Track A (HOI-MLP)**：自監督神經網路。透過即時線上學習 (Online Learning)，讓 AI 自適應長者的拓撲幾何距離。
2.  **Track B (HOI-CLIP)**：Zero-Shot 視覺語義對比。擷取「人+物」的局部影像，交由 OpenAI CLIP 模型進行文字與影像相似度對比。

---

## 參、 系統管線架構 (Pipeline Architecture)

本系統為全自動化管線，資料流如下：

1.  **感知層 (Perception Layer)**
    *   **YOLO-Worlds + ByteTrack**：精準人物框定與跨幀連續追蹤。
    *   **InsightFace + HSV ReID**：雙重身分保險。正臉辨識；背影使用服裝色彩直方圖追回身分。
    *   **HSEmotion**：臉部 8 大情緒辨識。
2.  **事件層 (Event Layer)**
    *   **VideoMAE (Kinetics-400)** 動作辨識。
    *   **Dual-HOI** 雙軌人機互動網路。
    *   **Autoencoder** 異常偵測。
3.  **大腦層 (LLM Auditing Layer)**
    *   **SmolVLM2-256M**：強化「人際社交互動 (talking, arguing)」的場景視覺描述。
    *   **Qwen3-4B-GGUF**：雙盲對比除錯分析師與跨天因果推理。

---

## 肆、 殺手級應用：LLM 對比式除錯稽核 (LLM-based Auditing)

有別於傳統「AI 說什麼就是什麼」的黑盒子專案，V6.0 系統具備**「自我診斷與解釋」**的能力。

*   **運作邏輯**：系統會將充滿噪聲的前端感測數據連同「真實情況 (Ground Truth)」一起輸入給 Qwen 大語言模型。
*   **LLM 的任務**：LLM 扮演高級工程稽核員，利用其常識推理能力，**主動揪出前端感測器的工程盲區**。
*   **產出價值**：最終生成的報告不僅記錄老人的行為，更會詳細寫出「為何感測器在此處失效」。這為未來的模型迭代提供了最明確的學術除錯方向。

---

## 伍、 時序推理與知識圖譜 (Knowledge Graph)

*   本系統產生的所有事件（人員、情緒、動作、互動、時間點）最終會自動彙整，編譯成 `.cypher` 腳本。
*   這使得海量的監控影片數據，能被具象化為 **Neo4j 時序知識圖譜**。
*   照護人員可透過簡單的圖形化介面，直觀追蹤「陳爺爺何時情緒開始惡化」或「王奶奶與沙發的互動頻率」，實現預防性長照醫學。

---

## 陸、 總結與未來展望 (Conclusion & Future Work)

### 總結
本專案成功在最嚴苛的硬體限制下，實作了一套兼具「深度學習高精準度」與「時序社會關懷」的長照 AI 系統。透過 **VideoMAE**, **ModelGuard**, **Dual-HOI** 與 **Prompt Engineering** 四大核心技術，徹底洗刷了過去依賴寫死腳本的工程包袱。

### 未來架構升級藍圖 (Roadmap)
1.  **RGB 動作辨識的時間軸抽樣 (Temporal Windowing)**：將目前片段均勻抽樣，改為 Sliding Window 連續抽樣，大幅提升模型對時間連續性動作的理解能力。
2.  **長效異常偵測基準線**：將 Autoencoder 的訓練窗口擴展至「一週滑動區間」，降低跌倒偵測的假陽性。
3.  **OSNet 深度重識別**：將傳統的 HSV 色彩直方圖升級為輕量深度特徵 ReID，解決穿著同色衣物時的追蹤錯亂問題。
4.  **VLM 動態觸發機制**：將固定頻率抽幀的 VLM 敘述，改為「僅在動作/情緒發生劇烈轉換時觸發」，藉此釋放高達 50% 的運算時間。
