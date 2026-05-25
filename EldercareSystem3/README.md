> **V6.0 真・最終定案更新**: 全面廢棄實驗室骨架模型，導入 Kinetics-400 RGB 連續動作模型 (VideoMAE)，解決 Domain Gap，並建立 `shared_data` / `shared_weights` 共用架構！
# 老人日照中心多模態行為分析系統 (V6.0 終極旗艦版)

本專案已徹底進化為**「解決真實環境 Domain Gap 的高階邊緣運算稽核系統」**。

系統運行於全新的 `EldercareSystem3/` 目錄下。V6.0 最大的創舉在於：我們發現傳統基於 NTU-60 訓練的骨架模型無法適應真實長照場景的靜態行為，因此全面改採 **VideoMAE (Kinetics-400) RGB 連續截圖模型**。同時，我們大幅重構了專案架構，將百MB級的模型與影片抽離至共用資料夾，保持系統的絕對輕量與乾淨。

---

## 🌟 核心系統架構與 VRAM 管理

本系統特別針對 **NVIDIA T600 (4GB VRAM)** 進行極限最佳化，實作了 `ModelGuard` 記憶體守衛機制，確保以下所有神經網路在 4GB 限制內「接力」運行而不崩潰：

### 1. 資源共用層 (Shared Storage)
*   **`shared_data/`**: 集中存放所有原始監視器影片 (`.mp4`)。
*   **`shared_weights/`**: 集中管理所有 YOLO、Face、MotionBERT、VideoMAE 與 Qwen LLM 權重檔案。

### 2. 感知層 (Perception)
*   **人物追蹤**：`YOLO-Worlds` + `ByteTrack`。
*   **真實臉部辨識**：`InsightFace (SCRFD + ArcFace)`。
*   **服裝外觀追蹤 (Appearance ReID)**：處理背對鏡頭的情況，提取 `HSV Color Histogram`（耗費 0 VRAM），從衣服辨認身分。
*   **情緒提取**：`HSEmotion` (8類情緒)。
*   **視覺語言模型 (VLM)**：`SmolVLM2-256M` 負責場景語義描述，並在 V6.0 中特別強化其觀察「人際社交互動 (talking, arguing)」的能力。

### 3. 事件層 (Event Extraction) - V6.0 核心突破
*   **動作辨識 (VideoMAE RGB Model)**：全面廢棄 MotionBERT 骨架模型！改為直接擷取追蹤 Bounding Box 的 RGB 連續截圖，送入 Hugging Face 的 `videomae-base-finetuned-kinetics`。結合專門設計的「長照動作白名單過濾機制」，徹底消除骨架模型在靜態行為下亂猜 (如 handshaking) 的荒謬問題。
*   **雙軌人機互動 (Dual-HOI)**：
    1. **HOI-MLP**：自監督多層感知機，線上即時學習非線性幾何特徵。
    2. **HOI-CLIP**：導入 OpenAI `clip-vit-base-patch32` 進行零樣本 (Zero-shot) 對比分析。
*   **時序記憶體資料庫**：內建 `SQLite Time-Series Database` 永久記憶所有人物的歷史特徵。

### 4. LLM 稽核與報告層 (Reasoning & Auditing)
*   **全局因果推理分析**：`Qwen3-4B-GGUF` 會讀取精準的動作、情緒、HOI 以及具備社交互動描述的 VLM 輸出，自動跨天推理出「爭吵、冷戰、和好」等深層社會關係轉變。

---

## 🚀 如何使用 (Usage)

### 執行完整 Pipeline
```bash
cd EldercareSystem3/
python main.py
```

### 📚 系統文件入口 (超級重要)
若您準備進行學術展示或接手開發，請務必閱讀以下核心文檔：
- **`V6.0_Architecture_Deep_Dive.md`**: 最完整的系統架構流程圖與資料流。
- **`AI_HANDOVER.md`**: 給下一代開發者的交接文件與技術地雷區標示 (包含從 V5 到 V6 的心路歷程)。
- **`docs/`**: 包含所有 9 大模組的詳細 I/O 規格、一致性檢驗與除錯日誌。

---

### 📊 產出結果位置 (Outputs)
所有分析結果將會安全地保存在獨立的資料夾下：

*   **`outputs4/`**: **V6.0 最新測試結果**。包含：
    *   **`llm_reports/`**: Qwen 產出的結構化報告。
    *   **`system_docs/`**: Neo4j 知識圖譜與最終分析檔案。
    *   **`debug/`**: JSON 格式的系統內部除錯輸出 (`events.json`, `actions.json`)。
    *   **`visuals/`**: 疊加 BBox 與動作的渲染影片。
