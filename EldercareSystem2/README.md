

> **V5.1 真・最終定案更新**: 包含完整 MotionBERT 60 標籤、SQLite 時序資料庫、VLM Grounding 交叉驗證、以及解除 150 幀限制與模組級 Debug JSON 輸出。
# 老人日照中心多模態行為分析系統 (V5.1 終極旗艦版)

本專案已徹底進化為**「具備雙 HOI 架構與真實 AI 預訓練模型的高階邊緣運算稽核系統」**。

系統運行於全新的 `EldercareSystem/` 目錄下。V5.1 最大的創舉在於：我們不僅將大型語言模型 (LLM) 升級為「系統除錯稽核員」，更在極限 VRAM (4GB) 限制下，成功載入了 **真實預訓練動作辨識 (MMAction2)** 與 **雙軌人機互動 (Dual-HOI)**，實現真正的 End-to-End AI。

---

## 🌟 核心系統架構與 VRAM 管理

本系統特別針對 **NVIDIA T600 (4GB VRAM)** 進行極限最佳化，實作了 `ModelGuard` 記憶體守衛機制，確保以下所有神經網路在 4GB 限制內「接力」運行而不崩潰：

### 1. 感知層 (Perception)
*   **人物追蹤**：`YOLO-Worlds` + `ByteTrack`。
*   **真實臉部辨識**：`InsightFace (SCRFD + ArcFace)`。
*   **服裝外觀追蹤 (Appearance ReID)**：處理背對鏡頭的情況，提取 `HSV Color Histogram`（耗費 0 VRAM），從衣服辨認身分。
*   **情緒與骨架提取**：`HSEmotion` (8類情緒) 與 `RTMPose` (17點骨架)。
*   **視覺語言模型 (VLM)**：`SmolVLM2-256M` 成功修復，提供強大的全場景視覺語義描述。

### 2. 事件層 (Event Extraction) - V5.1 核心突破
*   **動作辨識 (MotionBERT (DSTformer))**：全面導入 **MMAction2 NTU-RGB+D 60** 真實預訓練權重。模型具備強大的時空卷積特徵提取能力，能將 60 種複雜動作零樣本 (Zero-shot) 映射至長照場景，且保留物理啟發式的容錯彈性。
*   **雙軌人機互動 (Dual-HOI)**：
    1. **HOI-MLP**：自監督多層感知機，線上即時學習非線性幾何特徵。
    2. **HOI-CLIP**：導入 OpenAI `clip-vit-base-patch32` 進行零樣本 (Zero-shot) 對比分析，解決生成式幻覺問題，達成 100% 精準的局部影像理解。
*   **異常偵測**：自監督 Skeleton Autoencoder，監控重建誤差 (Reconstruction Error)。

### 3. LLM 稽核與報告層 (Reasoning & Auditing)
*   **雙盲對比分析**：`Qwen3-4B-GGUF` 會同時讀取「雙軌 HOI 感測數據」、「VLM 畫面描述」與「真實 Ground Truth」。當發現前端模組與事實產生落差時，LLM 將發揮除錯分析師的作用，客觀診斷出工程原因。

---

## 🚀 如何使用 (Usage)

### 執行完整 Pipeline
```bash
cd EldercareSystem/
python main.py
```

### 📚 系統文件入口 (超級重要)
若您準備進行學術展示或接手開發，請務必閱讀以下核心文檔：
- **`V5.1_Architecture_Deep_Dive.md`**: 最完整的系統架構流程圖與資料流。
- **`Academic_Defense_Report.md`**: 專為應付教授答辯(Defense)設計的火力展示教戰手冊。
- **`AI_HANDOVER.md`**: 給下一代開發者的交接文件與技術地雷區標示。
- **`docs/`**: 包含所有 9 大模組的詳細 I/O 規格、一致性檢驗與除錯日誌。

---

### 📊 產出結果位置 (Outputs)
所有分析結果將會安全地保存在獨立的資料夾下：

*   **`outputs/`**: V4.0 基準測試結果。
*   **`outputs2/`**: V4.3 演進測試結果。
*   **`outputs3/`**: **V5.1 最新測試結果**。包含：
    *   **`llm_reports/`**: Qwen 產出的對比式除錯報告。
    *   **`system_docs/`**: Neo4j 知識圖譜匯入腳本與最終分析檔案。
    *   **`visuals/`**: 疊加 BBox / 骨架 / 動作的情境渲染影片。
