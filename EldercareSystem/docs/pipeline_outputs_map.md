# EldercareSystem Pipeline 輸出對照表 (V4.4)

本文件列出系統執行 `main.py` 後，各個產出檔案的確切位置與用途。
為了保留每次 AI 演進的實驗結果，我們採用了多目錄架構：

## 輸出目錄結構演進
- `outputs/`：V4.0 基準測試結果。
- `outputs2/`：V4.3 演進測試結果。
- `outputs3/`：**目前 (V4.4 終極旗艦版)** 啟動的測試結果存放區。以下路徑皆以 `outputs3/` 為基準。

## 1. 大語言模型對比除錯報告 (llm_reports/)

這也是本系統 V4.4 最核心的學術亮點，所有 LLM 報告都會自動比對雙軌 HOI 的結果。

*   **`llm_reports/per_clip/`**：
    *   包含 18 支影片各自獨立的文字報告（例如 `clip01.txt`, `clip02.txt`）。
    *   內容包含三大區塊：`=== 感測器節點敘述 ===`、`=== VLM 視覺描述 ===`、`=== 誤差與對比分析 ===`。
*   **`llm_reports/王奶奶_dayX.txt` / `陳爺爺_dayX.txt`**：
    *   按照時間順序合併的單人行為日誌。
*   **`llm_reports/final_report.md`**：
    *   長照監控系統全局最終報告。包含跨天行為推理、異常預測、雙軌 HOI 與 ST-GCN 感測器盲區分析等終極統整。

## 2. 系統文件與知識圖譜 (system_docs/)

*   **`system_docs/knowledge_graph.cypher`**：
    *   這是一份可以匯入 **Neo4j 圖形資料庫** 的腳本。它定義了節點 (Person, Action, HOI_Event, Emotion, Day) 與關係 (DOING, FEELING, HAPPENED_ON)。

## 3. 情境渲染影片 (visuals/)

*   包含 `visuals/clip01_annotated.mp4` 等輸出。
*   這些影片直接將 YOLO BBox、RTMPose 骨架、以及 ST-GCN/Dual-HOI 的預測結果壓制在畫面上，適合做為學術展示 (Demo) 的直觀證據。
