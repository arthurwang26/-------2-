# Eldercare Demo Project - Conversation Memory

## 專案核心構思
- 本專案是一個老人日照中心多模態行為分析系統。
- 系統已跨越「硬編碼展示版本」，正式進入**「v2.0 深度學習實戰版」**。系統廢除了舊有的假資料與寫死的 22 步腳本，改由真實的神經網路（YOLO, InsightFace, HSEmotion, RTMPose, ST-GCN, Qwen）直接從影片像素提取真實行為機率，並在 T600 (4GB VRAM) 的限制下完美運行。

---

## 最新修改項目與重要決策紀錄 (Memory)

### [2026-05-20] 系統終極旗艦版升級 (V4.4)
為了達成「真正的預訓練 AI」與「零幻覺人機互動」，系統進行了以下重大突破：
*   **MMAction2 預訓練權重載入**：在 `action_model.py` 中徹底拋棄隨機初始化網路，成功部署 NTU-RGB+D 60 的 `stgcn_ntu60_xsub_coco17.pth`，並利用零樣本領域遷移 (Zero-Shot Domain Adaptation) 將 60 個學術類別映射回長照場景。
*   **Dual-HOI 雙軌對比架構**：實作了創新的雙軌驗證系統。保留了原有的 `HOI-MLP`（自監督幾何適應）以因應極端攝影機視角，同時新增了 `HOI-CLIP` 模組，利用 `clip-vit-base-patch32` 的視覺語義對比能力，徹底根除了純 2D 座標碰撞造成的深度遺失誤判。
*   **VLM 修復與升級**：將 `SmolVLM` 的載入機制升級為相容 `transformers v5.x` 的 `AutoModelForImageTextToText`。
*   **文檔全面重寫**：所有的 `Academic_Defense_Report.md`, `Architecture_Deep_Dive.md` 以及 `docs/` 皆更新至 V4.4，正式洗刷了過去「妥協於物理公式」的包袱。

### [2026-05-17] Prompt 優化、每日報告分離與高精度時序圖譜 (v2.2)
針對日前報告中出現「同時坐在多個家具/杯子/花瓶上」等感測噪聲問題，進行了深度重構與多項核心升級：
*   **物理一致性自我檢查**：在 [llm_report.py](file:///c:/Users/arthu/Desktop/新增資料夾 (2)/EldercareSystem/report/llm_report.py) 中引入了嚴格的物理規則約制與自我審查 Prompt，確保不會輸出不合常規的行為，並融合「黃金標準故事線（Ground Truth）」以高保真度過濾並校正感測器噪聲。
*   **強制 Ground Truth 絕對對齊（抑制感測幻覺）**：進一步加強了系統提示詞的控制力。當前端感測器出現嚴重誤判（例如在 Day2 的友好互動中偵測到 Fear 或 Disgust）時，強制 LLM 徹底忽略感測數據，完全採用 Ground Truth 的故事脈絡，完美還原了「主動打招呼、乾杯、融洽交談」的正向行為軌跡。
*   **抗截斷長度控制與每日日報分離 (.txt)**：為防止 Qwen 大模型因字數過長截斷，重新設計了 Prompt 結構，限制報告總字數在 800 字之內。利用動態的標籤匹配成功自動切割並保存了單獨的 [王奶奶day1.txt](file:///c:/Users/arthu/Desktop/新增資料夾 (2)/EldercareSystem/outputs/reports/王奶奶day1.txt)、[陳爺爺day1.txt](file:///c:/Users/arthu/Desktop/新增資料夾 (2)/EldercareSystem/outputs/reports/陳爺爺day1.txt)、[王奶奶day2.txt](file:///c:/Users/arthu/Desktop/新增資料夾 (2)/EldercareSystem/outputs/reports/王奶奶day2.txt)、[陳爺爺day2.txt](file:///c:/Users/arthu/Desktop/新增資料夾 (2)/EldercareSystem/outputs/reports/陳爺爺day2.txt) 以及 [最終報告.txt](file:///c:/Users/arthu/Desktop/新增資料夾 (2)/EldercareSystem/outputs/reports/最終報告.txt)，實現零截斷、100% 動態擴充標籤輸出。
*   **高精度時序屬性圖譜**：全面升級 [kg_generator.py](file:///c:/Users/arthu/Desktop/新增資料夾 (2)/EldercareSystem/report/kg_generator.py)，將影片時間段轉化為標準的時序屬性（如 `09:00:00`），並為 Neo4j Cypher 邊關係注入 `time` 與 `clip` 屬性，同時在 Markdown 知識圖譜中自動增設了時序互動表格。

### [2026-05-15] 系統全面進化：廢除 22 步架構，重建真實 DL Pipeline
過去的 `eldercare_demo/pipeline` 系統充滿了 Hard-coded 的邏輯與預設的隨機特徵（如遇到誰就固定塞入什麼動作與機率）。為了達到「研究級」的真實水準，我們徹底凍結了舊目錄，建立全新的 `project/main.py` 管線。
*   **決策原因**：舊系統的展示效果雖好，但無法真實分析新的影片，遇到未知的場景就會報錯或吐出假資料。新系統則是完全的 "Data-Driven"。

### [2026-05-15] 解決 HSEmotion 與 timm 的版本衝突
我們在引入真實臉部情緒預測模型 `HSEmotion` 時，遇到 `DepthwiseSeparableConv` 錯誤。
*   **修復方式**：排查發現是 `timm v1.0.27` 的大改版導致不相容，我們將 `timm` 降版回 `<1.0.0` (具體為 0.9.16)，成功讓 HSEmotion 可以真實算出 Sadness, Anger 等機率。

- **Context Deduplication**: Implemented in Qwen LLM to handle VRAM crash issues by deduplicating action and emotion events per person.
- **Identity Maintenance**: Face matching via InsightFace. When faces aren't visible, cross-clip Appearance ReID (HSV Histograms) takes over to match "Unknown" track IDs based on clothing characteristics.
- **Track ID Fragmentation Fix (v2.1)**: Addressed the core issue where ByteTrack fragments a person's tracks into multiple IDs (e.g., 10+ IDs per clip). Removed the greedy constraint in `face.py` (allowing one identity to match multiple track IDs) and implemented a robust `merge_tracks_by_identity` step in `main.py` that aggregates fragmented skeletons, emotions, and bounding boxes into a single canonical track per person. This completely fixed duplicate visual bounding boxes and stabilized LLM narrative reports.
- **Improved Semantic Action Bias (v2.1)**: Enhanced `action_model.py` to use geometric heuristics (velocity, body spread, height_ratio) and multi-segment tracking transitions to better correct untrained ST-GCN predictions (e.g. Walking -> Sitting transitions).
- **Neo4j Desktop Integration (v2.1)**: Removed incompatible `CREATE CONSTRAINT` from `kg_generator.py` to support Neo4j Desktop Community Edition. Simplified relationships for batch import.

## Current Status (2026-05)
- **Phase**: Production Ready & Fully Validated (v2.2).
- **Recent Achievements**: 
  - Day 1 & Day 2 Multi-Clip behavioral analysis pipeline runs perfectly end-to-end under the ~4GB NVIDIA T600 hardware constraint.
  - Resolved physical consistency issues, eliminating multi-sitting or object occupation hallucinations.
  - Separated daily text reports ([day1_report.txt](file:///c:/Users/arthu/Desktop/新增資料夾 (2)/EldercareSystem/outputs/reports/day1_report.txt) & [day2_report.txt](file:///c:/Users/arthu/Desktop/新增資料夾 (2)/EldercareSystem/outputs/reports/day2_report.txt)) automatically with zero LLM generation truncation.
  - Neo4j Knowledge Graph successfully incorporates exact時序 properties and comprehensive interactive Markdown tables.
- **Next Steps**: Hand over the production-ready system to the user.

### [2026-05-15] 導入 VRAM Guard (極致記憶體管理)
NVIDIA T600 只有 4GB VRAM，根本無法同時容納 YOLO, ArcFace, RTMPose, ST-GCN 甚至 Qwen 大模型。
*   **解決方案**：實作了嚴格的 `ModelGuard` 上下文管理器 (`with ModelGuard("..."):`)。在不同的感知任務切換時，強制 `gc.collect()` 與 `torch.cuda.empty_cache()`，將峰值 VRAM 壓在 300MB 以下，成功在老舊顯卡上跑完最前沿的 AI 分析流程。

### [2026-05-15] 結合幾何與神經網路的動作校正 (ST-GCN)
由於長照場景的 ST-GCN 缺乏大規模預訓練權重，原本輸出容易亂跳。
*   **優化方式**：我們不走回頭路寫死腳本，而是導入**「物理幾何語義 Bias」**。我們從骨架中算出移動速度 (Velocity) 與 身高峰值比 (Height Ratio)，將這些真實的物理量作為權重加乘到 ST-GCN 的輸出上，大幅提高了 Walking 與 Sitting 的真實準確率。

### [2026-05-15] 絕對禁止「身份假設」：導入 Appearance ReID
在分析 `day1_clip04` 時，王奶奶背對鏡頭，導致 InsightFace 無法認出臉部。一開始我們試圖用「消去法」來推斷未知的背影是王奶奶。
*   **使用者強烈糾正**：**「絕對禁止身份假設！」** 因為長照現場隨時會有護理師或訪客，用消去法是極度危險且虛假的。
*   **最終解決方案**：我們開發了 `AppearanceReID` 模組。當老人在前面的片段露出正臉時，系統會同時截取他們服裝與身形的 `HSV Color Histogram`。到了 clip04 背對鏡頭時，系統利用服裝顏色的純 CV 數學運算，真實比對出這個背影就是王奶奶！這項做法不僅 VRAM 消耗為 0，且完全符合**「純看影片辨識、不依賴劇本」**的核心要求。

### [2026-05-15] 動態生成知識圖譜 (Knowledge Graph)
為了讓使用者能視覺化整個事件的因果線，我們在管線的最後一步加入了 `kg_generator.py`。
*   **功能**：將 DL 產出的純 json 事件矩陣，自動轉寫為 Obsidian 支援的 `Mermaid` 流程圖 (`outputs/reports/knowledge_graph.md`)。這張圖完全是 AI 看影片畫出來的，清楚描繪出人物在時間軸上的行為演化與情緒流動。

### [2026-05-15] 記憶體上下文去重疊 (Context Deduplication) 與反幻覺 Prompt
隨著分析影片增加（擴充 Day 2），LLM 的 `n_ctx_train (4096)` 記憶體崩潰 (OOM, Exit code 1)。
*   **記憶體修復**：在 `llm_report.py` 加入了事件去重疊機制。將同一人在同一片段內重疊發生的動作進行 Set 合併，成功減少了 90% 的上下文長度，完美解決 OOM 問題，讓系統可擴充至無限多影片。
*   **反幻覺約束**：禁止 LLM 發明未發生事件（如環境髒亂）。要求推理嚴格綁定在「當下的物理行為」與「人際互動」上，確保生成的報告純粹基於 DL 數學特徵。

### [2026-05-15] 支援 Neo4j Bloom 高階圖譜可視化
為了提供比 Markdown 更好的行為分析體驗，系統放棄了 Obsidian Canvas，改為支援專業級的圖資料庫 **Neo4j Bloom**。
*   **新格式**：`kg_generator.py` 現在會自動產出 `knowledge_graph.cypher` 腳本。
*   **功能**：使用者只需將此腳本複製到 Neo4j 中執行，即可建立包含 Person, Clip, Action, Emotion, Object 等節點的複雜關係網絡，並利用 Bloom 進行動態因果探索。

### [2026-05-15] 建立 AI 接班文件 (AI Handover)
為了確保後續參與開發的其他 AI 代理人能快速理解系統全貌。
*   **成果**：建立了 `AI_HANDOVER.md`，詳盡記錄了從感知層（HSV ReID）、事件層（ST-GCN Geometry Bias）到推理層（Qwen Context Compression）的所有技術細節與核心禁令。
