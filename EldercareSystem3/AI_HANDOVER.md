> **V6.0 真・最終定案更新**: 包含完整 VideoMAE (Kinetics-400) RGB 連續動作模型、長照行為白名單過濾、VLM 社交互動提示詞強化、SQLite 時序資料庫，以及 Shared Data 輕量化共用架構。
# 給下一代開發者的交接文件 (AI Handover Document) - V6.0

你好，接手的工程師。
這個系統經歷了從實驗室骨架模型 (V5) 轉變為真實世界連續影像模型 (V6) 的重大架構重構。這個系統已經進化到 **V6.0 終極旗艦版**，請在修改任何程式碼之前，詳讀本交接文件，這會為你省下無數個熬夜除錯的夜晚。

## 一、 系統目前的極限與架構分佈

這個系統最核心的靈魂不是模型多強大，而是**它能在極度受限的 VRAM (4GB) 限制下，把 9 個神經網路無縫接力跑完**，並且具備非常聰明的共用架構。
1. **千萬不要拔掉 `ModelGuard`**。只要拔掉，系統絕對 OOM (Out Of Memory)。
2. **`shared_data` 與 `shared_weights`**：不要把大型 `.mp4` 影片或動輒數百 MB、幾 GB 的 `.pt` / `.bin` 模型權重丟進專案資料夾！系統的 `config.py` 已經設定好讀取上一層目錄的共用資料夾，這使得未來的 `EldercareSystem4` 可以直接掛載這些資源，瞬間完成系統初始化。

## 二、 各模組踩雷指南與 V6 重大變革

### 1. `events/rgb_action_model.py` (V6 新增的 VideoMAE)
*   **V6 變革原因**：我們在 V5 使用 MotionBERT 骨架模型，發現它對於「靜態姿勢 (如老人安靜坐著)」或是「缺乏劇烈位移的行為」毫無辨識能力，會狂噴幻覺 (如 handshaking)。
*   **雷區 (Context Loss)**：VideoMAE 是直接吃追蹤到的 Bounding Box 裁切影像。因為裁得太剛好，模型看不見旁邊的杯子或沙發，曾經把「老人從沙發站起」誤認為 `baby waking up`。
*   **解法 (Whitelist Filtering)**：為了避免上述問題並保持 BBox 緊密（防止拍到旁人干擾），我們在程式碼裡寫死了 `RELEVANT_ACTIONS` 白名單。模型輸出的 logits 會經過這個 Mask 過濾，強迫它在長照合理的 70 個動作中選一個。**如果你發現模型辨識不出某個動作，請先檢查它是否在白名單內！**

### 2. `report/vlm_caption.py` (視覺語言模型與社交互動)
*   **現狀**：使用 `SmolVLM2-256M`。
*   **V6 變革**：VideoMAE (Kinetics-400) 缺乏「交談 (Talking)」、「爭吵 (Arguing)」等詞彙，我們修改了 VLM 的 Prompt，強迫它在 Caption 中描寫 `social interactions`。請不要用傳統的寫死規則去判斷吵架，**完全交給 VLM 與 Qwen 的上下文推理**才是正道。

### 3. `events/hoi_model.py` (自監督 MLP)
*   **現狀**：保留了這個基於幾何的 MLP 作為 HOI 雙軌對比 (與 HOI-CLIP 對照)。
*   **設計概念**：利用距離特徵產生 Pseudo-labels 即時訓練 (Online Learning)。推論前跑 epochs 是正常的。

### 4. `report/llm_report.py` (最終報表生成)
*   **角色定位**：Qwen 吃「Ground Truth」與「Sensor Nodes」。
*   **Prompt 修改警告**：請絕對不要修改提示詞裡的 `VLM 獨立原則` 與 `最後才判斷 Ground Truth`。如果你讓 LLM 提早看到答案，它就不會去認真分析前端感測器為什麼算錯，這會毀掉系統「自動診斷與工程除錯」的核心價值。

## 三、 V6.0 殘留的已知問題與未來路線圖 (Roadmap)

雖然 V6 解決了動作辨識的巨大 Domain Gap，但仍有以下優化空間：

### 1. RGB 動作辨識的時間軸抽樣 (Temporal Windowing)
*   **問題點**：目前 `rgb_action_model.py` 是將一整個影片片段 (Clip) 的 BBox 影像「均勻抽樣」成 16 幀。如果 Clip 長達一分鐘，這種抽樣會導致動作在時間軸上嚴重扭曲，讓模型無法理解連貫性。
*   **未來解法**：實作 Sliding Window (滑動視窗) 機制。每 2~3 秒取連續的 16 幀進行動作推論，然後使用 Voting (多數決) 或平均池化來決定整個片段的主要動作。

### 2. 異常跌倒偵測 (Autoencoder) 的假警報
*   **問題點**：目前 `AnomalyDetector` 的基準線過短（僅第一支影片），導致假陽性高。
*   **未來解法**：延長基線緩衝區 (Baseline Buffer)。將訓練資料擴展到「過去一週的滑動區間 (Sliding Window)」。

### 3. 視覺語言模型 (VLM) 的效能瓶頸
*   **問題點**：在 T600 上非常耗時。
*   **未來解法**：僅在「發生異常分數大於 0.8」或「RGB 行為辨識發生劇烈轉換」的關鍵幀才觸發 VLM 進行敘述。

## 四、 給你的最後一句話

這個專案從零開始，經歷了無數次架構推翻，最終證明了**「演算法架構的設計與合適的特徵提取，遠比無腦堆疊算力更重要」**。
請善用 `ModelGuard`、`Shared Data` 與 `Prompt Engineering`，祝你好運！
— 系統開發與維護 AI Agent (2026 留)
