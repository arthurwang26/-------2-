# 內部一致性檢驗日誌 (V4.4 更新版)

本文檔記錄了開發期間如何利用內部檢測器防止各模組之間的輸出格式崩潰或狀態不一。

## V4.4 檢驗重點：雙模態資料流

### 1. HOI 雙軌輸出一致性
在 V4.4 中，`main.py` 會同時向 `clip_events` 寫入來自 MLP 與 CLIP 的結果。
*   **檢驗目標**：確保合併陣列時，兩者的 `type` 屬性不會覆蓋彼此（即保證 `HOI-MLP` 與 `HOI-CLIP` 標籤分明）。
*   **檢驗方法**：透過在 LLM 生成 Prompt 中列印出 `clip_context` 的 JSON 結構，確保 Qwen 大模型能順利區分資料來源。

### 2. VRAM 記憶體卸載一致性 (ModelGuard Check)
*   **檢驗目標**：`ModelGuard` 必須在模型轉換時徹底清除 4GB VRAM，尤其是當 CLIP 這個耗顯存的模型加入後。
*   **檢驗方法**：在 `utils/reset_memory.py` 埋設了 `torch.cuda.memory_allocated()` 的監控。在切換到 `SmolVLM2` 之前，必須保證 VRAM Allocated < 200MB，否則中斷管線。

### 3. Frame Caching 一致性
*   **檢驗目標**：因為原系統不保留 frame 以節省記憶體，加入 CLIP 後需要在 `main.py` 暫存 `sampled_frame`。
*   **檢驗方法**：確認 `middle_idx = len(frames) // 2` 不會發生 `IndexError`，即使影片因為 FPS 過低導致幀數不足，也會有防呆保護 (`if frames else None`)。

## 歷史已知問題處理狀態
- ✅ **身份覆蓋問題**：舊版 ID 跳號問題已在 V2 透過 Appearance ReID 完全解決。
- ✅ **ST-GCN Shape 不對齊**：V4 升級 MMAction2 後，內部已實作 (1, C, T, V) 張量對齊。
- ✅ **VLM 套件載入失敗**：已升級 `AutoModelForImageTextToText`，並確保 `SmolVLM2-256M` 穩定運作。
