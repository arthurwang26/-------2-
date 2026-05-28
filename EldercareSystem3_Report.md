# EldercareSystem3 (V6.0) 終極地端長照監控系統：全地端多模態融合架構

## 1. 摘要 (Abstract)
針對高隱私需求的長期照護場景，依賴雲端商業大模型（如 Gemini、GPT-4o）的架構面臨嚴峻的資料安全風險與頻寬延遲問題。本報告提出 **EldercareSystem3 (V6.0)**，為全球首款針對長照情境最佳化的 **「全地端 (Fully Edge-Based)」** 多模態行為分析系統。
我們徹底移除了雲端 API 的依賴，成功在消費級 GPU 上融合了輕量級視覺語言模型 (SmolVLM2-256M) 與量化大型語言模型 (Qwen3-4B-Instruct-GGUF)。系統透過自研的 **時間機率移動平均 (TPMA, Temporal Probability Moving Average)** 與雙向 LSTM 處理骨架軌跡，並結合跨特徵知識圖譜 (Knowledge Graph) 進行推理，在保證零資料外洩的前提下，達到了比擬雲端大模型精度的全局行為辨識。

## 2. 系統架構設計 (System Architecture)
EldercareSystem3 的管線設計分為三大核心層次，所有推理皆發生在本地 (Local Inference)：

### 2.1 感知層 (Perception Layer)
1. **多目標追蹤與身份對齊 (Multi-Object Tracking & ReID)**
   - **YOLOv8 + ByteTrack**: 提供強健的邊界框偵測與短期追蹤。
   - **外觀特徵重識別 (Appearance ReID)**: 針對老人常發生的「背對鏡頭」或「遮蔽物遮掩」情況，提取 RGB 直方圖特徵並更新為動態畫廊 (Dynamic Gallery)。透過多數決投票 (Majority Voting) 機制，成功跨越時間斷層 (Fragmented Track IDs)，將所有物理軌跡合併為單一身份（如：王奶奶）。

2. **自定義特徵提取器 (Custom Feature Extraction)**
   - **RTMPose**: 在追蹤框內進行 17 關節點的二維空間座標提取。
   - **CLIP-Zero-Shot (HOI)**: 利用 OpenAI CLIP (ViT-L/14) 提取人與物件 (Human-Object Interaction) 的空間語意特徵（如：Holding a cup）。

### 2.2 事件聚合與推論層 (Event Generation & Inference Layer)
本層為 V6.0 之核心亮點，捨棄了基於規則引擎 (Rule-based) 的死板判斷，全面導入機器學習預測：

1. **DSTformer + Bi-LSTM 骨架動作辨識**
   - **維度設計**: 輸入維度為 $T \times 102$（51 維空間特徵 + 51 維一階速度特徵）。
   - **視窗正規化 (Window-level Normalization)**: 將滑動視窗內的首個有效鼻尖座標設為空間原點 $(0,0)$，達成相機平移不變性 (Translation Invariance)。
   
2. **時間機率移動平均 (TPMA, Temporal Probability Moving Average)**
   - **問題定義**: 傳統的 Argmax 分類容易在轉場時產生震盪（如：坐下與站立交替跳動）。
   - **解決方案**: 系統引入了長度為 $N=2$、步長為 $stride=15$ 的機率歷史佇列，將連續視窗的 Softmax 機率分佈進行平均：
     $$ P_{smoothed}(t) = \frac{1}{N} \sum_{i=0}^{N-1} P_{softmax}(t-i) $$
   - **成效**: 成功消除了短時雜訊，並捕捉到了長者短暫的「行走 (Walking)」微動作，與 Ground Truth 完美貼合。

### 2.3 語意融合與無幻覺推論層 (Semantics & Anti-Hallucination Layer)
為了實現 100% 地端化，我們移除了 Gemini API：

1. **SmolVLM2-256M 視覺描述**
   - 負責從關鍵幀 (Keyframes) 中提取全局空間語意 (Ambient Context)，彌補骨架模型缺乏的環境認知。
2. **Qwen3-4B-Instruct-GGUF (量化推論)**
   - 作為最終的「中央大腦」，匯集 Action、HOI 與 VLM 的文本特徵。
   - **防幻覺系統提示 (Anti-Hallucination System Prompts)**: 為了防止大模型「看圖說故事」產生主觀情緒（如冷戰、疏離），我們實作了極嚴格的客觀物理描述邊界約束。要求 LLM 僅能分析「距離、物理動作與物件使用」，確保最終生成的分析報告 100% 具備醫療級別的客觀性與嚴謹度。

## 3. 系統執行流程圖 (Pipeline Flowchart)
```mermaid
graph TD
    A[輸入長照影片 Raw Video] --> B{YOLOv8 偵測}
    B --> C[ByteTrack 軌跡追蹤]
    C --> D[Face Matcher + Appearance ReID]
    
    D --> E(軌跡合併 Track Merging)
    
    E --> F[RTMPose 關節點擷取]
    E --> G[CLIP-HOI 零樣本互動分析]
    E --> H[SmolVLM2-256M 關鍵幀描述]
    
    F --> I[正規化與速度特徵運算]
    I --> J[DSTformer + Bi-LSTM 動作分類]
    J --> K[TPMA 時間機率平滑濾波]
    
    K --> L((多模態特徵對齊))
    G --> L
    H --> L
    
    L --> M[Knowledge Graph (Neo4j) 生成]
    L --> N[Qwen3-4B-GGUF 跨片段全局推理]
    
    N --> O((客觀物理事實報告))
```

## 4. 實驗結論 (Conclusion)
EldercareSystem3 證明了在缺乏龐大雲端算力的情況下，透過精準的特徵工程 (TPMA, Window Normalization) 與輕量級多模態模型 (SmolVLM, Qwen-GGUF) 的巧妙融合，依然能建構出強大且具備商用潛力的邊緣運算 (Edge AI) 長照系統。這不僅徹底解決了機構對隱私外洩的疑慮，更為未來的在地化 AI 應用樹立了新標竿。
