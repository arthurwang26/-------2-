# 第三代系統 (Eldercare System 3)：客製化架構、雙重驗證與時序防毒 (巔峰之作)

## 摘要 (Abstract)
在未受限的長照環境中進行連續、強健且具備醫療客觀性的行為分析，需要徹底拋棄通用的開源模型。本文提出 **EldercareSystem3**，這是我們多模態行為分析架構的巔峰之作。我們提出了一個高度客製化的 `Transformer + Bi-LSTM` 動作辨識模型，並輔以嶄新的「時序機率移動平均 (TPMA)」與「視窗級座標平移不變性 (Window-Level Translation Invariance)」正規化技術。為了對抗身份毒化，我們導入了「多數決身份重識別 (Majority Voting ReID)」機制。此外，我們建構了整合 CLIP 與 BLIP 的雙重人機互動 (HOI) 驗證管線，並與 LLM/VLM 的全局語境進行深度融合。消融實驗證明，EldercareSystem3 在時序動作解構上達到了商業級的極致精準度。

---

## 1. 系統架構與詳細流程圖
System 3 是徹底重構的完全體。系統捨棄了盲目的模型堆疊，導入了大量的特徵正規化與多重驗證防錯機制 (Error-Correction Mechanisms)。

```mermaid
graph TD
    A[輸入連續監視器影像] --> B(YOLOv8 / DeepOCSORT 盲追蹤)
    
    %% 身份多數決分支
    B -->|匿名軌跡 (Anonymous Tracks)| C1(InsightFace 擷取臉部特徵)
    C1 -->|統計整段軌跡的所有投票| C2{Majority Voting ReID}
    C2 -->|指派最高票身份| C3[防毒：最終確定身份]
    
    B -->|軌跡 BBoxes| D{深度特徵解構與驗證}
    
    %% 骨架動作分支 (客製化架構 + TPMA)
    D -->|提取| E1(YOLO-Pose 17關節點)
    E1 -->|計算物理速度 (Velocity)| E2(視窗級座標平移不變性正規化)
    E2 -->|102維特徵序列| E3(Transformer + Bi-LSTM 客製模型)
    E3 -->|Raw Probabilities| E4(時序機率移動平均 TPMA)
    E4 -->|平滑化特徵| F1[精準連續動作狀態 (Action)]
    
    %% HOI 雙重驗證分支
    D -->|比對| G1(BBox 交集區域裁切)
    G1 -->|Proposal| G2{OpenAI CLIP 粗篩}
    G2 -->|若機率 > 閾值| G3{Salesforce BLIP 視覺問答}
    G3 -->|Yes / No 驗證| F2[零幻覺人機互動 (HOI)]
    
    %% 全局語境雙引擎
    A -->|抽樣| H1(SmolVLM2 區域語意)
    A -->|全影片| H2(Gemini Video API 全局故事線)
    H1 & H2 --> F3[雙引擎 Ambient Context]
    
    %% 結構化與最終輸出
    C3 & F1 & F2 & F3 --> I(Knowledge Graph Generator)
    I -->|Neo4j Cypher & Markdown| J(Qwen3-4B 地端大語言模型)
    J -->|嚴格約束情緒的主觀腦補| K[終極長照事實觀測報告]
    
    classDef highlight fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    class C2,E2,E4,G3 highlight;
```

## 2. 核心技術與研究方法 (Methodology)

### 2.1 視窗級正規化與 Transformer + Bi-LSTM 架構
為了解決 MotionBERT 的尺度變異問題，我們自行訓練了針對長照 7 種核心動作的輕量化模型。
**特徵表徵 (Feature Representation)**：對於每一幀 $t$，提取 51 維的空間座標 $S_t$ (17 關節 $\times$ 3)。為了明確編碼時間動態，我們在做任何空間位移前，先計算絕對物理速度 $V_t = S_t - S_{t-1}$。將兩者串聯為 $F_t = [S_t, V_t] \in \mathbb{R}^{102}$，這完美保留了「跌倒」或「坐下」時極具鑑別度的重力加速度特徵。
**視窗級座標平移不變性 (Window-Level Translation Invariance)**：我們以滑動視窗 $W = \{F_{t}, \dots, F_{t+59}\}$ 進行掃描。對於每個視窗，將第一幀的有效鼻子座標 $(N_x, N_y)$ 扣除於該視窗所有的空間座標：
$$ \hat{S}_{t+k} = S_{t+k} - (N_x, N_y), \quad \forall k \in [0, 59] $$
此舉讓所有動作的起始點完美對齊原點，消除了鏡頭遠近造成的絕對座標誤差，同時又保留了相對位移的趨勢。

### 2.2 時序機率移動平均 (Temporal Probability Moving Average, TPMA)
為了徹底消除連續預測時發生的判定震盪 (Action Oscillation)，且不依賴人工撰寫的 `if-else` 規則，我們導入了 TPMA 數學平滑技術。
令 $P_t \in \mathbb{R}^7$ 為神經網路對於視窗 $t$ 輸出的原始 Softmax 機率分佈。我們不直接對其進行 $\arg\max$，而是維護一個長度為 $N$ (例如 $N=3$) 的歷史佇列，計算其加權平均：
$$ \hat{P}_t = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i} $$
$$ \text{Action}_t = \arg\max(\hat{P}_t) $$
這項純數學操作確保了短暫的雜訊 (例如因為轉頭導致單一視窗給出 55% 的 Standing 誤判) 會被相鄰視窗強大的真實狀態 (例如 99% 的 Sitting) 徹底壓制攤平，輸出的動作邊界極度平穩滑順。

### 2.3 臉部多數決防毒 (Majority Voting ReID)
我們徹底廢除了逐幀 (Frame-by-frame) 賦予身份的脆弱機制。`DeepOCSORT` 首先進行完全匿名的盲追蹤 (Blind Tracking) 生成軌跡。接著，`InsightFace` 對軌跡中每一幀合格的臉部提取 512 維 Embedding 並進行投票。整條軌跡的最終身份 (Canonical Identity) 由最大概似估計 (Maximum Likelihood Estimation) 決定，也就是最高票者得勝。此機制從數學底層免疫了瞬間的臉孔誤判污染。

### 2.4 雙重驗證 HOI 管線 (CLIP + BLIP)
為了解決零樣本幻覺，我們設計了「提案-驗證」雙階段管線 (Proposal-Verification Pipeline)。
首先由 `CLIP` 進行高召回率 (High Recall) 的初篩，提出互動假設 (如：Sitting_On)。若機率過門檻，再將局部影像送入 `Salesforce BLIP` 進行高精準度 (High Precision) 的視覺問答 (VQA)："Is the person sitting on a chair?"。唯有兩大模型同時給出肯定判斷，知識圖譜才會將該人機互動事件立案，達成近乎零誤報的優異表現。
