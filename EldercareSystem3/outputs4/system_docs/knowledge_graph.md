## 🕸️ 系統因果與行為知識圖譜 (Knowledge Graph)
> [!info] 這是基於真實 Deep Learning 萃取特徵建立的時序行為關係圖，支援 Obsidian 渲染。

```mermaid
flowchart TD
    classDef person fill:#f9f,stroke:#333,stroke-width:2px;
    classDef object fill:#bbf,stroke:#333,stroke-width:1px;
    classDef emotion fill:#ffd,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef action fill:#dfd,stroke:#333,stroke-width:1px;
    subgraph day1_clip02_11點
        "陳爺爺_day1_clip02_11點"["陳爺爺"]:::person
        "shaving head_day1_clip02_11點_陳爺爺"["shaving head"]:::action
        "王奶奶_day1_clip02_11點"["王奶奶"]:::person
        "dying hair_day1_clip02_11點_王奶奶"["dying hair"]:::action
        "Sadness_day1_clip02_11點_陳爺爺"["Sadness"]:::emotion
        "Neutral_day1_clip02_11點_王奶奶"["Neutral"]:::emotion
    end
    subgraph day1_clip05_17點
        "陳爺爺_day1_clip05_17點"["陳爺爺"]:::person
        "baby waking up_day1_clip05_17點_陳爺爺"["baby waking up"]:::action
        "王奶奶_day1_clip05_17點"["王奶奶"]:::person
        "reading book_day1_clip05_17點_王奶奶"["reading book"]:::action
        "Disgust_day1_clip05_17點_陳爺爺"["Disgust"]:::emotion
        "Disgust_day1_clip05_17點_王奶奶"["Disgust"]:::emotion
    end
    subgraph day1_clip01_9點
        "王奶奶_day1_clip01_9點"["王奶奶"]:::person
        "folding clothes_day1_clip01_9點_王奶奶"["folding clothes"]:::action
        "Sadness_day1_clip01_9點_王奶奶"["Sadness"]:::emotion
    end
        "陳爺爺_day1_clip02_11點" -- performs --> "shaving head_day1_clip02_11點_陳爺爺"
        "王奶奶_day1_clip02_11點" -- performs --> "dying hair_day1_clip02_11點_王奶奶"
        "陳爺爺_day1_clip02_11點" -- feels {clip: 'day1_clip02_11點', time: '11:00:04'} --> "Sadness_day1_clip02_11點_陳爺爺"
        "王奶奶_day1_clip02_11點" -- feels {clip: 'day1_clip02_11點', time: '11:00:06'} --> "Neutral_day1_clip02_11點_王奶奶"
        "陳爺爺_day1_clip05_17點" -- performs --> "baby waking up_day1_clip05_17點_陳爺爺"
        "王奶奶_day1_clip05_17點" -- performs --> "reading book_day1_clip05_17點_王奶奶"
        "陳爺爺_day1_clip05_17點" -- feels {clip: 'day1_clip05_17點', time: '17:00:05'} --> "Disgust_day1_clip05_17點_陳爺爺"
        "王奶奶_day1_clip05_17點" -- feels {clip: 'day1_clip05_17點', time: '17:00:06'} --> "Disgust_day1_clip05_17點_王奶奶"
        "王奶奶_day1_clip01_9點" -- performs --> "folding clothes_day1_clip01_9點_王奶奶"
        "王奶奶_day1_clip01_9點" -- feels {clip: 'day1_clip01_9點', time: '09:00:05'} --> "Sadness_day1_clip01_9點_王奶奶"
        "王奶奶_day1_clip02_11點" -.-|Temporal| "王奶奶_day1_clip05_17點"
        "王奶奶_day1_clip05_17點" -.-|Temporal| "王奶奶_day1_clip01_9點"
        "陳爺爺_day1_clip02_11點" -.-|Temporal| "陳爺爺_day1_clip05_17點"
```

## 互動紀錄
| 時間 | 人物 | 行為 | 物件 |
|------|------|------|------|
| 11:00:00 | 陳爺爺 | shaving head |  |
| 11:00:01 | 王奶奶 | dying hair |  |
| 11:00:04 | 陳爺爺 | Emotion | Sadness |
| 11:00:06 | 王奶奶 | Emotion | Neutral |
| 17:00:00 | 陳爺爺 | baby waking up |  |
| 17:00:01 | 王奶奶 | reading book |  |
| 17:00:05 | 陳爺爺 | Emotion | Disgust |
| 17:00:06 | 王奶奶 | Emotion | Disgust |
| 09:00:00 | 王奶奶 | folding clothes |  |
| 09:00:05 | 王奶奶 | Emotion | Sadness |