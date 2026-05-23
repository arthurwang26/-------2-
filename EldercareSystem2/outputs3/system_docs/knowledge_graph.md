## 🕸️ 系統因果與行為知識圖譜 (Knowledge Graph)
> [!info] 這是基於真實 Deep Learning 萃取特徵建立的時序行為關係圖，支援 Obsidian 渲染。

```mermaid
flowchart TD
    classDef person fill:#f9f,stroke:#333,stroke-width:2px;
    classDef object fill:#bbf,stroke:#333,stroke-width:1px;
    classDef emotion fill:#ffd,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef action fill:#dfd,stroke:#333,stroke-width:1px;
    subgraph day2_clip03_13點
        "陳爺爺_day2_clip03_13點"["陳爺爺"]:::person
        "handshaking_day2_clip03_13點_陳爺爺"["handshaking"]:::action
        "王奶奶_day2_clip03_13點"["王奶奶"]:::person
        "handshaking_day2_clip03_13點_王奶奶"["handshaking"]:::action
        "Happiness_day2_clip03_13點_陳爺爺"["Happiness"]:::emotion
        "Disgust_day2_clip03_13點_王奶奶"["Disgust"]:::emotion
    end
    subgraph day2_clip06_20點
        "陳爺爺_day2_clip06_20點"["陳爺爺"]:::person
        "brushing teeth_day2_clip06_20點_陳爺爺"["brushing teeth"]:::action
        "王奶奶_day2_clip06_20點"["王奶奶"]:::person
        "brushing teeth_day2_clip06_20點_王奶奶"["brushing teeth"]:::action
        "Happiness_day2_clip06_20點_陳爺爺"["Happiness"]:::emotion
        "Happiness_day2_clip06_20點_王奶奶"["Happiness"]:::emotion
    end
    subgraph day2_clip04_15點
        "王奶奶_day2_clip04_15點"["王奶奶"]:::person
        "use a fan_day2_clip04_15點_王奶奶"["use a fan"]:::action
        "Disgust_day2_clip04_15點_王奶奶"["Disgust"]:::emotion
    end
        "陳爺爺_day2_clip03_13點" -- performs --> "handshaking_day2_clip03_13點_陳爺爺"
        "王奶奶_day2_clip03_13點" -- performs --> "handshaking_day2_clip03_13點_王奶奶"
        "陳爺爺_day2_clip03_13點" -- feels {clip: 'day2_clip03_13點', time: '13:00:06'} --> "Happiness_day2_clip03_13點_陳爺爺"
        "王奶奶_day2_clip03_13點" -- feels {clip: 'day2_clip03_13點', time: '13:00:07'} --> "Disgust_day2_clip03_13點_王奶奶"
        "陳爺爺_day2_clip06_20點" -- performs --> "brushing teeth_day2_clip06_20點_陳爺爺"
        "王奶奶_day2_clip06_20點" -- performs --> "brushing teeth_day2_clip06_20點_王奶奶"
        "陳爺爺_day2_clip06_20點" -- feels {clip: 'day2_clip06_20點', time: '20:00:04'} --> "Happiness_day2_clip06_20點_陳爺爺"
        "王奶奶_day2_clip06_20點" -- feels {clip: 'day2_clip06_20點', time: '20:00:06'} --> "Happiness_day2_clip06_20點_王奶奶"
        "王奶奶_day2_clip04_15點" -- performs --> "use a fan_day2_clip04_15點_王奶奶"
        "王奶奶_day2_clip04_15點" -- feels {clip: 'day2_clip04_15點', time: '15:00:05'} --> "Disgust_day2_clip04_15點_王奶奶"
        "王奶奶_day2_clip03_13點" -.-|Temporal| "王奶奶_day2_clip06_20點"
        "王奶奶_day2_clip06_20點" -.-|Temporal| "王奶奶_day2_clip04_15點"
        "陳爺爺_day2_clip03_13點" -.-|Temporal| "陳爺爺_day2_clip06_20點"
```

## 互動紀錄
| 時間 | 人物 | 行為 | 物件 |
|------|------|------|------|
| 13:00:00 | 陳爺爺 | handshaking |  |
| 13:00:01 | 王奶奶 | handshaking |  |
| 13:00:06 | 陳爺爺 | Emotion | Happiness |
| 13:00:07 | 王奶奶 | Emotion | Disgust |
| 20:00:00 | 陳爺爺 | brushing teeth |  |
| 20:00:02 | 王奶奶 | brushing teeth |  |
| 20:00:04 | 陳爺爺 | Emotion | Happiness |
| 20:00:06 | 王奶奶 | Emotion | Happiness |
| 15:00:00 | 王奶奶 | use a fan |  |
| 15:00:05 | 王奶奶 | Emotion | Disgust |