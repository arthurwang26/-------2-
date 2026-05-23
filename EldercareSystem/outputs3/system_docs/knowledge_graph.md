## 🕸️ 系統因果與行為知識圖譜 (Knowledge Graph)
> [!info] 這是基於真實 Deep Learning 萃取特徵建立的時序行為關係圖，支援 Obsidian 渲染。

```mermaid
flowchart TD
    classDef person fill:#f9f,stroke:#333,stroke-width:2px;
    classDef object fill:#bbf,stroke:#333,stroke-width:1px;
    classDef emotion fill:#ffd,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef action fill:#dfd,stroke:#333,stroke-width:1px;
    subgraph day1_clip01_9點
        "王奶奶_day1_clip01_9點"["王奶奶"]:::person
        "Unknown_day1_clip01_9點_王奶奶"["Unknown"]:::action
        "Sadness_day1_clip01_9點_王奶奶"["Sadness"]:::emotion
    end
        "王奶奶_day1_clip01_9點" -- performs --> "Unknown_day1_clip01_9點_王奶奶"
        "王奶奶_day1_clip01_9點" -- feels {clip: 'day1_clip01_9點', time: '09:00:06'} --> "Sadness_day1_clip01_9點_王奶奶"
```

## 互動紀錄
| 時間 | 人物 | 行為 | 物件 |
|------|------|------|------|
| 09:00:00 | 王奶奶 | Unknown |  |
| 09:00:06 | 王奶奶 | Emotion | Sadness |