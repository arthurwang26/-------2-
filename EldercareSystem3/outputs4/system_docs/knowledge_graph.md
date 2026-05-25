## 🕸️ 系統因果與行為知識圖譜 (Knowledge Graph)
> [!info] 這是基於真實 Deep Learning 萃取特徵建立的時序行為關係圖，支援 Obsidian 渲染。

```mermaid
flowchart TD
    classDef person fill:#f9f,stroke:#333,stroke-width:2px;
    classDef object fill:#bbf,stroke:#333,stroke-width:1px;
    classDef emotion fill:#ffd,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef action fill:#dfd,stroke:#333,stroke-width:1px;
    subgraph day2_clip02_11點
        "陳爺爺_day2_clip02_11點"["陳爺爺"]:::person
        "laughing_day2_clip02_11點_陳爺爺"["laughing"]:::action
        "王奶奶_day2_clip02_11點"["王奶奶"]:::person
        "petting cat_day2_clip02_11點_王奶奶"["petting cat"]:::action
    end
    subgraph day1_clip06_20點
        "陳爺爺_day1_clip06_20點"["陳爺爺"]:::person
        "shaking head_day1_clip06_20點_陳爺爺"["shaking head"]:::action
        "王奶奶_day1_clip06_20點"["王奶奶"]:::person
        "massaging back_day1_clip06_20點_王奶奶"["massaging back"]:::action
    end
    subgraph day1_clip03_13點
        "王奶奶_day1_clip03_13點"["王奶奶"]:::person
        "shaking head_day1_clip03_13點_王奶奶"["shaking head"]:::action
        "陳爺爺_day1_clip03_13點"["陳爺爺"]:::person
        "knitting_day1_clip03_13點_陳爺爺"["knitting"]:::action
    end
        "陳爺爺_day2_clip02_11點" -- performs --> "laughing_day2_clip02_11點_陳爺爺"
        "王奶奶_day2_clip02_11點" -- performs --> "petting cat_day2_clip02_11點_王奶奶"
        "陳爺爺_day1_clip06_20點" -- performs --> "shaking head_day1_clip06_20點_陳爺爺"
        "王奶奶_day1_clip06_20點" -- performs --> "massaging back_day1_clip06_20點_王奶奶"
        "王奶奶_day1_clip03_13點" -- performs --> "shaking head_day1_clip03_13點_王奶奶"
        "陳爺爺_day1_clip03_13點" -- performs --> "knitting_day1_clip03_13點_陳爺爺"
        "王奶奶_day2_clip02_11點" -.-|Temporal| "王奶奶_day1_clip06_20點"
        "王奶奶_day1_clip06_20點" -.-|Temporal| "王奶奶_day1_clip03_13點"
        "陳爺爺_day2_clip02_11點" -.-|Temporal| "陳爺爺_day1_clip06_20點"
        "陳爺爺_day1_clip06_20點" -.-|Temporal| "陳爺爺_day1_clip03_13點"
```

## 互動紀錄
| 時間 | 人物 | 行為 | 物件 |
|------|------|------|------|
| 11:00:00 | 陳爺爺 | laughing |  |
| 11:00:01 | 王奶奶 | petting cat |  |
| 20:00:00 | 陳爺爺 | shaking head |  |
| 20:00:01 | 王奶奶 | massaging back |  |
| 13:00:00 | 王奶奶 | shaking head |  |
| 13:00:02 | 陳爺爺 | knitting |  |