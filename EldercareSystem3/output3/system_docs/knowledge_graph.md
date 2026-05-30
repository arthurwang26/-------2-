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
        "Walking_day1_clip01_9點_王奶奶"["Walking"]:::action
        "Sitting_day1_clip01_9點_王奶奶"["Sitting"]:::action
        "Standing_day1_clip01_9點_王奶奶"["Standing"]:::action
    end
    subgraph day1_clip02_11點
        "陳爺爺_day1_clip02_11點"["陳爺爺"]:::person
        "Standing_day1_clip02_11點_陳爺爺"["Standing"]:::action
        "Sitting_day1_clip02_11點_陳爺爺"["Sitting"]:::action
        "Walking_day1_clip02_11點_陳爺爺"["Walking"]:::action
        "王奶奶_day1_clip02_11點"["王奶奶"]:::person
        "Standing_day1_clip02_11點_王奶奶"["Standing"]:::action
    end
    subgraph day1_clip03_13點
        "王奶奶_day1_clip03_13點"["王奶奶"]:::person
        "Standing_day1_clip03_13點_王奶奶"["Standing"]:::action
        "陳爺爺_day1_clip03_13點"["陳爺爺"]:::person
        "Sitting_day1_clip03_13點_陳爺爺"["Sitting"]:::action
    end
    subgraph day1_clip04_15點
        "陳爺爺_day1_clip04_15點"["陳爺爺"]:::person
        "Walking_day1_clip04_15點_陳爺爺"["Walking"]:::action
        "Standing_day1_clip04_15點_陳爺爺"["Standing"]:::action
        "王奶奶_day1_clip04_15點"["王奶奶"]:::person
        "Sitting_day1_clip04_15點_王奶奶"["Sitting"]:::action
    end
    subgraph day1_clip05_17點
        "陳爺爺_day1_clip05_17點"["陳爺爺"]:::person
        "Walking_day1_clip05_17點_陳爺爺"["Walking"]:::action
        "Standing_day1_clip05_17點_陳爺爺"["Standing"]:::action
        "Sitting_day1_clip05_17點_陳爺爺"["Sitting"]:::action
        "王奶奶_day1_clip05_17點"["王奶奶"]:::person
        "Sitting_day1_clip05_17點_王奶奶"["Sitting"]:::action
    end
    subgraph day1_clip06_20點
        "陳爺爺_day1_clip06_20點"["陳爺爺"]:::person
        "Sitting_day1_clip06_20點_陳爺爺"["Sitting"]:::action
        "Standing_day1_clip06_20點_陳爺爺"["Standing"]:::action
        "王奶奶_day1_clip06_20點"["王奶奶"]:::person
        "Standing_day1_clip06_20點_王奶奶"["Standing"]:::action
    end
    subgraph day2_clip01_9點
        "陳爺爺_day2_clip01_9點"["陳爺爺"]:::person
        "Walking_day2_clip01_9點_陳爺爺"["Walking"]:::action
        "Standing_day2_clip01_9點_陳爺爺"["Standing"]:::action
        "王奶奶_day2_clip01_9點"["王奶奶"]:::person
        "Sitting_day2_clip01_9點_王奶奶"["Sitting"]:::action
    end
    subgraph day2_clip02_11點
        "陳爺爺_day2_clip02_11點"["陳爺爺"]:::person
        "Standing_day2_clip02_11點_陳爺爺"["Standing"]:::action
        "Sitting_day2_clip02_11點_陳爺爺"["Sitting"]:::action
        "王奶奶_day2_clip02_11點"["王奶奶"]:::person
        "Standing_day2_clip02_11點_王奶奶"["Standing"]:::action
    end
    subgraph day2_clip03_13點
        "陳爺爺_day2_clip03_13點"["陳爺爺"]:::person
        "Standing_day2_clip03_13點_陳爺爺"["Standing"]:::action
        "王奶奶_day2_clip03_13點"["王奶奶"]:::person
        "Sitting_day2_clip03_13點_王奶奶"["Sitting"]:::action
        "Standing_day2_clip03_13點_王奶奶"["Standing"]:::action
    end
    subgraph day2_clip04_15點
        "王奶奶_day2_clip04_15點"["王奶奶"]:::person
        "Sitting_day2_clip04_15點_王奶奶"["Sitting"]:::action
        "Standing_day2_clip04_15點_王奶奶"["Standing"]:::action
        "陳爺爺_day2_clip04_15點"["陳爺爺"]:::person
        "Sitting_day2_clip04_15點_陳爺爺"["Sitting"]:::action
        "Standing_day2_clip04_15點_陳爺爺"["Standing"]:::action
        "Walking_day2_clip04_15點_陳爺爺"["Walking"]:::action
    end
    subgraph day2_clip05_17點
        "王奶奶_day2_clip05_17點"["王奶奶"]:::person
        "Sitting_day2_clip05_17點_王奶奶"["Sitting"]:::action
        "陳爺爺_day2_clip05_17點"["陳爺爺"]:::person
        "Walking_day2_clip05_17點_陳爺爺"["Walking"]:::action
        "Sitting_day2_clip05_17點_陳爺爺"["Sitting"]:::action
        "Standing_day2_clip05_17點_陳爺爺"["Standing"]:::action
    end
    subgraph day2_clip06_20點
        "陳爺爺_day2_clip06_20點"["陳爺爺"]:::person
        "Walking_day2_clip06_20點_陳爺爺"["Walking"]:::action
        "Standing_day2_clip06_20點_陳爺爺"["Standing"]:::action
        "王奶奶_day2_clip06_20點"["王奶奶"]:::person
        "Walking_day2_clip06_20點_王奶奶"["Walking"]:::action
    end
        "王奶奶_day1_clip01_9點" -- performs --> "Walking_day1_clip01_9點_王奶奶"
        "王奶奶_day1_clip01_9點" -- performs --> "Sitting_day1_clip01_9點_王奶奶"
        "王奶奶_day1_clip01_9點" -- performs --> "Walking_day1_clip01_9點_王奶奶"
        "王奶奶_day1_clip01_9點" -- performs --> "Standing_day1_clip01_9點_王奶奶"
        "王奶奶_day1_clip01_9點" -- performs --> "Sitting_day1_clip01_9點_王奶奶"
        "陳爺爺_day1_clip02_11點" -- performs --> "Standing_day1_clip02_11點_陳爺爺"
        "陳爺爺_day1_clip02_11點" -- performs --> "Sitting_day1_clip02_11點_陳爺爺"
        "陳爺爺_day1_clip02_11點" -- performs --> "Standing_day1_clip02_11點_陳爺爺"
        "陳爺爺_day1_clip02_11點" -- performs --> "Walking_day1_clip02_11點_陳爺爺"
        "陳爺爺_day1_clip02_11點" -- performs --> "Standing_day1_clip02_11點_陳爺爺"
        "王奶奶_day1_clip02_11點" -- performs --> "Standing_day1_clip02_11點_王奶奶"
        "王奶奶_day1_clip03_13點" -- performs --> "Standing_day1_clip03_13點_王奶奶"
        "陳爺爺_day1_clip03_13點" -- performs --> "Sitting_day1_clip03_13點_陳爺爺"
        "陳爺爺_day1_clip04_15點" -- performs --> "Walking_day1_clip04_15點_陳爺爺"
        "陳爺爺_day1_clip04_15點" -- performs --> "Standing_day1_clip04_15點_陳爺爺"
        "陳爺爺_day1_clip04_15點" -- performs --> "Walking_day1_clip04_15點_陳爺爺"
        "王奶奶_day1_clip04_15點" -- performs --> "Sitting_day1_clip04_15點_王奶奶"
        "陳爺爺_day1_clip05_17點" -- performs --> "Walking_day1_clip05_17點_陳爺爺"
        "陳爺爺_day1_clip05_17點" -- performs --> "Standing_day1_clip05_17點_陳爺爺"
        "陳爺爺_day1_clip05_17點" -- performs --> "Sitting_day1_clip05_17點_陳爺爺"
        "陳爺爺_day1_clip05_17點" -- performs --> "Standing_day1_clip05_17點_陳爺爺"
        "王奶奶_day1_clip05_17點" -- performs --> "Sitting_day1_clip05_17點_王奶奶"
        "陳爺爺_day1_clip06_20點" -- performs --> "Sitting_day1_clip06_20點_陳爺爺"
        "陳爺爺_day1_clip06_20點" -- performs --> "Standing_day1_clip06_20點_陳爺爺"
        "陳爺爺_day1_clip06_20點" -- performs --> "Sitting_day1_clip06_20點_陳爺爺"
        "王奶奶_day1_clip06_20點" -- performs --> "Standing_day1_clip06_20點_王奶奶"
        "陳爺爺_day2_clip01_9點" -- performs --> "Walking_day2_clip01_9點_陳爺爺"
        "陳爺爺_day2_clip01_9點" -- performs --> "Standing_day2_clip01_9點_陳爺爺"
        "王奶奶_day2_clip01_9點" -- performs --> "Sitting_day2_clip01_9點_王奶奶"
        "陳爺爺_day2_clip02_11點" -- performs --> "Standing_day2_clip02_11點_陳爺爺"
        "陳爺爺_day2_clip02_11點" -- performs --> "Sitting_day2_clip02_11點_陳爺爺"
        "陳爺爺_day2_clip02_11點" -- performs --> "Standing_day2_clip02_11點_陳爺爺"
        "王奶奶_day2_clip02_11點" -- performs --> "Standing_day2_clip02_11點_王奶奶"
        "陳爺爺_day2_clip03_13點" -- performs --> "Standing_day2_clip03_13點_陳爺爺"
        "王奶奶_day2_clip03_13點" -- performs --> "Sitting_day2_clip03_13點_王奶奶"
        "王奶奶_day2_clip03_13點" -- performs --> "Standing_day2_clip03_13點_王奶奶"
        "王奶奶_day2_clip04_15點" -- performs --> "Sitting_day2_clip04_15點_王奶奶"
        "王奶奶_day2_clip04_15點" -- performs --> "Standing_day2_clip04_15點_王奶奶"
        "陳爺爺_day2_clip04_15點" -- performs --> "Sitting_day2_clip04_15點_陳爺爺"
        "陳爺爺_day2_clip04_15點" -- performs --> "Standing_day2_clip04_15點_陳爺爺"
        "陳爺爺_day2_clip04_15點" -- performs --> "Walking_day2_clip04_15點_陳爺爺"
        "陳爺爺_day2_clip04_15點" -- performs --> "Standing_day2_clip04_15點_陳爺爺"
        "王奶奶_day2_clip05_17點" -- performs --> "Sitting_day2_clip05_17點_王奶奶"
        "陳爺爺_day2_clip05_17點" -- performs --> "Walking_day2_clip05_17點_陳爺爺"
        "陳爺爺_day2_clip05_17點" -- performs --> "Sitting_day2_clip05_17點_陳爺爺"
        "陳爺爺_day2_clip05_17點" -- performs --> "Standing_day2_clip05_17點_陳爺爺"
        "陳爺爺_day2_clip05_17點" -- performs --> "Sitting_day2_clip05_17點_陳爺爺"
        "陳爺爺_day2_clip06_20點" -- performs --> "Walking_day2_clip06_20點_陳爺爺"
        "陳爺爺_day2_clip06_20點" -- performs --> "Standing_day2_clip06_20點_陳爺爺"
        "王奶奶_day2_clip06_20點" -- performs --> "Walking_day2_clip06_20點_王奶奶"
        "陳爺爺_day1_clip02_11點" -.-|Temporal| "陳爺爺_day1_clip03_13點"
        "陳爺爺_day1_clip03_13點" -.-|Temporal| "陳爺爺_day1_clip04_15點"
        "陳爺爺_day1_clip04_15點" -.-|Temporal| "陳爺爺_day1_clip05_17點"
        "陳爺爺_day1_clip05_17點" -.-|Temporal| "陳爺爺_day1_clip06_20點"
        "陳爺爺_day1_clip06_20點" -.-|Temporal| "陳爺爺_day2_clip01_9點"
        "陳爺爺_day2_clip01_9點" -.-|Temporal| "陳爺爺_day2_clip02_11點"
        "陳爺爺_day2_clip02_11點" -.-|Temporal| "陳爺爺_day2_clip03_13點"
        "陳爺爺_day2_clip03_13點" -.-|Temporal| "陳爺爺_day2_clip04_15點"
        "陳爺爺_day2_clip04_15點" -.-|Temporal| "陳爺爺_day2_clip05_17點"
        "陳爺爺_day2_clip05_17點" -.-|Temporal| "陳爺爺_day2_clip06_20點"
        "王奶奶_day1_clip01_9點" -.-|Temporal| "王奶奶_day1_clip02_11點"
        "王奶奶_day1_clip02_11點" -.-|Temporal| "王奶奶_day1_clip03_13點"
        "王奶奶_day1_clip03_13點" -.-|Temporal| "王奶奶_day1_clip04_15點"
        "王奶奶_day1_clip04_15點" -.-|Temporal| "王奶奶_day1_clip05_17點"
        "王奶奶_day1_clip05_17點" -.-|Temporal| "王奶奶_day1_clip06_20點"
        "王奶奶_day1_clip06_20點" -.-|Temporal| "王奶奶_day2_clip01_9點"
        "王奶奶_day2_clip01_9點" -.-|Temporal| "王奶奶_day2_clip02_11點"
        "王奶奶_day2_clip02_11點" -.-|Temporal| "王奶奶_day2_clip03_13點"
        "王奶奶_day2_clip03_13點" -.-|Temporal| "王奶奶_day2_clip04_15點"
        "王奶奶_day2_clip04_15點" -.-|Temporal| "王奶奶_day2_clip05_17點"
        "王奶奶_day2_clip05_17點" -.-|Temporal| "王奶奶_day2_clip06_20點"
```

## 互動紀錄
| 時間 | 人物 | 行為 | 物件 |
|------|------|------|------|
| 09:00:00 | 王奶奶 | Walking |  |
| 09:00:01 | 王奶奶 | Sitting |  |
| 09:00:02 | 王奶奶 | Walking |  |
| 09:00:03 | 王奶奶 | Standing |  |
| 09:00:04 | 王奶奶 | Sitting |  |
| 11:00:00 | 陳爺爺 | Standing |  |
| 11:00:01 | 陳爺爺 | Sitting |  |
| 11:00:02 | 陳爺爺 | Standing |  |
| 11:00:03 | 陳爺爺 | Walking |  |
| 11:00:04 | 陳爺爺 | Standing |  |
| 11:00:05 | 王奶奶 | Standing |  |
| 13:00:00 | 王奶奶 | Standing |  |
| 13:00:02 | 陳爺爺 | Sitting |  |
| 15:00:00 | 陳爺爺 | Walking |  |
| 15:00:01 | 陳爺爺 | Standing |  |
| 15:00:03 | 陳爺爺 | Walking |  |
| 15:00:04 | 王奶奶 | Sitting |  |
| 17:00:00 | 陳爺爺 | Walking |  |
| 17:00:01 | 陳爺爺 | Standing |  |
| 17:00:02 | 陳爺爺 | Sitting |  |
| 17:00:04 | 陳爺爺 | Standing |  |
| 17:00:05 | 王奶奶 | Sitting |  |
| 20:00:00 | 陳爺爺 | Sitting |  |
| 20:00:01 | 陳爺爺 | Standing |  |
| 20:00:02 | 陳爺爺 | Sitting |  |
| 20:00:03 | 王奶奶 | Standing |  |
| 09:00:00 | 陳爺爺 | Walking |  |
| 09:00:01 | 陳爺爺 | Standing |  |
| 09:00:03 | 王奶奶 | Sitting |  |
| 11:00:00 | 陳爺爺 | Standing |  |
| 11:00:00 | 陳爺爺 | Sitting |  |
| 11:00:01 | 陳爺爺 | Standing |  |
| 11:00:02 | 王奶奶 | Standing |  |
| 13:00:00 | 陳爺爺 | Standing |  |
| 13:00:01 | 王奶奶 | Sitting |  |
| 13:00:03 | 王奶奶 | Standing |  |
| 15:00:00 | 王奶奶 | Sitting |  |
| 15:00:01 | 王奶奶 | Standing |  |
| 15:00:02 | 陳爺爺 | Sitting |  |
| 15:00:03 | 陳爺爺 | Standing |  |
| 15:00:04 | 陳爺爺 | Walking |  |
| 15:00:05 | 陳爺爺 | Standing |  |
| 17:00:00 | 王奶奶 | Sitting |  |
| 17:00:00 | 陳爺爺 | Walking |  |
| 17:00:01 | 陳爺爺 | Sitting |  |
| 17:00:02 | 陳爺爺 | Standing |  |
| 17:00:03 | 陳爺爺 | Sitting |  |
| 20:00:00 | 陳爺爺 | Walking |  |
| 20:00:01 | 陳爺爺 | Standing |  |
| 20:00:03 | 王奶奶 | Walking |  |