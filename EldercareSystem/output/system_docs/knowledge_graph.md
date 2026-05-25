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
        "chair_day1_clip01_9點"["chair"]:::object
        "cup_day1_clip01_9點"["cup"]:::object
        "dining table_day1_clip01_9點"["dining table"]:::object
        "couch_day1_clip01_9點"["couch"]:::object
    end
    subgraph day1_clip02_11點
        "陳爺爺_day1_clip02_11點"["陳爺爺"]:::person
        "Unknown_day1_clip02_11點_陳爺爺"["Unknown"]:::action
        "王奶奶_day1_clip02_11點"["王奶奶"]:::person
        "Unknown_day1_clip02_11點_王奶奶"["Unknown"]:::action
        "couch_day1_clip02_11點"["couch"]:::object
    end
    subgraph day1_clip03_13點
        "王奶奶_day1_clip03_13點"["王奶奶"]:::person
        "Unknown_day1_clip03_13點_王奶奶"["Unknown"]:::action
        "陳爺爺_day1_clip03_13點"["陳爺爺"]:::person
        "Unknown_day1_clip03_13點_陳爺爺"["Unknown"]:::action
        "cup_day1_clip03_13點"["cup"]:::object
        "dining table_day1_clip03_13點"["dining table"]:::object
        "chair_day1_clip03_13點"["chair"]:::object
    end
    subgraph day1_clip04_15點
        "陳爺爺_day1_clip04_15點"["陳爺爺"]:::person
        "Unknown_day1_clip04_15點_陳爺爺"["Unknown"]:::action
        "chair_day1_clip04_15點"["chair"]:::object
        "couch_day1_clip04_15點"["couch"]:::object
        "tv_day1_clip04_15點"["tv"]:::object
        "dining table_day1_clip04_15點"["dining table"]:::object
    end
    subgraph day1_clip05_17點
        "陳爺爺_day1_clip05_17點"["陳爺爺"]:::person
        "Unknown_day1_clip05_17點_陳爺爺"["Unknown"]:::action
        "王奶奶_day1_clip05_17點"["王奶奶"]:::person
        "Unknown_day1_clip05_17點_王奶奶"["Unknown"]:::action
    end
    subgraph day1_clip06_20點
        "陳爺爺_day1_clip06_20點"["陳爺爺"]:::person
        "Unknown_day1_clip06_20點_陳爺爺"["Unknown"]:::action
        "王奶奶_day1_clip06_20點"["王奶奶"]:::person
        "Unknown_day1_clip06_20點_王奶奶"["Unknown"]:::action
    end
    subgraph day2_clip01_9點
        "陳爺爺_day2_clip01_9點"["陳爺爺"]:::person
        "Unknown_day2_clip01_9點_陳爺爺"["Unknown"]:::action
        "王奶奶_day2_clip01_9點"["王奶奶"]:::person
        "Unknown_day2_clip01_9點_王奶奶"["Unknown"]:::action
        "couch_day2_clip01_9點"["couch"]:::object
        "remote_day2_clip01_9點"["remote"]:::object
    end
    subgraph day2_clip02_11點
        "陳爺爺_day2_clip02_11點"["陳爺爺"]:::person
        "Unknown_day2_clip02_11點_陳爺爺"["Unknown"]:::action
        "王奶奶_day2_clip02_11點"["王奶奶"]:::person
        "Unknown_day2_clip02_11點_王奶奶"["Unknown"]:::action
    end
    subgraph day2_clip03_13點
        "陳爺爺_day2_clip03_13點"["陳爺爺"]:::person
        "Unknown_day2_clip03_13點_陳爺爺"["Unknown"]:::action
        "王奶奶_day2_clip03_13點"["王奶奶"]:::person
        "Unknown_day2_clip03_13點_王奶奶"["Unknown"]:::action
    end
    subgraph day2_clip04_15點
        "王奶奶_day2_clip04_15點"["王奶奶"]:::person
        "Unknown_day2_clip04_15點_王奶奶"["Unknown"]:::action
        "陳爺爺_day2_clip04_15點"["陳爺爺"]:::person
        "Unknown_day2_clip04_15點_陳爺爺"["Unknown"]:::action
        "chair_day2_clip04_15點"["chair"]:::object
    end
    subgraph day2_clip05_17點
        "王奶奶_day2_clip05_17點"["王奶奶"]:::person
        "Unknown_day2_clip05_17點_王奶奶"["Unknown"]:::action
        "陳爺爺_day2_clip05_17點"["陳爺爺"]:::person
        "Unknown_day2_clip05_17點_陳爺爺"["Unknown"]:::action
        "chair_day2_clip05_17點"["chair"]:::object
    end
    subgraph day2_clip06_20點
        "陳爺爺_day2_clip06_20點"["陳爺爺"]:::person
        "Unknown_day2_clip06_20點_陳爺爺"["Unknown"]:::action
        "王奶奶_day2_clip06_20點"["王奶奶"]:::person
        "Unknown_day2_clip06_20點_王奶奶"["Unknown"]:::action
    end
        "王奶奶_day1_clip01_9點" -- performs --> "Unknown_day1_clip01_9點_王奶奶"
        "王奶奶_day1_clip01_9點" -- Sitting_On {clip: 'day1_clip01_9點', time: '09:00:00'} --> "chair_day1_clip01_9點"
        "王奶奶_day1_clip01_9點" -- Sitting_On {clip: 'day1_clip01_9點', time: '09:00:01'} --> "cup_day1_clip01_9點"
        "王奶奶_day1_clip01_9點" -- Sitting_On {clip: 'day1_clip01_9點', time: '09:00:01'} --> "dining table_day1_clip01_9點"
        "王奶奶_day1_clip01_9點" -- Sitting_On {clip: 'day1_clip01_9點', time: '09:00:02'} --> "couch_day1_clip01_9點"
        "陳爺爺_day1_clip02_11點" -- performs --> "Unknown_day1_clip02_11點_陳爺爺"
        "王奶奶_day1_clip02_11點" -- performs --> "Unknown_day1_clip02_11點_王奶奶"
        "王奶奶_day1_clip02_11點" -- Sitting_On {clip: 'day1_clip02_11點', time: '11:00:01'} --> "couch_day1_clip02_11點"
        "王奶奶_day1_clip03_13點" -- performs --> "Unknown_day1_clip03_13點_王奶奶"
        "陳爺爺_day1_clip03_13點" -- performs --> "Unknown_day1_clip03_13點_陳爺爺"
        "王奶奶_day1_clip03_13點" -- Using {clip: 'day1_clip03_13點', time: '13:00:01'} --> "cup_day1_clip03_13點"
        "王奶奶_day1_clip03_13點" -- Using {clip: 'day1_clip03_13點', time: '13:00:01'} --> "dining table_day1_clip03_13點"
        "王奶奶_day1_clip03_13點" -- Sitting_On {clip: 'day1_clip03_13點', time: '13:00:02'} --> "chair_day1_clip03_13點"
        "陳爺爺_day1_clip04_15點" -- performs --> "Unknown_day1_clip04_15點_陳爺爺"
        "陳爺爺_day1_clip04_15點" -- Touching {clip: 'day1_clip04_15點', time: '15:00:00'} --> "chair_day1_clip04_15點"
        "陳爺爺_day1_clip04_15點" -- Touching {clip: 'day1_clip04_15點', time: '15:00:01'} --> "couch_day1_clip04_15點"
        "陳爺爺_day1_clip04_15點" -- Touching {clip: 'day1_clip04_15點', time: '15:00:01'} --> "tv_day1_clip04_15點"
        "陳爺爺_day1_clip04_15點" -- Touching {clip: 'day1_clip04_15點', time: '15:00:02'} --> "dining table_day1_clip04_15點"
        "陳爺爺_day1_clip05_17點" -- performs --> "Unknown_day1_clip05_17點_陳爺爺"
        "王奶奶_day1_clip05_17點" -- performs --> "Unknown_day1_clip05_17點_王奶奶"
        "陳爺爺_day1_clip06_20點" -- performs --> "Unknown_day1_clip06_20點_陳爺爺"
        "王奶奶_day1_clip06_20點" -- performs --> "Unknown_day1_clip06_20點_王奶奶"
        "陳爺爺_day2_clip01_9點" -- performs --> "Unknown_day2_clip01_9點_陳爺爺"
        "王奶奶_day2_clip01_9點" -- performs --> "Unknown_day2_clip01_9點_王奶奶"
        "王奶奶_day2_clip01_9點" -- Sitting_On {clip: 'day2_clip01_9點', time: '09:00:01'} --> "couch_day2_clip01_9點"
        "王奶奶_day2_clip01_9點" -- Holding {clip: 'day2_clip01_9點', time: '09:00:01'} --> "remote_day2_clip01_9點"
        "陳爺爺_day2_clip02_11點" -- performs --> "Unknown_day2_clip02_11點_陳爺爺"
        "王奶奶_day2_clip02_11點" -- performs --> "Unknown_day2_clip02_11點_王奶奶"
        "陳爺爺_day2_clip03_13點" -- performs --> "Unknown_day2_clip03_13點_陳爺爺"
        "王奶奶_day2_clip03_13點" -- performs --> "Unknown_day2_clip03_13點_王奶奶"
        "王奶奶_day2_clip04_15點" -- performs --> "Unknown_day2_clip04_15點_王奶奶"
        "陳爺爺_day2_clip04_15點" -- performs --> "Unknown_day2_clip04_15點_陳爺爺"
        "王奶奶_day2_clip04_15點" -- Sitting_On {clip: 'day2_clip04_15點', time: '15:00:01'} --> "chair_day2_clip04_15點"
        "王奶奶_day2_clip05_17點" -- performs --> "Unknown_day2_clip05_17點_王奶奶"
        "陳爺爺_day2_clip05_17點" -- performs --> "Unknown_day2_clip05_17點_陳爺爺"
        "陳爺爺_day2_clip05_17點" -- Sitting_On {clip: 'day2_clip05_17點', time: '17:00:00'} --> "chair_day2_clip05_17點"
        "陳爺爺_day2_clip06_20點" -- performs --> "Unknown_day2_clip06_20點_陳爺爺"
        "王奶奶_day2_clip06_20點" -- performs --> "Unknown_day2_clip06_20點_王奶奶"
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
| 09:00:00 | 王奶奶 | Unknown |  |
| 09:00:00 | 王奶奶 | Sitting_On | chair |
| 09:00:01 | 王奶奶 | Sitting_On | cup |
| 09:00:01 | 王奶奶 | Sitting_On | dining table |
| 09:00:02 | 王奶奶 | Sitting_On | couch |
| 11:00:00 | 陳爺爺 | Unknown |  |
| 11:00:00 | 王奶奶 | Unknown |  |
| 11:00:01 | 王奶奶 | Sitting_On | couch |
| 13:00:00 | 王奶奶 | Unknown |  |
| 13:00:00 | 陳爺爺 | Unknown |  |
| 13:00:01 | 王奶奶 | Using | cup |
| 13:00:01 | 王奶奶 | Using | dining table |
| 13:00:02 | 王奶奶 | Sitting_On | chair |
| 15:00:00 | 陳爺爺 | Unknown |  |
| 15:00:00 | 陳爺爺 | Touching | chair |
| 15:00:01 | 陳爺爺 | Touching | couch |
| 15:00:01 | 陳爺爺 | Touching | tv |
| 15:00:02 | 陳爺爺 | Touching | dining table |
| 17:00:00 | 陳爺爺 | Unknown |  |
| 17:00:00 | 王奶奶 | Unknown |  |
| 20:00:00 | 陳爺爺 | Unknown |  |
| 20:00:00 | 王奶奶 | Unknown |  |
| 09:00:00 | 陳爺爺 | Unknown |  |
| 09:00:00 | 王奶奶 | Unknown |  |
| 09:00:01 | 王奶奶 | Sitting_On | couch |
| 09:00:01 | 王奶奶 | Holding | remote |
| 11:00:00 | 陳爺爺 | Unknown |  |
| 11:00:00 | 王奶奶 | Unknown |  |
| 13:00:00 | 陳爺爺 | Unknown |  |
| 13:00:01 | 王奶奶 | Unknown |  |
| 15:00:00 | 王奶奶 | Unknown |  |
| 15:00:00 | 陳爺爺 | Unknown |  |
| 15:00:01 | 王奶奶 | Sitting_On | chair |
| 17:00:00 | 王奶奶 | Unknown |  |
| 17:00:00 | 陳爺爺 | Unknown |  |
| 17:00:00 | 陳爺爺 | Sitting_On | chair |
| 20:00:00 | 陳爺爺 | Unknown |  |
| 20:00:01 | 王奶奶 | Unknown |  |