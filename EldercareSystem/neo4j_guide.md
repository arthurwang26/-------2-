# 🔗 Neo4j Desktop 連接指南

## 一、安裝與設定

### 1. 建立新資料庫
1. 開啟 **Neo4j Desktop**
2. 點選 **「+ New」** 或 **「Add」** 新增一個 Project
3. 在 Project 中，點選 **「Add」→「Local DBMS」**
4. 設定名稱（如 `ElderCare`）和密碼
5. 版本建議選 **5.x** 以上
6. 點選 **「Create」** 建立

### 2. 啟動資料庫
1. 在剛建立的 DBMS 上點選 **「Start」**
2. 等待狀態變為 **「Active」**（綠色燈號）

### 3. 開啟 Neo4j Browser
1. 點選 **「Open」** 按鈕
2. Neo4j Browser 會在瀏覽器中開啟（預設 `http://localhost:7474`）

---

## 二、匯入行為圖譜

### 方法：在 Neo4j Browser 中執行 Cypher

1. 開啟檔案：`outputs/reports/knowledge_graph.cypher`
2. **複製全部內容**
3. 貼到 Neo4j Browser 頂部的指令輸入區
4. 按 **Ctrl+Enter** 或 **Shift+Enter** 執行全部

> ⚠️ 如果出現 `already exists` 錯誤，先執行以下清除指令：
> ```cypher
> MATCH (n) DETACH DELETE n;
> ```

### 驗證匯入
執行以下查詢，確認資料已匯入：
```cypher
// 查看所有節點數量
MATCH (n) RETURN labels(n) AS type, count(n) AS count;

// 查看王奶奶的所有行為
MATCH (p:Person {name: '王奶奶'})-[r]->(n) RETURN p, r, n;

// 查看時間軸
MATCH path = (:Clip)-[:NEXT_CLIP*]->(:Clip) RETURN path;
```

---

## 三、視覺化探索

### 在 Neo4j Browser 中
```cypher
// 查看完整圖譜
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 200;

// 查看特定人物軌跡
MATCH (p:Person {name: '陳爺爺'})-[r]->(n) RETURN p, r, n;

// 查看情緒流動
MATCH (p:Person)-[:FEELS]->(e:Emotion) RETURN p.name, e.name;
```

### 在 Neo4j Bloom 中（如有安裝）
1. 點選 Neo4j Desktop 中的 **「Open with Bloom」**
2. 在搜尋欄輸入 `Person` 查看所有人物
3. 雙擊節點展開其關聯

---

## 四、Bolt 連線資訊
如果需要從外部工具連線：
- **URI**: `bolt://localhost:7687`
- **User**: `neo4j`
- **Password**: 您在建立 DBMS 時設定的密碼
