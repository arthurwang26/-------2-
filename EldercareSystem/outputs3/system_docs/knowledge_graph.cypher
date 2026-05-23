// ====================================================================
// 👵 ElderCare Behavior Knowledge Graph - Neo4j Cypher Import Script
// ====================================================================
//
// 【使用方法 - Neo4j Desktop】
// 1. 開啟 Neo4j Desktop -> 建立或啟動一個 Local DBMS
// 2. 點選「Open」進入 Neo4j Browser
// 3. 將以下所有指令貼到瀏覽器的指令列中
// 4. 按下 Ctrl+Enter (或 Shift+Enter) 執行全部
// 5. 執行完成後可在左側看到節點和關係
//
// 【注意事項】
// - 如果出現重複節點，先執行: MATCH (n) DETACH DELETE n;
// - 社群版不支援 CREATE CONSTRAINT，已移除
// ====================================================================

// 先清除舊資料 (可選)
MATCH (n) DETACH DELETE n;

// === 建立時間軸 (Clip 節點) ===
MERGE (c_day1_clip01_9點:Clip {name: 'day1_clip01_9點', day: 'day1', time: '09:00:00'});

// === 時間順序連接 Clip ===

// --- day1_clip01_9點 ---
MERGE (p:Person {name: '王奶奶'});
MATCH (p:Person {name: '王奶奶'}), (c:Clip {name: 'day1_clip01_9點'})
MERGE (p)-[:APPEARS_IN]->(c);
MERGE (a:Action {name: 'Unknown'});
MERGE (p)-[:PERFORMS {clip: 'day1_clip01_9點', time: '09:00:00'}]->(a);
MERGE (e:Emotion {name: 'Sadness'});
MATCH (p:Person {name: '王奶奶'}), (e:Emotion {name: 'Sadness'})
MERGE (p)-[:FEELS {clip: 'day1_clip01_9點', confidence: 0.16, time: '09:00:00'}]->(e);
