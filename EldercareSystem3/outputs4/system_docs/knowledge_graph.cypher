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
// MATCH (n) DETACH DELETE n;

// === 建立時間軸 (Clip 節點) ===
MERGE (c_day1_clip03_13點:Clip {name: 'day1_clip03_13點', day: 'day1', time: '13:00:00'});
MERGE (c_day1_clip06_20點:Clip {name: 'day1_clip06_20點', day: 'day1', time: '20:00:00'});
MERGE (c_day2_clip02_11點:Clip {name: 'day2_clip02_11點', day: 'day2', time: '11:00:00'});

// === 時間順序連接 Clip ===
MATCH (c1:Clip {name: 'day1_clip03_13點'}), (c2:Clip {name: 'day1_clip06_20點'})
MERGE (c1)-[:NEXT_CLIP]->(c2);
MATCH (c1:Clip {name: 'day1_clip06_20點'}), (c2:Clip {name: 'day2_clip02_11點'})
MERGE (c1)-[:NEXT_CLIP]->(c2);

// --- day2_clip02_11點 ---
MERGE (p:Person {name: '陳爺爺'});
MATCH (p:Person {name: '陳爺爺'}), (c:Clip {name: 'day2_clip02_11點'})
MERGE (p)-[:APPEARS_IN]->(c);
MERGE (a:Action {name: 'laughing'});
MERGE (p)-[:PERFORMS {clip: 'day2_clip02_11點', time: '11:00:00'}]->(a);
MERGE (p:Person {name: '王奶奶'});
MATCH (p:Person {name: '王奶奶'}), (c:Clip {name: 'day2_clip02_11點'})
MERGE (p)-[:APPEARS_IN]->(c);
MERGE (a:Action {name: 'petting cat'});
MERGE (p)-[:PERFORMS {clip: 'day2_clip02_11點', time: '11:00:00'}]->(a);

// --- day1_clip06_20點 ---
MERGE (p:Person {name: '陳爺爺'});
MATCH (p:Person {name: '陳爺爺'}), (c:Clip {name: 'day1_clip06_20點'})
MERGE (p)-[:APPEARS_IN]->(c);
MERGE (a:Action {name: 'shaking head'});
MERGE (p)-[:PERFORMS {clip: 'day1_clip06_20點', time: '20:00:00'}]->(a);
MERGE (p:Person {name: '王奶奶'});
MATCH (p:Person {name: '王奶奶'}), (c:Clip {name: 'day1_clip06_20點'})
MERGE (p)-[:APPEARS_IN]->(c);
MERGE (a:Action {name: 'massaging back'});
MERGE (p)-[:PERFORMS {clip: 'day1_clip06_20點', time: '20:00:00'}]->(a);

// --- day1_clip03_13點 ---
MERGE (p:Person {name: '王奶奶'});
MATCH (p:Person {name: '王奶奶'}), (c:Clip {name: 'day1_clip03_13點'})
MERGE (p)-[:APPEARS_IN]->(c);
MERGE (a:Action {name: 'shaking head'});
MERGE (p)-[:PERFORMS {clip: 'day1_clip03_13點', time: '13:00:00'}]->(a);
MERGE (p:Person {name: '陳爺爺'});
MATCH (p:Person {name: '陳爺爺'}), (c:Clip {name: 'day1_clip03_13點'})
MERGE (p)-[:APPEARS_IN]->(c);
MERGE (a:Action {name: 'knitting'});
MERGE (p)-[:PERFORMS {clip: 'day1_clip03_13點', time: '13:00:00'}]->(a);
