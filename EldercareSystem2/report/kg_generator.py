import json
from pathlib import Path


def parse_clip_name(clip_name: str):
    """Parse a clip name like 'day1_clip01_9點' or 'day1_clip01' to extract day, clip id and time.
    Returns a tuple (day, clip_id, time_str) where time_str is in HH:MM:SS format or empty.
    """
    parts = clip_name.split('_')
    day = parts[0] if len(parts) > 0 else ''
    clip_id = parts[1] if len(parts) > 1 else ''
    time_str = ''
    if len(parts) > 2:
        raw = parts[2]
        if raw.endswith('點'):
            hour = raw.rstrip('點')
            try:
                hour_int = int(hour)
                time_str = f"{hour_int:02d}:00:00"
            except ValueError:
                time_str = ''
    return day, clip_id, time_str


def add_seconds(time_str: str, seconds: int) -> str:
    """Helper to add seconds to a HH:MM:SS time string."""
    if not time_str:
        return time_str
    try:
        parts = time_str.split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        s += seconds
        m += s // 60
        s = s % 60
        h += m // 60
        m = m % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    except:
        return time_str


def generate_mermaid_graph(events_by_clip: dict, output_path: Path):
    """Generate Mermaid knowledge graph and append an interaction record table."""
    lines = [
        "## 🕸️ 系統因果與行為知識圖譜 (Knowledge Graph)",
        "> [!info] 這是基於真實 Deep Learning 萃取特徵建立的時序行為關係圖，支援 Obsidian 渲染。",
        "",
        "```mermaid",
        "flowchart TD",
        "    classDef person fill:#f9f,stroke:#333,stroke-width:2px;",
        "    classDef object fill:#bbf,stroke:#333,stroke-width:1px;",
        "    classDef emotion fill:#ffd,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;",
        "    classDef action fill:#dfd,stroke:#333,stroke-width:1px;",
    ]
    nodes = set()
    edges = []
    interaction_rows = []
    for clip_name, events in events_by_clip.items():
        lines.append(f"    subgraph {clip_name}")
        day, cid, base_time_str = parse_clip_name(clip_name)
        
        # We assume 8 seconds clip, distribute events evenly or sequentially
        total_events = len(events)
        for idx, ev in enumerate(events):
            # 確保內容逐秒遞增，最多 8 秒，不會卡在同一個時間點
            sec_offset = int((idx / max(1, total_events)) * 8)
            time_str = add_seconds(base_time_str, sec_offset)
            person = ev.get('person', 'Unknown')
            if person == 'Unknown':
                continue
            p_node = f'"{person}_{clip_name}"'
            if p_node not in nodes:
                lines.append(f'        {p_node}["{person}"]:::person')
                nodes.add(p_node)
            e_type = ev.get('type')
            if e_type == 'Action':
                action = ev.get('action')
                a_node = f'"{action}_{clip_name}_{person}"'
                if a_node not in nodes:
                    lines.append(f'        {a_node}["{action}"]:::action')
                    nodes.add(a_node)
                edges.append(f'        {p_node} -- performs --> {a_node}')
                interaction_rows.append([time_str or clip_name, person, action, ""])
            elif e_type == 'HOI':
                action = ev.get('action')
                obj = ev.get('object')
                o_node = f'"{obj}_{clip_name}"'
                if o_node not in nodes:
                    lines.append(f'        {o_node}["{obj}"]:::object')
                    nodes.add(o_node)
                time_attr = f"{{clip: '{clip_name}', time: '{time_str}'}}" if time_str else f"{{clip: '{clip_name}'}}"
                edges.append(f'        {p_node} -- {action} {time_attr} --> {o_node}')
                interaction_rows.append([time_str or clip_name, person, action, obj])
            elif e_type == 'Emotion':
                emotion = ev.get('emotion')
                em_node = f'"{emotion}_{clip_name}_{person}"'
                if em_node not in nodes:
                    lines.append(f'        {em_node}["{emotion}"]:::emotion')
                    nodes.add(em_node)
                time_attr = f"{{clip: '{clip_name}', time: '{time_str}'}}" if time_str else f"{{clip: '{clip_name}'}}"
                edges.append(f'        {p_node} -- feels {time_attr} --> {em_node}')
                interaction_rows.append([time_str or clip_name, person, "Emotion", emotion])
        lines.append("    end")
    # Temporal arrows between same person across clips
    persons = set()
    for events in events_by_clip.values():
        for ev in events:
            p = ev.get('person')
            if p and p != 'Unknown':
                persons.add(p)
    clip_names = list(events_by_clip.keys())
    for p in persons:
        for i in range(len(clip_names) - 1):
            cur = f'"{p}_{clip_names[i]}"'
            nxt = f'"{p}_{clip_names[i+1]}"'
            if cur in nodes and nxt in nodes:
                edges.append(f'        {cur} -.-|Temporal| {nxt}')
    lines.extend(edges)
    lines.append("```")
    lines.append("")
    # Interaction record table
    lines.append("## 互動紀錄")
    lines.append("| 時間 | 人物 | 行為 | 物件 |")
    lines.append("|------|------|------|------|")
    for row in interaction_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_neo4j_cypher(events_by_clip: dict, output_path: Path):
    """Generate Neo4j Cypher script with time attributes on Clip and HOI edges."""
    lines = [
        "// ====================================================================",
        "// 👵 ElderCare Behavior Knowledge Graph - Neo4j Cypher Import Script",
        "// ====================================================================",
        "//",
        "// 【使用方法 - Neo4j Desktop】",
        "// 1. 開啟 Neo4j Desktop -> 建立或啟動一個 Local DBMS",
        "// 2. 點選「Open」進入 Neo4j Browser",
        "// 3. 將以下所有指令貼到瀏覽器的指令列中",
        "// 4. 按下 Ctrl+Enter (或 Shift+Enter) 執行全部",
        "// 5. 執行完成後可在左側看到節點和關係",
        "//",
        "// 【注意事項】",
        "// - 如果出現重複節點，先執行: MATCH (n) DETACH DELETE n;",
        "// - 社群版不支援 CREATE CONSTRAINT，已移除",
        "// ====================================================================",
        "",
        "// 先清除舊資料 (可選)",
        "// MATCH (n) DETACH DELETE n;",
        "",
    ]
    clip_names = sorted(events_by_clip.keys())
    lines.append("// === 建立時間軸 (Clip 節點) ===")
    for clip in clip_names:
        day, cid, time_str = parse_clip_name(clip)
        props = [f"name: '{clip}'", f"day: '{day}'"]
        if time_str:
            props.append(f"time: '{time_str}'")
        lines.append(f"MERGE (c_{clip}:Clip {{{', '.join(props)}}});")
    lines.append("")
    lines.append("// === 時間順序連接 Clip ===")
    for i in range(1, len(clip_names)):
        prev = clip_names[i-1]
        cur = clip_names[i]
        lines.append(f"MATCH (c1:Clip {{name: '{prev}'}}), (c2:Clip {{name: '{cur}'}})")
        lines.append(f"MERGE (c1)-[:NEXT_CLIP]->(c2);")
    lines.append("")
    for clip_name, events in events_by_clip.items():
        lines.append(f"// --- {clip_name} ---")
        person_data = {}
        for ev in events:
            person = ev.get('person', 'Unknown')
            if person == 'Unknown':
                continue
            if person not in person_data:
                person_data[person] = {"actions": set(), "emotions": {}, "hois": set()}
            typ = ev.get('type')
            if typ == 'Action':
                person_data[person]["actions"].add(ev.get('action'))
            elif typ == 'Emotion':
                emo = ev.get('emotion')
                conf = ev.get('confidence', 0)
                if conf > 0.15:
                    person_data[person]["emotions"][emo] = max(person_data[person]["emotions"].get(emo, 0), conf)
            elif typ == 'HOI':
                person_data[person]["hois"].add((ev.get('action'), ev.get('object')))
        for person, data in person_data.items():
            lines.append(f"MERGE (p:Person {{name: '{person}'}});")
            lines.append(f"MATCH (p:Person {{name: '{person}'}}), (c:Clip {{name: '{clip_name}'}})")
            lines.append(f"MERGE (p)-[:APPEARS_IN]->(c);")
            for action in data["actions"]:
                lines.append(f"MERGE (a:Action {{name: '{action}'}});")
                day, cid, time_str = parse_clip_name(clip_name)
                time_prop = f", time: '{time_str}'" if time_str else ""
                lines.append(f"MERGE (p)-[:PERFORMS {{clip: '{clip_name}'{time_prop}}}]->(a);")
            for emo, conf in data["emotions"].items():
                lines.append(f"MERGE (e:Emotion {{name: '{emo}'}});")
                lines.append(f"MATCH (p:Person {{name: '{person}'}}), (e:Emotion {{name: '{emo}'}})")
                day, cid, time_str = parse_clip_name(clip_name)
                time_prop = f", time: '{time_str}'" if time_str else ""
                lines.append(f"MERGE (p)-[:FEELS {{clip: '{clip_name}', confidence: {conf:.2f}{time_prop}}}]->(e);")
            for act, obj in data["hois"]:
                lines.append(f"MERGE (o:Object {{name: '{obj}'}});")
                lines.append(f"MATCH (p:Person {{name: '{person}'}}), (o:Object {{name: '{obj}'}})")
                day, cid, time_str = parse_clip_name(clip_name)
                time_prop = f", time: '{time_str}'" if time_str else ""
                lines.append(f"MERGE (p)-[:INTERACTS_WITH {{action: '{act}', clip: '{clip_name}'{time_prop}}}]->(o);")
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
