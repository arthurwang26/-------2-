import os
import re
from pathlib import Path
from llama_cpp import Llama

base_dir = Path(r"C:\Users\arthu\Desktop\新增資料夾 (2)\EldercareSystem3")
per_clip_dir = base_dir / "output3" / "llm_reports" / "per_clip"
llm_dir = base_dir / "output3" / "llm_reports"
qwen_path = r"C:\Users\arthu\Desktop\新增資料夾 (2)\Qwen3-4B-Instruct-2507-Q5_K_M.gguf"

clips = sorted(os.listdir(per_clip_dir))
daily_collections = {}

for clip_file in clips:
    if not clip_file.endswith(".txt"): continue
    clip_name = clip_file.replace(".txt", "")
    day_match = re.search(r'(day\d+)', clip_name, re.IGNORECASE)
    day_str = day_match.group(1).lower() if day_match else "dayX"
    
    with open(per_clip_dir / clip_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    for person in ["王奶奶", "陳爺爺"]:
        if person in text:
            key = (person, day_str)
            if key not in daily_collections:
                daily_collections[key] = []
            # Truncate to first 1500 chars to avoid overwhelming context even with higher limit
            daily_collections[key].append(f"【{clip_name}】\n{text[:1500]}...")

print("Loading Qwen with expanded context (16384)...")
llm = Llama(
    model_path=qwen_path,
    n_ctx=16384,
    n_gpu_layers=-1,
    verbose=False
)

for (person, day_str), logs in daily_collections.items():
    daily_logs_text = "\n\n".join(logs)
    daily_prompt = f"""<|im_start|>system
你是一位長照中心護理長。請根據長輩今天的詳細活動紀錄，總結成一份給家屬的【一日報表】。

【嚴格遵守事項】
1. **極度精簡與溫暖**：不需要詳細的分析，只需要簡單、精簡、溫暖地總結今日重點作息即可。
2. **語氣自然**：用語要溫暖，像是在用 Line 傳訊息給家屬報平安。
3. **絕對禁止科技術語**：絕對不可提及「AI」、「感測器」、「視覺模型」、「VLM」、「YOLO」等詞彙。

<|im_end|>
<|im_start|>user
長輩姓名：{person}
日期：{day_str}

今日活動紀錄摘要：
{daily_logs_text}

請直接輸出給家屬的一日報表文字，不要有任何多餘的標題：
<|im_end|>
<|im_start|>assistant
"""
    try:
        response = llm(daily_prompt, max_tokens=1000, temperature=0.2, top_p=0.9, stop=["<|im_end|>"])
        daily_summary = response["choices"][0]["text"].strip()
        if "<think>" in daily_summary:
            daily_summary = re.sub(r'<think>.*?</think>', '', daily_summary, flags=re.DOTALL).strip()
        
        out_path = llm_dir / f"{person}_{day_str}_一日報表.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{daily_summary}\n")
        print(f"Generated daily summary for {person} on {day_str}")
    except Exception as e:
        print(f"Daily summary failed for {person}: {e}")
