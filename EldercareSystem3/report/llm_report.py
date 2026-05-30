import sys
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import cfg
from utils.logger import get_logger
from utils.reset_memory import ModelGuard
from database.vector_db import ReportVectorDB

logger = get_logger("llm_report")


class QwenReporter:
    """Qwen3-4B GGUF for generating Chinese structured behavioral analysis reports."""

    def __init__(self):
        self.model_path = cfg.qwen_gguf_path
        self.llm = None
        self.vector_db = ReportVectorDB()

    def load(self):
        self.vector_db.load()
        if self.llm is not None:
            return
        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.error(f"Qwen GGUF not found at {self.model_path}")
            self.llm = "MOCK"
            return

        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=16384,
                n_gpu_layers=-1,
                verbose=False
            )
            logger.info("Qwen3 loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            self.llm = "MOCK"

    def unload(self):
        self.llm = None

    def _read_ground_truth(self) -> str:
        """Read the ground truth narrative file for reference guidance."""
        gt_path = Path(cfg.project_root) / "ground_truth_narrative.md"
        if gt_path.exists():
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read ground truth: {e}")
        return ""

    def _format_all_clips_context(self, events_by_clip: Dict[str, List[Dict]], objects_by_clip: Dict[str, List[List[Dict]]] = None) -> str:
        """Format events of all clips with chronological order, timestamps, and confidence."""
        lines = []
        fps = 24
        
        action_map = {
            "Walking": "行走", "Standing": "站立", "Sitting": "坐著",
            "Sit down": "坐下", "Stand up": "站起", "Lying Down": "躺下",
            "Fall Down": "跌倒", "Talking": "交談", "Unknown": "未知",
            "Sitting_On": "坐在", "Drinking_From": "喝", "Reading": "閱讀",
            "Watching": "觀看", "Using": "使用", "Holding": "拿著",
            "Looking_At": "看著", "Eating_At": "用餐", "Sitting_At": "坐在",
        }
        
        for clip in sorted(events_by_clip.keys()):
            parts = clip.split('_')
            day_num = "1"
            clip_num = "01"
            time_part = ""
            if len(parts) > 0 and parts[0].lower().startswith("day"):
                day_num = parts[0].lower().replace("day", "")
            if len(parts) > 1 and parts[1].lower().startswith("clip"):
                clip_num = parts[1].lower().replace("clip", "")
            if len(parts) > 2:
                time_part = f" ({parts[2]})"
            clip_id = f"Day{day_num} clip{clip_num}{time_part}"

            evts = events_by_clip[clip]
            persons = set()
            
            # Separate Action and HOI events, sorted by start_frame
            action_events = []
            hoi_events = []
            
            for ev in evts:
                p = ev.get("person", "Unknown")
                if p == "Unknown":
                    continue
                persons.add(p)
                t = ev.get("type", "")
                if t == "Action":
                    action_events.append(ev)
                elif t in ("HOI", "HOI-CLIP"):
                    hoi_events.append(ev)
            
            # Sort action events by person then start_frame
            action_events.sort(key=lambda x: (x.get("person", ""), x.get("start_frame", 0)))
            
            # Build per-person timeline
            person_timelines = {}
            for ev in action_events:
                p = ev["person"]
                if p not in person_timelines:
                    person_timelines[p] = []
                sf = ev.get("start_frame", 0)
                ef = ev.get("end_frame", 0)
                start_sec = round(sf / fps, 1)
                end_sec = round(ef / fps, 1)
                act = ev.get("action", "Unknown")
                conf = ev.get("confidence", 0)
                act_cn = action_map.get(act, act)
                person_timelines[p].append(f"{start_sec}s~{end_sec}s: {act_cn} (信心度:{conf:.0%})")
            
            # Build HOI summary
            hoi_parts = []
            for ev in hoi_events:
                p = ev.get("person", "?")
                act = ev.get("action", "?")
                obj = ev.get("object", "?")
                conf = ev.get("confidence", 0)
                act_cn = action_map.get(act, act)
                sf = ev.get("start_frame", ev.get("frame_idx", "?"))
                start_sec = round(sf / fps, 1) if isinstance(sf, (int, float)) else "?"
                hoi_parts.append(f"{p}在{start_sec}s{act_cn}{obj} (信心度:{conf:.0%})")
            
            persons_str = "、".join(persons) if persons else "無人"
            
            # Process YOLO Objects if provided
            obj_summary = ""
            if objects_by_clip and clip in objects_by_clip:
                clip_objs = objects_by_clip[clip]
                obj_counts = {}
                total_f = len(clip_objs)
                for f_objs in clip_objs:
                    seen_in_frame = set(o.get("class_name") for o in f_objs if o.get("class_name") != "person")
                    for o_name in seen_in_frame:
                        obj_counts[o_name] = obj_counts.get(o_name, 0) + 1
                
                if obj_counts:
                    obj_summary = "；".join([f"{k}:出現在{v}/{total_f}幀" for k, v in obj_counts.items()])
            
            # Build clip line
            clip_line = f"[{clip_id}] 人物：{persons_str}"
            if obj_summary:
                clip_line += f"\n  YOLO-World環境物件：{obj_summary}"
            for p, timeline in person_timelines.items():
                clip_line += f"\n  {p}動作時間軸：{'→'.join(timeline)}"
            if hoi_parts:
                clip_line += f"\n  人物互動(HOI)：{'；'.join(hoi_parts)}"
            
            lines.append(clip_line)
        return "\n".join(lines)

    def generate_report(self, events_by_clip: Dict[str, List[Dict]],
                        captions_by_clip: Dict[str, List[Dict]],
                        vlm_captions_by_clip: Dict[str, List[Dict]],
                        kg_text: str = "",
                        objects_by_clip: Dict[str, List[List[Dict]]] = None) -> str:
        if self.llm is None:
            raise RuntimeError("Model is not loaded.")

        logger.info("Generating Detailed Per-Clip Chinese Structured Report...")

        base_dir = cfg.output_dir
        llm_dir = base_dir / "llm_reports"
        per_clip_dir = llm_dir / "per_clip"
        sys_dir = base_dir / "system_docs"
        
        for d in [llm_dir, per_clip_dir, sys_dir]:
            d.mkdir(parents=True, exist_ok=True)

        ground_truth_text = self._read_ground_truth()

        clip_names = sorted(list(events_by_clip.keys()))
        all_clips_context = self._format_all_clips_context(events_by_clip, objects_by_clip)
        
        def format_captions(clip_name, caps_dict, name):
            caps = caps_dict.get(clip_name, [])
            if not caps:
                return f"無 {name} 視覺描述資料。"
            return "\n".join([f"Frame {c.get('frame_idx', '?')}: {c.get('caption', '')}" for c in caps])
        
        for clip in clip_names:
            clip_context = self._format_all_clips_context({clip: events_by_clip[clip]}, objects_by_clip)
            vlm_context = format_captions(clip, vlm_captions_by_clip, "VLM")
            
            # --- 1. Generate Per-Clip Detailed Report ---
            prompt = f"""<|im_start|>system
你是一位專業的長照中心護理長。請根據以下紀錄，為這個影片片段寫出一段**詳細的分析內容**。

【嚴格遵守事項】
1. **詳細描述**：詳盡記錄長輩在此片段內的物理動作、人機互動、移動軌跡。
2. **客觀事實**：以系統感測紀錄與VLM視覺描述為基礎，不要過度腦補內心情緒或動機。
3. **場景重建**：如果VLM提供了視覺描述或環境物件，請巧妙融合進句子中，讓分析更有畫面感。

<|im_end|>
<|im_start|>user
當前影片片段：{clip}

前端感測紀錄與去幻覺驗證結果：
{clip_context}

SmolVLM 視覺描述紀錄：
{vlm_context}

知識圖譜關聯：
{kg_text}

請直接輸出詳細的片段分析內容：

<|im_end|>
<|im_start|>assistant
"""
            try:
                if isinstance(self.llm, str):
                    report = f"長輩在此片段({clip})中，主要在走廊散步，動作正常。"
                else:
                    response = self.llm(prompt, max_tokens=2500, temperature=0.2, top_p=0.9, stop=["<|im_end|>"])
                    report = response["choices"][0]["text"].strip()
                    if "<think>" in report:
                        report = re.sub(r'<think>.*?</think>', '', report, flags=re.DOTALL).strip()
                
                logger.info(f"Generated detailed analysis for {clip}")
                detailed_report = report.strip()
                
                day_match = re.search(r'(day\d+)', clip, re.IGNORECASE)
                day_str = day_match.group(1).lower() if day_match else "dayX"
                
                present_persons = set(ev.get("person") for ev in events_by_clip[clip] if ev.get("person", "Unknown") != "Unknown")
                
                # Per Clip Save (Detailed Analysis)
                with open(per_clip_dir / f"{clip}.txt", "w", encoding="utf-8") as f:
                    f.write(f"{detailed_report}\n")
                    
                # Collect for daily personal summary
                for person in present_persons:
                    key = (person, day_str)
                    if key not in getattr(self, '_daily_collections', {}):
                        if not hasattr(self, '_daily_collections'):
                            self._daily_collections = {}
                        self._daily_collections[key] = []
                    self._daily_collections[key].append(f"【{clip}】\n{detailed_report}")
                            
            except Exception as e:
                logger.error(f"Generation failed for {clip}: {e}")
                
        # --- 2. Generate Daily Personal Reports ---
        if hasattr(self, '_daily_collections'):
            for (person, day_str), logs in self._daily_collections.items():
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

今日詳細活動紀錄：
{daily_logs_text}

請直接輸出給家屬的一日報表文字，不要有任何多餘的標題：
<|im_end|>
<|im_start|>assistant
"""
                try:
                    if isinstance(self.llm, str):
                        daily_summary = "長輩今天狀況良好，有好好吃飯與散步。"
                    else:
                        response = self.llm(daily_prompt, max_tokens=1000, temperature=0.2, top_p=0.9, stop=["<|im_end|>"])
                        daily_summary = response["choices"][0]["text"].strip()
                        if "<think>" in daily_summary:
                            daily_summary = re.sub(r'<think>.*?</think>', '', daily_summary, flags=re.DOTALL).strip()
                    
                    with open(llm_dir / f"{person}_{day_str}_一日報表.txt", "w", encoding="utf-8") as f:
                        f.write(f"{daily_summary}\n")
                    logger.info(f"Generated daily summary for {person} on {day_str}")
                except Exception as e:
                    logger.error(f"Daily summary failed for {person}: {e}")

        # --- 3. Final Report (No GT) ---
        final_prompt = f"""<|im_start|>system
你是長照監控系統的總分析師。你需要根據「所有影片片段的整合資料」，寫出一段總結性的全局客觀物理行為記錄。絕對不要參考外部的真實答案，僅憑系統感測紀錄來推理。

【核心分析要求】
1. **全局物理行為總結 (Day1 -> Day2 -> Day3)**：梳理每一天的事件與物理動作脈絡。
2. **嚴禁情緒腦補**：絕對禁止使用任何帶有情緒、心理狀態或人際關係推測的詞彙（如：冷戰、疏離、情緒低落、試探、和好等）。
3. **客觀事實**：只能描述「A與B的距離」、「A做什麼動作」、「A拿著什麼物品」，不得猜測他們內心的動機或感受。
4. **跨 Clip 來源標註**：在提到關鍵事件或系統誤判時，必須標註來源（例如：根據 clip02...）。

<|im_end|>
<|im_start|>user
全部影片的前端感測紀錄整合：
{all_clips_context}

請產出最終報告：
=== 最終報告 ===
[跨片段物理行為全局總結]
(詳細分析 Day1 到後續天數的純客觀物理行為軌跡，務必標示事件來源)
=== 最終報告_END ===
<|im_end|>
<|im_start|>assistant
"""
        try:
            if isinstance(self.llm, str):
                final_report_text = "=== 最終報告 ===\n這是測試的最終報告。\n=== 最終報告_END ==="
            else:
                response = self.llm(final_prompt, max_tokens=2500, temperature=0.2, top_p=0.9, stop=["<|im_end|>"])
                final_report_text = response["choices"][0]["text"].strip()
                if "<think>" in final_report_text:
                    final_report_text = re.sub(r'<think>.*?</think>', '', final_report_text, flags=re.DOTALL).strip()
            
            with open(llm_dir / "最終報告.txt", "w", encoding="utf-8") as f:
                f.write(final_report_text)
            logger.info("Saved Final Report (最終報告.txt).")
        except Exception as e:
            logger.error(f"Final report generation failed: {e}")
            final_report_text = "Error"
            
        # --- 4. Final Report GT Analysis ---
        final_gt_prompt = f"""<|im_start|>system
你是系統評估工程師。你的任務是將「最終報告的全局推理」與「真實答案 (Ground Truth)」進行嚴格比對，產出整體的盲點與誤差分析。

<|im_end|>
<|im_start|>user
【系統最終報告】
{final_report_text}

【正確答案參考 (Ground Truth)】
{ground_truth_text}

請綜合比對系統全局推理與 Ground Truth 的差異，指出系統在跨片段情緒演進與人物關係推演上的主要盲點與失誤。
請直接輸出分析內容。
<|im_end|>
<|im_start|>assistant
"""
        try:
            if isinstance(self.llm, str):
                final_gt_analysis = "最終報告 GT 差異分析測試。"
            else:
                response = self.llm(final_gt_prompt, max_tokens=1500, temperature=0.2, top_p=0.9, stop=["<|im_end|>"])
                final_gt_analysis = response["choices"][0]["text"].strip()
                if "<think>" in final_gt_analysis:
                    final_gt_analysis = re.sub(r'<think>.*?</think>', '', final_gt_analysis, flags=re.DOTALL).strip()
            
            # Disable final GT analysis output
            # final_gt_path = llm_dir / "最終報告_GT差異分析.txt"
            # with open(final_gt_path, "w", encoding="utf-8") as f:
            #     f.write(final_gt_analysis)
            # logger.info("Saved Final GT Analysis (最終報告_GT差異分析.txt).")
        except Exception as e:
            logger.error(f"Final GT Analysis failed: {e}")

        return final_report_text
    def _mock_report(self, unique_days):
        res = ""
        for day in unique_days:
            res += f"=== 王奶奶_DAY{day} ===\n王奶奶在Day{day}的情感軌跡模擬。\n=== 王奶奶_DAY{day}_END ===\n\n"
            res += f"=== 陳爺爺_DAY{day} ===\n陳爺爺在Day{day}的情感軌跡模擬。\n=== 陳爺爺_DAY{day}_END ===\n\n"
        res += "=== 最終報告 ===\n這是一份測試用的最終報告。\n=== 最終報告_END ==="
        return res


if __name__ == "__main__":
    dummy_events = {"day1_clip01_9點": [{"type": "Action", "action": "Walking", "person": "王奶奶"}]}
    with ModelGuard("Qwen"):
        reporter = QwenReporter()
        reporter.load()
        report = reporter.generate_report(dummy_events, {"day1_clip01_9點": 0.1}, 0.3, "Stable", {})
        logger.info(f"Report:\n{report}")
