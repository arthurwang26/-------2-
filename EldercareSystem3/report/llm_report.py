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
                n_ctx=8192,
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
            blip_context = format_captions(clip, captions_by_clip, "BLIP")
            vlm_context = format_captions(clip, vlm_captions_by_clip, "VLM")
            
            # --- 1. Generate Main Report (No GT) ---
            prompt = f"""<|im_start|>system
你是長照中心院長。你需要根據提供的資料，輸出兩份不同用途的報表。絕對不要使用任何外部正確解答(Ground Truth)來推理。

【核心規則與限制】
1. **結構嚴格劃分**：你必須嚴格按照以下兩個區塊輸出。
2. **家屬報告區塊**：用語要像長照中心院長讓家屬知道老人在幹嘛的感覺，語氣要自然像真人回答，不能太生硬。絕對遵守以下限制：
   - 只能敘述真實確認發生的物理事實，絕對不能使用「可能」、「或許」、「似乎」等猜測性字眼。
   - 絕對禁止使用任何帶有情緒、心理狀態或人際關係推測的詞彙（如：冷戰、疏離、情緒低落、試探、和好等）。
   - 絕對不要透露技術層面的除錯資訊。
   - **【反骨架幻覺規則 (極重要)】**：骨架辨識模型(Action)非常容易因為下半身被桌子或物體遮擋，而將「站立」或「行走」錯誤判斷為「坐著(Sitting)」。如果在【行為辨識】看到某人是「坐著」，但【VLM 判斷】或【BLIP 判斷】的整體畫面描述是「兩人行走」、「交談」、「走進長廊」、「站立」等，你必須**絕對相信全域影像的描述**，將該人物的動作修正為「站立」或「行走」，並在【多重驗證推理】中明確指出骨架模型產生了「坐著」的誤判。
   - **【反過度推論禁令】**
     * 物件推論：如果 YOLO-World 未偵測到某物件（如書本、卡片、水杯、手機），絕對不能在家屬報告中提及該物件。如果 VLM/BLIP 說有拿書但 YOLO 沒看到，那就是幻覺，絕對不能寫。
     * 空間推論：不能推論具體地點名稱（如護理室、客廳、房間），只能說「在室內」或「走廊」。
     * 動機推論：不能推論人物的動機或意圖（如「想要和好」、「驚醒」、「好像在等什麼」）。
     * 時間推論：不能推論影片前後發生了什麼（如「剛吃完早餐」）。
     * 互動推論：如果兩人沒有物理接觸、沒有明確在交談，不能說他們在「聊天」或「坐在一起」。
     * 偏好推論：不能推論人物的喜好。
3. **工程師對比區塊**：必須明確列出各個模型的原始輸出（行為辨識、人與物互動、VLM判斷、BLIP判斷、YOLO-World環境物件）。
4. **多重驗證 (Multiple Verification)**：在工程師區塊中，所有的感測輸出(Action, HOI)都只是初步參考，最終的行為判斷必須由你進行邏輯推理後得出，絕對不能盲目相信單一感測器。如果 HOI 或 VLM 偵測到物件，但 YOLO-World 沒有偵測到，則該物件為幻覺，不得採信。

<|im_end|>
<|im_start|>user
當前影片片段：{clip}

前端感測紀錄與去幻覺驗證結果：
{clip_context}

BLIP 視覺描述紀錄：
{blip_context}

SmolVLM 視覺描述紀錄：
{vlm_context}

知識圖譜關聯：
{kg_text}

請嚴格依照以下格式輸出：

=== 工程師對比區塊 ===
【模型原始輸出】
- 行為辨識 (Action): (列出感測紀錄中的 Action 結果)
- 人與物互動 (HOI): (列出感測紀錄中的 HOI 結果)
- 實際環境物件 (YOLO-World): (列出感測紀錄中的環境物件偵測統計)
- VLM 判斷: (列出 SmolVLM 的描述)
- BLIP 判斷: (列出 BLIP 的描述)

【多重驗證推理】
(分析上述來源是否有衝突？何者比較合理？並給出最終推論的真實行為判斷。)

=== 家屬報告 ===
(根據你在【多重驗證推理】得出的最終真實行為，以溫暖、日常的口吻，告訴家屬長輩在這個片段中正在做什麼、精神狀況如何。不要提及任何AI、感測器或系統字眼。)

<|im_end|>
<|im_start|>assistant
"""
            try:
                if isinstance(self.llm, str):
                    report = f"=== 家屬報告 ===\n長輩今天狀況良好。\n=== 工程師對比區塊 ===\n測試分析。"
                else:
                    response = self.llm(prompt, max_tokens=1500, temperature=0.2, top_p=0.9, stop=["<|im_end|>"])
                    report = response["choices"][0]["text"].strip()
                    if "<think>" in report:
                        report = re.sub(r'<think>.*?</think>', '', report, flags=re.DOTALL).strip()
                
                logger.info(f"Generated concise report for {clip}")
                
                parts = report.split("=== 家屬報告 ===")
                eng_report = parts[0].strip()
                family_report = parts[1].strip() if len(parts) > 1 else ""
                
                day_match = re.search(r'(day\d+)', clip, re.IGNORECASE)
                day_str = day_match.group(1).lower() if day_match else "dayX"
                
                # Get present persons
                present_persons = set(ev.get("person") for ev in events_by_clip[clip] if ev.get("person", "Unknown") != "Unknown")
                
                # Global Append
                for person in present_persons:
                    with open(llm_dir / f"{person}_{day_str}_家屬報告.txt", "a", encoding="utf-8") as f:
                        f.write(f"{family_report}\n\n")
                    if eng_report:
                        with open(llm_dir / f"{person}_{day_str}_工程師除錯.txt", "a", encoding="utf-8") as f:
                            f.write(f"\n--- 【來源：{clip}】 ---\n{eng_report}\n")
                            
                # Per Clip Save (Combined)
                with open(per_clip_dir / f"{clip}.txt", "w", encoding="utf-8") as f:
                    f.write(f"=== 家屬報告 ===\n{family_report}\n\n{eng_report}")
                            
            except Exception as e:
                logger.error(f"Generation failed for {clip}: {e}")
                report = "Error"
                eng_report = "Error"

            # --- 2. Generate GT Analysis Report ---
            gt_prompt = f"""<|im_start|>system
你是系統評估工程師。你的任務是將「系統多重驗證後的推論結果」與「真實答案 (Ground Truth)」進行嚴格比對，並產出差異分析報告。

<|im_end|>
<|im_start|>user
當前影片片段：{clip}

【系統推論結果】
{eng_report}

【正確答案參考 (Ground Truth)】
{ground_truth_text}

請針對此片段，比對系統推論與 Ground Truth 的差異，指出系統是否誤判了行為或情緒，並分析可能的盲點原因。
請直接輸出分析內容。
<|im_end|>
<|im_start|>assistant
"""
            try:
                if isinstance(self.llm, str):
                    gt_analysis = "GT 差異分析測試。"
                else:
                    response = self.llm(gt_prompt, max_tokens=1000, temperature=0.2, top_p=0.9, stop=["<|im_end|>"])
                    gt_analysis = response["choices"][0]["text"].strip()
                    if "<think>" in gt_analysis:
                        gt_analysis = re.sub(r'<think>.*?</think>', '', gt_analysis, flags=re.DOTALL).strip()
                
                logger.info(f"Generated GT Analysis for {clip}")
                
                # ---- Disable GT Analysis to prevent harsh criticism as requested by user ----
                # for person in ["王奶奶", "陳爺爺"]:
                #     with open(llm_dir / f"{person}_{day_str}_GT差異分析.txt", "a", encoding="utf-8") as f:
                #         f.write(f"\n--- 【來源：{clip}】 ---\n{gt_analysis}\n")
                        
            except Exception as e:
                logger.error(f"GT Analysis failed for {clip}: {e}")

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
