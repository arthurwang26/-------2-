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
        if not Path(self.model_path).exists():
            logger.error(f"Qwen GGUF not found at {self.model_path}")
            self.llm = "MOCK"
            return

        from llama_cpp import Llama
        logger.info(f"Loading Qwen3-4B from {self.model_path}...")
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=35,
            n_ctx=4096,
            verbose=False
        )
        logger.info("Qwen3 loaded successfully.")

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

    def _format_all_clips_context(self, events_by_clip: Dict[str, List[Dict]]) -> str:
        """Format events of all clips compactly as requested by user."""
        lines = []
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
            actions = set()
            emotions = set()
            
            action_map = {
                "Walking": "行走",
                "Standing": "站立",
                "Sitting": "坐著",
                "Talking": "交談",
                "Sitting_On": "坐在",
                "Using": "使用",
                "Holding": "拿著",
                "Walking_With": "伴隨行走",
                "Looking_At": "看著",
                "Touching": "觸摸",
                "Arguing": "爭吵",
            }
            
            for ev in evts:
                p = ev.get("person", "Unknown")
                if p == "Unknown":
                    continue
                persons.add(p)
                t = ev.get("type", "")
                if t == "Action":
                    act = ev.get("action")
                    actions.add(f"{p}{action_map.get(act, act)}")
                elif t == "Emotion":
                    if ev.get("confidence", 0) > 0.12:
                        emo = ev.get("emotion")
                        emotions.add(f"{p}:{emo}")
                elif t == "HOI":
                    act = ev.get("action")
                    obj = ev.get("object")
                    actions.add(f"{p}{action_map.get(act, act)}{obj}")

            persons_str = "、".join(persons) if persons else "無人"
            actions_str = "、".join(actions) if actions else "無動作"
            emotions_str = "、".join(emotions) if emotions else "平靜"
            
            lines.append(f"[{clip_id}] 人物:{persons_str}, 動作:{actions_str}, 情緒:{emotions_str}")
        return "\n".join(lines)

    def generate_report(self, events_by_clip: Dict[str, List[Dict]],
                        captions_by_clip: Dict[str, List[Dict]],
                        vlm_captions_by_clip: Dict[str, List[Dict]],
                        gemini_by_clip: Dict[str, str] = None,
                        kg_text: str = "") -> str:
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
        all_clips_context = self._format_all_clips_context(events_by_clip)
        
        def format_captions(clip_name, caps_dict, name):
            caps = caps_dict.get(clip_name, [])
            if not caps:
                return f"無 {name} 視覺描述資料。"
            return "\n".join([f"Frame {c.get('frame_idx', '?')}: {c.get('caption', '')}" for c in caps])
        
        for clip in clip_names:
            clip_context = self._format_all_clips_context({clip: events_by_clip[clip]})
            blip_context = format_captions(clip, captions_by_clip, "BLIP")
            vlm_context = format_captions(clip, vlm_captions_by_clip, "VLM")
            
            # --- 1. Generate Main Report (No GT) ---
            prompt = f"""<|im_start|>system
你是長照中心院長。你需要根據提供的資料，輸出兩份不同用途的報表。絕對不要使用任何外部正確解答(Ground Truth)來推理。

【核心規則與限制】
1. **結構嚴格劃分**：你必須嚴格按照以下兩個區塊輸出。
2. **家屬報告區塊**：用語要像長照中心院長讓家屬知道老人在幹嘛的感覺，語氣要自然像真人回答，不能太生硬。絕對遵守以下限制：
   - 只能敘述真實確認發生的事，絕對不能使用「可能」、「或許」、「似乎」等猜測性字眼。
   - 絕對不能分析或提及臉部表情。
   - 絕對不能使用「我們覺得」、「我們認為」等主觀感受字眼。
   - 絕對不要透露技術層面的除錯資訊。
3. **工程師對比區塊**：必須明確列出各個模型的原始輸出（行為辨識、人與物互動、VLM判斷、BLIP判斷）。
4. **多重驗證 (Multiple Verification)**：在工程師區塊中，所有的感測輸出(Action, HOI)都只是初步參考，最終的行為判斷必須由你進行邏輯推理後得出，絕對不能盲目相信單一感測器。請根據 VLM 和 BLIP 等視覺描述來做最終裁定，輸出你認為最合理的真實行為。

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

=== 家屬報告 ===
(以溫暖、日常的口吻，告訴家屬長輩在這個片段中正在做什麼、精神狀況如何。不要提及任何AI、感測器或系統字眼。)

=== 工程師對比區塊 ===
【模型原始輸出】
- 行為辨識 (Action): (列出感測紀錄中的 Action 結果)
- 人與物互動 (HOI): (列出感測紀錄中的 HOI 結果)
- VLM 判斷: (列出 SmolVLM 的描述)
- BLIP 判斷: (列出 BLIP 的描述)

【多重驗證推理】
(分析上述來源是否有衝突？何者比較合理？並給出最終推論的真實行為判斷。)

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
                
                parts = report.split("=== 工程師對比區塊 ===")
                family_report = parts[0].replace("=== 家屬報告 ===", "").strip()
                eng_report = "=== 工程師對比區塊 ===\n" + parts[1].strip() if len(parts) > 1 else ""
                
                day_match = re.search(r'(day\d+)', clip, re.IGNORECASE)
                day_str = day_match.group(1).lower() if day_match else "dayX"
                
                # Global Append
                for person in ["王奶奶", "陳爺爺"]:
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
                
                for person in ["王奶奶", "陳爺爺"]:
                    with open(llm_dir / f"{person}_{day_str}_GT差異分析.txt", "a", encoding="utf-8") as f:
                        f.write(f"\n--- 【來源：{clip}】 ---\n{gt_analysis}\n")
                        
            except Exception as e:
                logger.error(f"GT Analysis failed for {clip}: {e}")

        # --- 3. Final Report (No GT) ---
        final_prompt = f"""<|im_start|>system
你是長照監控系統的總分析師。你需要根據「所有影片片段的整合資料」，寫出一段總結性的跨片段因果推理與全局系統除錯分析報告。絕對不要參考外部的真實答案，僅憑系統感測紀錄來推理。

【核心分析要求】
1. **全局推理 (Day1 -> Day2 -> Day3)**：梳理每一天的事件脈絡。
2. **人物關係轉變**：詳細推演衝突、冷戰、試探、和好的互動因果。
3. **趨勢延續**：基於歷史軌跡預測後續行為。
4. **跨 Clip 來源標註**：在提到關鍵事件或系統誤判時，必須標註來源（例如：根據 clip02...）。

<|im_end|>
<|im_start|>user
全部影片的前端感測紀錄整合：
{all_clips_context}

請產出最終報告：
=== 最終報告 ===
[跨片段人際因果推理與全局總結]
(詳細分析 Day1 到後續天數的情緒演進、關係轉變與趨勢延續，務必標示事件來源)
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
            
            with open(llm_dir / "最終報告_GT差異分析.txt", "w", encoding="utf-8") as f:
                f.write(final_gt_analysis)
            logger.info("Saved Final GT Analysis (最終報告_GT差異分析.txt).")
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
