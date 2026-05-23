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
                        anomaly_by_clip: Dict[str, float],
                        risk_score: float, trend: str,
                        captions_by_clip: Dict[str, List[Dict]]) -> str:
        if self.llm is None:
            raise RuntimeError("Model is not loaded.")

        logger.info("Generating Detailed Per-Clip Chinese Structured Report...")

        # 建立報告資料夾 (Clean up is now handled exclusively by main.py)
        base_dir = cfg.output_dir
        llm_dir = base_dir / "llm_reports"
        per_clip_dir = llm_dir / "per_clip"
        sys_dir = base_dir / "system_docs"
        
        for d in [llm_dir, per_clip_dir, sys_dir]:
            d.mkdir(parents=True, exist_ok=True)

        ground_truth_text = self._read_ground_truth()

        clip_names = sorted(list(events_by_clip.keys()))
        all_clips_context = self._format_all_clips_context(events_by_clip)
        
        # Helper to format VLM captions for a clip
        def format_captions(clip_name):
            caps = captions_by_clip.get(clip_name, [])
            if not caps:
                return "無 VLM 視覺描述資料。"
            return "\n".join([f"Frame {c.get('frame_idx', '?')}: {c.get('caption', '')}" for c in caps])
        
        # 1. 針對每個 clip，要求 LLM 生成王奶奶與陳爺爺的超詳細報告
        for clip in clip_names:
            clip_context = self._format_all_clips_context({clip: events_by_clip[clip]})
            vlm_context = format_captions(clip)
            
            prompt = f"""<|im_start|>system
你是長照監控系統的嚴謹分析師。你需要根據提供的資料，輸出「適度詳細的敘述性報表」。

【核心規則與限制】
1. **結構嚴格劃分**：你必須嚴格按照以下三個區塊輸出，區塊名稱請加上雙冒號（例如： === 感測器節點敘述 === ）。
2. **VLM 獨立原則**：VLM（視覺語言模型）的結果是完全獨立的。你在撰寫「感測器節點敘述」時，**絕對不可以**參考 VLM 的結論來推斷或修正感測器的錯誤。感測器看到什麼，你就詳細敘述什麼。
3. **最後才判斷 Ground Truth**：在第三個區塊才拿出 Ground Truth 來做比較與錯誤分析。
4. **HOI 雙架構比較**：系統目前有人機互動的雙重 AI 架構 (HOI-MLP 與 HOI-CLIP)，請特別關注兩者預測結果的差異。

【Ground Truth 參考】
{ground_truth_text}
<|im_end|>
<|im_start|>user
當前影片片段：{clip}

前端神經網路感測紀錄 (System Nodes) [包含 HOI-MLP 與 HOI-CLIP]：
{clip_context}

VLM 視覺描述紀錄 (VLM Nodes)：
{vlm_context}

请严格依照以下格式输出适度详细的叙述：

=== 感測器節點敘述 ===
(詳細敘述前端神經網路偵測到的人物、動作、情緒、物件互動軌跡。不要提及 Ground Truth，也不要提及 VLM。)

=== VLM 視覺描述 ===
(詳細敘述 VLM 模型看到的畫面內容，並用文字生動描繪出來。)

=== 誤差與對比分析 ===
(1. 感測器 vs 正確答案：指出兩者的差異，並分析感測器會發生此工程錯誤的原因。
 2. VLM vs 正確答案：指出 VLM 描述與真實情況的差異。)
<|im_end|>
<|im_start|>assistant
"""
            try:
                if isinstance(self.llm, str):
                    report = f"=== 感測器節點敘述 ===\n測試敘述。\n=== VLM 視覺描述 ===\n測試。\n=== 誤差與對比分析 ===\n測試分析。"
                else:
                    response = self.llm(prompt, max_tokens=1200, temperature=0.2, top_p=0.9, stop=["<|im_end|>"])
                    report = response["choices"][0]["text"].strip()
                    if "<think>" in report:
                        report = re.sub(r'<think>.*?</think>', '', report, flags=re.DOTALL).strip()
                
                logger.info(f"Generated concise report for {clip}")
                
                # 1. 儲存每個 clip 的獨立檔案 (per_clip)
                clip_file_path = per_clip_dir / f"{clip}.txt"
                with open(clip_file_path, "w", encoding="utf-8") as f:
                    f.write(report)
                
                # 2. 依照天數合併檔案
                day_match = re.search(r'(day\d+)', clip, re.IGNORECASE)
                day_str = day_match.group(1).lower() if day_match else "dayX"
                
                # 同時寫入王奶奶與陳爺爺的檔案
                for person in ["王奶奶", "陳爺爺"]:
                    person_file_path = llm_dir / f"{person}_{day_str}.txt"
                    with open(person_file_path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- 【來源：{clip}】 ---\n{report}\n")
                        
            except Exception as e:
                logger.error(f"Generation failed for {clip}: {e}")

        # 2. 生成最終總結報告
        final_prompt = f"""<|im_start|>system
你是長照監控系統的總分析師。你需要根據「所有影片片段的整合資料」，寫出一段總結性的跨片段因果推理與全局系統除錯分析報告。

【核心分析要求】
1. **全局推理 (Day1 -> Day2 -> Day3)**：梳理每一天的事件脈絡。
2. **情緒變化脈絡與感測誤差總結**：分析人物情緒隨時間、事件的演化，並特別總結本系統前端感測器（如情緒辨識、動作辨識）在這些天的表現中，最容易出錯的盲點是什麼？為什麼？
3. **人物關係轉變**：詳細推演衝突、冷戰、試探、和好的互動因果。
4. **趨勢延續**：基於歷史軌跡預測後續行為。
5. **跨 Clip 來源標註**：在提到關鍵事件或系統誤判時，必須標註來源（例如：根據 clip02...）。

【正確答案參考 (Ground Truth)】
{ground_truth_text}
<|im_end|>
<|im_start|>user
全部影片的前端感測紀錄整合：
{all_clips_context}

=== 跨天趨勢 ===
- 風險分數: {risk_score:.4f}
- 行為趨勢: {trend}

請產出最終報告：
=== 最終報告 ===
[跨片段人際因果推理、全局總結與系統誤差分析]
(詳細分析 Day1 到後續天數的情緒演進、關係轉變與趨勢延續，並點出系統感測器的盲點，務必標示事件來源)
=== 最終報告_END ===
<|im_end|>
<|im_start|>assistant
"""
        try:
            if isinstance(self.llm, str):
                final_report_text = "=== 最終報告 ===\n這是測試的最終報告。\n=== 最終報告_END ==="
            else:
                response = self.llm(final_prompt, max_tokens=3000, temperature=0.2, top_p=0.9, stop=["<|im_end|>"])
                final_report_text = response["choices"][0]["text"].strip()
                if "<think>" in final_report_text:
                    final_report_text = re.sub(r'<think>.*?</think>', '', final_report_text, flags=re.DOTALL).strip()
            
            # 直接存檔，不再依賴 regex（避免截斷時 _END 標籤消失導致匹配失敗）
            with open(llm_dir / "最終報告.txt", "w", encoding="utf-8") as f:
                f.write(final_report_text)
            logger.info("Saved Final Report (最終報告.txt).")
        except Exception as e:
            logger.error(f"Final report generation failed: {e}")

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
