import os
import glob
from pathlib import Path

# Paths to update
docs = [
    "README.md",
    "AI_HANDOVER.md",
    "V5.1_Architecture_Deep_Dive.md",
    "EldercareSystem_Presentation.md"
]

replacements = {
    "MotionBERT 被閹割成只輸出 12 種標籤": "MotionBERT 已完全釋放，使用官方預訓練權重直接輸出完整的 60 種原生動作標籤 (Zero-Shot)",
    "MotionBERT 透過 mapping": "MotionBERT 使用原生 60 分類",
    "沒有時序資料庫": "新增了 SQLite based 的 Time-Series Database (ts_db.py) 負責記錄隨時間變化的異常分數與情緒信心度",
    "影片長度被截斷": "移除了 150 幀的擷取限制，系統可完整處理全段影片",
    "CLIP 的假設沒有被二次驗證": "VLM (SmolVLM2) 已經實作了 Grounding 交叉比對機制，針對 CLIP 提出的 HOI 互動假設進行 'Yes/No' 驗證以消除幻覺"
}

def update_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        modified = False
        for k, v in replacements.items():
            if k in content:
                content = content.replace(k, v)
                modified = True
                
        # Append V5.1 status if not present
        if "V5.1 真・最終定案" not in content:
            content = "\n\n> **V5.1 真・最終定案更新**: 包含完整 MotionBERT 60 標籤、SQLite 時序資料庫、VLM Grounding 交叉驗證、以及解除 150 幀限制與模組級 Debug JSON 輸出。\n" + content
            modified = True
            
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated {filepath}")
    except Exception as e:
        print(f"Error updating {filepath}: {e}")

if __name__ == "__main__":
    base_dir = r"C:\Users\arthu\Desktop\新增資料夾 (2)\EldercareSystem2"
    for doc in docs:
        filepath = os.path.join(base_dir, doc)
        if os.path.exists(filepath):
            update_file(filepath)
    
    # Also update any .md in docs/
    for doc in glob.glob(os.path.join(base_dir, "docs", "*.md")):
        update_file(doc)
    print("Done updating docs.")
