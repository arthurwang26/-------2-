from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProjectConfig:
    project_root: Path = Path(__file__).parent
    shared_root: Path = project_root.parent
    data_root: Path = project_root / "data"
    raw_dir: Path = shared_root / "shared_data"
    enrollment_dir: Path = data_root / "enrollment"
    weights_dir: Path = shared_root / "shared_weights"
    output_dir: Path = project_root / "output3"

    # Model paths
    yolo_model: Path = weights_dir / "yolov8s-worldv2.pt"
    pose_model_path: Path = weights_dir / "yolov8s-pose.pt"

    # Target identities
    target_names: tuple = ("王奶奶", "陳爺爺")

    # Qwen3 (Local GGUF)
    qwen_gguf_path: str = r"C:\Users\arthu\Desktop\新增資料夾 (2)\Qwen3-4B-Instruct-2507-Q5_K_M.gguf"

    # Gemini API Key
    gemini_api_key: str = "AIzaSyBSt8FC-KwlKYj-x-7uggy_GM39LDZDNJ4"

    # YOLO relevant object classes to keep for HOI
    relevant_objects: tuple = ("cup", "chair", "couch", "tv", "book", "cell phone",
                               "bottle", "dining table", "remote", "vase")

    def __post_init__(self):
        for sub in ["visuals", "reports", "debug"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

cfg = ProjectConfig()
