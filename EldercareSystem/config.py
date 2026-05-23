from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProjectConfig:
    project_root: Path = Path(__file__).parent
    data_root: Path = project_root / "data"
    raw_dir: Path = data_root / "raw"
    enrollment_dir: Path = data_root / "enrollment"
    weights_dir: Path = project_root / "weights"
    output_dir: Path = project_root / "outputs3"

    # Model paths
    yolo_model: str = "yolov8s.pt"
    pose_model_path: Path = project_root / "yolov8s-pose.pt"

    # Target identities
    target_names: tuple = ("王奶奶", "陳爺爺")

    # Qwen3 GGUF
    qwen_gguf_path: str = r"C:\Users\arthu\Downloads\skill agent\skill_agent\Qwen3-4B-Instruct-2507-Q5_K_M.gguf"

    # YOLO relevant object classes to keep for HOI
    relevant_objects: tuple = ("cup", "chair", "couch", "tv", "book", "cell phone",
                               "bottle", "dining table", "remote", "vase")

    def __post_init__(self):
        for sub in ["visuals", "reports", "debug"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

cfg = ProjectConfig()
