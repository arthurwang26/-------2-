import torch
import gc
import shutil
from pathlib import Path
from .logger import get_logger

logger = get_logger("vram_guard")


def enforce_vram_clear():
    """強制清除所有未參照的 GPU tensors，清空 CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.debug(f"VRAM Cleared -> Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")


def clear_all_cache(project_root: Path):
    """清除所有 cache 和 __pycache__"""
    for d in project_root.rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
    cache_dir = project_root / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    logger.info("All caches cleared.")


class ModelGuard:
    """Context manager to ensure VRAM is cleared before and after a model is loaded."""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        logger.debug(f"[{self.name}] Preparing to load model...")
        enforce_vram_clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f"[{self.name}] Unloading model and clearing VRAM...")
        enforce_vram_clear()
