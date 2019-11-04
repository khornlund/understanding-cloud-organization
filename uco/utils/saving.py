from pathlib import Path
import datetime

TRAINING_DIR = "training"
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
RUN_DIR = "runs"


def ensure_exists(p: Path) -> Path:
    """
    Helper to ensure a directory exists.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def datetime_path(config: dict) -> Path:
    start_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    p = Path(config["save_dir"]) / TRAINING_DIR / start_time
    return ensure_exists(p)


def get_log_path(config: dict) -> Path:
    p = Path(config["save_dir"]) / LOG_DIR
    return ensure_exists(p)


def get_trainer_paths(config: dict) -> Path:
    """
    Returns the paths to save checkpoints and tensorboard runs. eg.

    .. code::

        saved/1002-123456/checkpoints
        saved/1002-123456/runs
    """
    parent = datetime_path(config)
    return (ensure_exists(parent / CHECKPOINT_DIR), ensure_exists(parent / RUN_DIR))
