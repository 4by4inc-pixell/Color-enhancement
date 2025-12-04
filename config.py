from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    train_input_dir: str = "./data/train/input"
    train_target_dir: str = "./data/train/target"
    val_input_dir: str = "./data/val/input"
    val_target_dir: str = "./data/val/target"
    batch_size: int = 16
    num_workers: int = 4
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-6
    model_name: str = "GlobalLocalColorEnhancer"
    device: str = "cuda"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    resume_path: Optional[str] = None
    img_size: int = 512

@dataclass
class StageConfig:
    train_input_dir: str
    train_target_dir: str
    val_input_dir: str
    val_target_dir: str
    batch_size: int
    num_workers: int
    num_epochs: int
    lr: float
    kind: str

@dataclass
class HybridConfig:
    public: StageConfig
    custom: StageConfig
    device: str
    img_size: int
    weight_decay: float
    save_dir: str
    log_dir: str
    base_channels: int
    num_colors: int

public_stage_default = StageConfig(
    train_input_dir="./data/MIT-Adobe_5K_Dataset_+/training/INPUT_IMAGES",
    train_target_dir="./data/MIT-Adobe_5K_Dataset_+/training/GT_IMAGES",
    val_input_dir="./data/MIT-Adobe_5K_Dataset_+/validation/INPUT_IMAGES",
    val_target_dir="./data/MIT-Adobe_5K_Dataset_+/validation/GT_IMAGES",
    batch_size=16,
    num_workers=4,
    num_epochs=200,
    lr=1e-4,
    kind="public",
)

custom_stage_default = StageConfig(
    train_input_dir="./data/all_dataset_nolol/train/input",
    train_target_dir="./data/all_dataset_nolol/train/target",
    val_input_dir="./data/all_dataset_nolol/val/input",
    val_target_dir="./data/all_dataset_nolol/val/target",
    batch_size=16,
    num_workers=4,
    num_epochs=300,
    lr=5e-5,
    kind="custom",
)

hybrid_cfg = HybridConfig(
    public=public_stage_default,
    custom=custom_stage_default,
    device="cuda",
    img_size=512,
    weight_decay=1e-6,
    save_dir="./checkpoints_hybrid",
    log_dir="./logs_hybrid",
    base_channels=32,
    num_colors=8,
)
