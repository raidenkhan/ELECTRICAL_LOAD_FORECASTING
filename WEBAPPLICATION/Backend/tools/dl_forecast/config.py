"""Configuration for deep learning forecast training."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # Data
    data_path: str = "ecg_demand_2018_2026.csv"
    input_window: int = 168  # 7 days of history
    forecast_horizon: int = 24

    # Training
    batch_size: int = 4096
    max_epochs: int = 200
    patience: int = 15  # early stopping
    lr: float = 1e-3
    lr_min: float = 1e-6
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    warmup_epochs: int = 5

    # GPU optimization
    mixed_precision: bool = True  # AMP FP16
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    compile_model: bool = True  # torch.compile
    gradient_accumulation_steps: int = 1

    # Model selection
    model: Literal["lstm", "gru", "transformer", "tcn", "dlinear"] = "lstm"

    # LSTM/GRU params
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.2

    # Transformer params
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 512
    transformer_dropout: float = 0.1

    # TCN params
    tcn_channels: list = field(default_factory=lambda: [128, 64, 64, 32])
    tcn_kernel_size: int = 7

    # DLinear params
    dlinear_kernel: int = 25  # moving average kernel for decomposition

    # Folds (same as the paper's 6-fold CV)
    folds: list = field(default_factory=lambda: [
        ("Fold_1", "2018-01-01", "2019-12-31", "2020-01-01", "2020-06-30"),
        ("Fold_2", "2018-01-01", "2020-12-31", "2021-01-01", "2021-06-30"),
        ("Fold_3", "2018-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),
        ("Fold_4", "2018-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
        ("Fold_5", "2018-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
        ("Fold_6", "2018-01-01", "2024-12-31", "2025-01-01", "2025-06-30"),
    ])
