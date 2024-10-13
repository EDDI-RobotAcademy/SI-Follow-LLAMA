from dataclasses import dataclass


@dataclass
class TrainerArgs:
    output_dir: str = "finetuned_models/test_checkpoints"
    run_name: str = "test"
    log_interval: int = 1
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "constant"
    eval_ratio: float = 0.1
    save_ratio: float = 0.1
    fp16: bool = True
    eval_strategy: str = "steps"
    remove_unused_columns: bool = False
    report_to: str = "none"
