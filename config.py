from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataGeneratorConfig:
    amount: int = 3
    raw_data_dir: Path = Path("./datasets")
    

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/media/lawrence/Work/checkpoints/openvla-7b"   # Path to OpenVLA model 

    # Directory Paths
    data_root_dir: Path = Path("./datasets")        # Path to Open-X dataset directory
    dataset_name: str = "test1"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    data_path: Path = Path("./data.pt")
    run_root_dir: Path = Path("./runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("./adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 2#16                                            # Fine-tuning batch size
    max_steps: int = 200#200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5#5000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100#100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = True                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "lawrence-rs-lin"                          # Name of entity to log under

    # fmt: on