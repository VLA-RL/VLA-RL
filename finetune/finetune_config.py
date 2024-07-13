
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/media/lawrence/Work/checkpoints/openvla-7b"   # Path to OpenVLA model 
    vla_path_q: str = "/media/lawrence/Work/checkpoints/openvla-7b-4b-wo-projector"   # Path to OpenVLA model


    dataset_name: str = "test1"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    data_path: Path = Path("./datasets/data.pt")
    train_data_path: Path = Path("./datasets/train_data.pt")
    test_data_path: Path = Path("./datasets/test_data.pt")
    valid_data_path: Path = Path("./datasets/valid_data.pt")
    run_root_dir: Path = Path("./runs")                               # Path to directory to store logs & checkpoints
    adapter_dir: Path = Path("./adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    episode: int = 5
    batch_size: int = 2#16                                            # Fine-tuning batch size
    save_steps: int = 10#5000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 5                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100#100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 4#32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = True                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    #quantization config
    

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "lawrence-rs-lin-university-of-toronto"                           # Name of entity to log under

    # fmt: on