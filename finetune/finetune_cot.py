"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...

From https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py
"""

# torchrun --standalone --nnodes 1 --nproc-per-node 1 finetune/finetune_cot.py
import os, sys
sys.path.append('.')

from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
import torch.distributed as dist
import tqdm
import wandb
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.optim.lr_scheduler import StepLR
from transformers import get_linear_schedule_with_warmup

from vla.base_prompter import PurePromptBuilder
from vla.utils import PaddedCollatorForPosePrediction, runningLoss, SamplerForPosePrediction
from vla.action_tokenizer import RLbenchPoseTokenizer
from vla.dataset import RLbenchCotDataset
import numpy as np
import torch.nn.functional as F


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/media/lawrence/Work/checkpoints/ecot-openvla-7b-bridge"   # Path to OpenVLA model 
    vla_path_q: str = "/media/lawrence/Work/checkpoints/openvla-cot-4b"   # Path to OpenVLA model

    experiment_name: str = "0"
    dataset_name: str = "pick_described_object"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    # data_path: Path = Path(f"./datasets/{dataset_name}/data.pt")
    train_data_path: Path = Path(f"./datasets/{dataset_name}/train_data.pt")
    test_data_path: Path = Path(f"./datasets/{dataset_name}/test_data.pt")
    item_num = 5
    stage_num = 2
    add_tokens = ['<g>', '</g>'] + [f'<item_{i}>' for i in np.arange(item_num)] + ['<o>', '</o>', '<t>', '</t>'] + [f'<stage_{i}>' for i in np.arange(stage_num)] + ['<a>', '</a>']

    run_root_dir: Path = Path("./runs")                               # Path to directory to store logs & checkpoints
    adapter_dir: Path = Path("./adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    seed: int = 42                                                  # Random seed
    episode: int = 1
    batch_size: int = 2#16                                            # Fine-tuning batch size
    test_limit_length: int = 30
    save_steps: int = 20#5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    weight_decay: float = 0.01                                      # Fine-tuning weight decay
    grad_accumulation_steps: int = 4                                # Gradient accumulation steps
    train_loss: str = "weighted"                                         # Loss to optimize during fine-tuning
    schedular : bool = False

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 16#32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = True                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance
    dataset_statistics: tuple = (np.array([-0.2, -0.35,  0.75199986, -np.pi/2, -np.pi/2, -np.pi/2,  0. ]), np.array([0.5, 0.35, 1.3, np.pi/2, 0, np.pi/2, 1.])) # Min-Max normalization statistics

    # Tracking Parameters
    wandb_project: str = "vla-rl"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "lawrence-rs-lin-university-of-toronto"                           # Name of entity to log under

    # fmt: on
cfg = FinetuneConfig()

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    torch.manual_seed(cfg.seed)

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.experiment_name}+{cfg.train_loss}+{cfg.dataset_name}+e{cfg.episode}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", #llm_int8_skip_modules = ['projector'],
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    processor.tokenizer.add_tokens(cfg.add_tokens)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map = "cuda",
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    # vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]

    # Create Action Tokenizer
    action_tokenizer = RLbenchPoseTokenizer(processor.tokenizer, cfg.dataset_statistics)

    trainset = RLbenchCotDataset(
        cfg.train_data_path,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
    )

    testset = RLbenchCotDataset(
        cfg.test_data_path,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
    )

    # Create Collator and DataLoader
    collator = PaddedCollatorForPosePrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    # train_sampler = SamplerForPosePrediction(np.array(trainset.data['stages']), group1_ratio=0.1)
    # test_sampler = SamplerForPosePrediction(np.array(testset.data['stages']), group1_ratio=0.1)

    train_dataloader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        # sampler=train_sampler,
        collate_fn=collator,
        num_workers=1,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    test_dataloader = DataLoader(
        testset,
        batch_size=cfg.batch_size,
        shuffle=True,
        # sampler=test_sampler,
        collate_fn=collator,
        num_workers=1,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    if cfg.schedular:
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * len(train_dataloader) * cfg.episode, num_training_steps=len(train_dataloader) * cfg.episode)
    scaler = torch.cuda.amp.GradScaler()

    train_running_loss = runningLoss()
    # Train!
    vla.train()
    vla.gradient_checkpointing_enable()
    best_test_loss = float("inf")
    for epoch in range(cfg.episode):
        print(f"Starting Epoch {epoch + 1} of {cfg.episode}")
        with tqdm.tqdm(total=train_dataloader.__len__() , leave=False) as progress:
            optimizer.zero_grad()
            for step_idx, batch in enumerate(train_dataloader):
                total_step = step_idx + epoch * train_dataloader.__len__()
                vla.train()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                        use_cache=False
                    )

                    output_start_idx = vla.vision_backbone.featurizer.patch_embed.num_patches
                    loss_dict = action_tokenizer.get_loss(output, batch, output_start_idx)
                    normalized_loss_dict = train_running_loss.update(loss_dict = loss_dict)

                if cfg.train_loss == "nll":
                    train_loss = loss_dict['nll_loss']/cfg.grad_accumulation_steps
                elif cfg.train_loss == "weighted":
                    train_loss = normalized_loss_dict['total_loss']/cfg.grad_accumulation_steps
                scaler.scale(train_loss).backward()
                
                # Push Metrics to W&B (every 10 steps)
                if distributed_state.is_main_process and total_step % 10 == 0:
                    log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
                    wandb.log(log_dict,
                        step = total_step
                    )

                # Optimizer Step
                if total_step % cfg.grad_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.empty_cache()

                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                if total_step % cfg.save_steps == 0:
                    ##testing
                    vla.eval()

                    test_loss_dict = {
                        "nll_loss": [],
                        # "gripper_position_loss": [],
                        # "gripper_orientation_loss": [],
                        # "gripper_open_loss": [],
                        "item_loss": [],
                        "object_position_loss": [],
                        "target_position_loss": [],
                        "stage_loss": [],
                        "action_position_loss": [],
                        "action_orientation_loss": [],
                        "action_open_loss": [],
                    }
                    for test_idx, batch in enumerate(test_dataloader):
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            output: CausalLMOutputWithPast = vla(
                                input_ids=batch["input_ids"].to(device_id),
                                attention_mask=batch["attention_mask"].to(device_id),
                                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                                labels=batch["labels"],
                            )

                            output_start_idx = vla.vision_backbone.featurizer.patch_embed.num_patches
                            loss_dict = action_tokenizer.get_loss(output, batch, output_start_idx)
                            for k, v in loss_dict.items():
                                test_loss_dict[k].append(v)

                            if test_idx >= cfg.test_limit_length:
                                break

                    for k, v in test_loss_dict.items():
                        test_loss_dict[k] = torch.stack(v).mean()
                    
                    log_dict = {f"test/{k}": v for k, v in test_loss_dict.items()}

                    wandb.log(
                        log_dict,
                        step = total_step
                    )

                    test_loss = test_loss_dict['nll_loss']

                    if best_test_loss > test_loss:
                        print(f"Saving Model Checkpoint for Step {total_step}")
                        best_test_loss = test_loss
                        save_dir = adapter_dir if cfg.use_lora else run_dir
                        processor.save_pretrained(run_dir)
                        vla.save_pretrained(save_dir, save_embedding_layers=True, save_adapter=True, save_config=True)
                    
                    torch.cuda.empty_cache()

                    # Block on Main Process Checkpointing
                    # dist.barrier()
                if cfg.schedular:
                    scheduler.step()
                progress.update()
            
if __name__ == "__main__":
    cfg = FinetuneConfig()
    finetune()
