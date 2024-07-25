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

# torchrun --standalone --nnodes 1 --nproc-per-node 1 finetune/finetune_cot_weighted_loss.py
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
from vla.utils import PaddedCollatorForPosePrediction, runningLoss
from vla.action_tokenizer import RLbenchPoseTokenizer
from vla.dataset import RLbenchCotDataset
import numpy as np
import torch.nn.functional as F


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/media/lawrence/Work/checkpoints/ecot-openvla-7b-bridge"   # Path to OpenVLA model 
    vla_path_q: str = "/media/lawrence/Work/checkpoints/openvla-cot-4b"   # Path to OpenVLA model

    experiment_name: str = "nll_loss"
    dataset_name: str = "pick_described_object"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    # data_path: Path = Path(f"./datasets/{dataset_name}/data.pt")
    train_data_path: Path = Path(f"./datasets/{dataset_name}/train_data.pt")
    test_data_path: Path = Path(f"./datasets/{dataset_name}/test_data.pt")
    run_root_dir: Path = Path("./runs")                               # Path to directory to store logs & checkpoints
    adapter_dir: Path = Path("./adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    seed: int = 42                                                  # Random seed
    episode: int = 5
    batch_size: int = 2#16                                            # Fine-tuning batch size
    test_batch_size: int = 2
    test_limit_length: int = 30
    save_steps: int = 20#5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-5                                     # Fine-tuning learning rate
    weight_decay: float = 0.01                                       # Fine-tuning weight decay
    grad_accumulation_steps: int = 4                                # Gradient accumulation steps

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 16#32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = True                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance
    dataset_statistics: tuple = (np.array([-0.20173775, -0.36754665,  0.81396234, -3.14153998, -0.38798628, -3.14158631,  0. ]), np.array([0.41802976, 0.45118147, 1.47966564, 3.14159215, 0.30391057, 3.14157801, 1.])) # Min-Max normalization statistics

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
        f"{cfg.experiment_name}+{cfg.dataset_name}+e{cfg.episode}"
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
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map = "auto"
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
        prompt_builder_fn=PurePromptBuilder,
    )

    testset = RLbenchCotDataset(
        cfg.test_data_path,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    # Create Collator and DataLoader
    collator = PaddedCollatorForPosePrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_dataloader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=None,
        collate_fn=collator,
        num_workers=1,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    test_dataloader = DataLoader(
        testset,
        batch_size=cfg.test_batch_size,
        shuffle=True,
        sampler=None,
        collate_fn=collator,
        num_workers=1,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
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
                total_step = step_idx + 1 + epoch * train_dataloader.__len__()
                vla.train()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    train_nll_loss = output.loss
                
                    output_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches:-1]
                    output_gt = batch["labels"][:, 1:].to(device_id)
                    action_mask, gripper_mask, object_mask, target_mask = action_tokenizer.get_mask(output_gt)

                    action_logits = output_logits[action_mask][:,action_tokenizer.action_token_begin_idx:processor.tokenizer.vocab_size].view(cfg.batch_size,-1,action_tokenizer.n_bins)
                    gripper_logits = output_logits[gripper_mask][:,action_tokenizer.action_token_begin_idx:processor.tokenizer.vocab_size].view(cfg.batch_size,-1,action_tokenizer.n_bins)
                    object_logits = output_logits[object_mask][:,action_tokenizer.action_token_begin_idx:processor.tokenizer.vocab_size].view(cfg.batch_size,-1,action_tokenizer.n_bins)
                    target_logits = output_logits[target_mask][:,action_tokenizer.action_token_begin_idx:processor.tokenizer.vocab_size].view(cfg.batch_size,-1,action_tokenizer.n_bins)

                    # object
                    pred_object = action_tokenizer.decode(object_logits, soft = True)
                    gt_object = batch['target_item_poses'].to(device_id)
                    assert pred_object.shape == gt_object.shape, f"Object shape {pred_object.shape} != {gt_object.shape}"
                    object_position_loss = F.mse_loss(pred_object[:,:3], gt_object[:,:3].to(torch.float32))
                    # object_orientation_loss = F.mse_loss(pred_object[:,3:], gt_object[:,3:].to(torch.float32))

                    # target
                    pred_target = action_tokenizer.decode(target_logits, soft = True)
                    gt_target = batch['basket_positions'].to(device_id)
                    assert pred_target.shape == gt_target.shape, f"Target shape {pred_target.shape} != {gt_target.shape}"
                    # target_position_loss = F.mse_loss(pred_target[:,:3], gt_target[:,:3].to(torch.float32))

                    # gripper
                    pred_gripper = action_tokenizer.decode(gripper_logits, soft = True)
                    gt_gripper = batch['gripper_poses'].to(device_id)
                    assert pred_gripper.shape == gt_gripper.shape, f"Gripper shape {pred_gripper.shape} != {gt_gripper.shape}"
                    gripper_position_loss = F.mse_loss(pred_gripper[:,:3], gt_gripper[:,:3].to(torch.float32))
                    gripper_orientation_loss = F.mse_loss(pred_gripper[:,3:6], gt_gripper[:,3:6].to(torch.float32))
                    gripper_open_gt = torch.zeros_like(gripper_logits[:,6,-2:]).scatter_(1, gt_gripper[:,6].unsqueeze(1).to(torch.int64), 1)
                    gripper_open_loss = F.cross_entropy(gripper_logits[:,6,-2:], gripper_open_gt.to(torch.float32))

                    #action
                    pred_action = action_tokenizer.decode(action_logits, soft = True)
                    gt_action = batch['actions'].to(device_id)
                    assert pred_action.shape == gt_action.shape, f"Action shape {pred_action.shape} != {gt_action.shape}"
                    action_position_loss = F.mse_loss(pred_action[:,:3], gt_action[:,:3].to(torch.float32))
                    action_orientation_loss = F.mse_loss(pred_action[:,3:6], gt_action[:,3:6].to(torch.float32))
                    action_open_gt = torch.zeros_like(action_logits[:,6,-2:]).scatter_(1, gt_action[:,6].unsqueeze(1).to(torch.int64), 1)
                    action_open_loss = F.cross_entropy(action_logits[:,6,-2:], action_open_gt.to(torch.float32))

                # normalized_loss = train_running_loss.update(
                #     nll_loss=train_nll_loss,
                #     object_position_loss=object_position_loss,
                #     # object_orientation_loss,
                #     # target_position_loss=target_position_loss,
                #     gripper_position_loss=gripper_position_loss,
                #     gripper_orientation_loss=gripper_orientation_loss,
                #     gripper_open_loss=gripper_open_loss,
                #     action_position_loss=action_position_loss,
                #     action_orientation_loss=action_orientation_loss,
                #     action_open_loss=action_open_loss
                # )

                train_loss = train_nll_loss/cfg.grad_accumulation_steps
                scaler.scale(train_loss).backward()
                
                # Push Metrics to W&B (every 10 steps)
                if distributed_state.is_main_process and step_idx % 10 == 0:

                    wandb.log(
                        {
                            "train/nll_loss": train_nll_loss,
                            "train/object_position_loss": object_position_loss,
                            # "train/object_orientation_loss": object_orientation_loss,
                            # "train/target_position_loss": target_position_loss,
                            "train/gripper_position_loss": gripper_position_loss,
                            "train/gripper_orientation_loss": gripper_orientation_loss,
                            "train/gripper_open_loss": gripper_open_loss,
                            "train/action_position_loss": action_position_loss,
                            "train/action_orientation_loss": action_orientation_loss,
                            "train/action_open_loss": action_open_loss,
                        },
                        step = total_step
                    )

                # Optimizer Step
                if total_step % cfg.grad_accumulation_steps == 0 or step_idx == train_dataloader.__len__():
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.empty_cache()

                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                if total_step > 0 and total_step % cfg.save_steps == 0:
                    ##testing
                    vla.eval()

                    test_nll_loss = []
                    test_object_position_loss = []
                    test_object_orientation_loss = []
                    test_target_position_loss = []
                    test_gripper_position_loss = []
                    test_gripper_orientation_loss = []
                    test_gripper_open_loss = []
                    test_action_position_loss = []
                    test_action_orientation_loss = []
                    test_action_open_loss = []
                    for test_idx, batch in enumerate(test_dataloader):
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            output: CausalLMOutputWithPast = vla(
                                input_ids=batch["input_ids"].to(device_id),
                                attention_mask=batch["attention_mask"].to(device_id),
                                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                                labels=batch["labels"],
                            )
                            test_nll_loss_ = output.loss
                            test_nll_loss.append(test_nll_loss_)

                            output_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches:-1]
                            output_gt = batch["labels"][:, 1:].to(device_id)
                            action_mask, gripper_mask, object_mask, target_mask = action_tokenizer.get_mask(output_gt)

                            action_logits = output_logits[action_mask][:,action_tokenizer.action_token_begin_idx:processor.tokenizer.vocab_size].view(cfg.test_batch_size,-1,action_tokenizer.n_bins)
                            gripper_logits = output_logits[gripper_mask][:,action_tokenizer.action_token_begin_idx:processor.tokenizer.vocab_size].view(cfg.test_batch_size,-1,action_tokenizer.n_bins)
                            object_logits = output_logits[object_mask][:,action_tokenizer.action_token_begin_idx:processor.tokenizer.vocab_size].view(cfg.test_batch_size,-1,action_tokenizer.n_bins)
                            target_logits = output_logits[target_mask][:,action_tokenizer.action_token_begin_idx:processor.tokenizer.vocab_size].view(cfg.test_batch_size,-1,action_tokenizer.n_bins)

                            # object
                            pred_object = action_tokenizer.decode(object_logits, soft = True)
                            gt_object = batch['target_item_poses'].to(device_id)
                            assert pred_object.shape == gt_object.shape, f"Object shape {pred_object.shape} != {gt_object.shape}"
                            object_position_loss = F.mse_loss(pred_object[:,:3], gt_object[:,:3])
                            # object_orientation_loss = F.mse_loss(pred_object[:,3:], gt_object[:,3:])
                            test_object_position_loss.append(object_position_loss)
                            # test_object_orientation_loss.append(object_orientation_loss)

                            # target
                            pred_target = action_tokenizer.decode(target_logits, soft = True)
                            gt_target = batch['basket_positions'].to(device_id)
                            assert pred_target.shape == gt_target.shape, f"Target shape {pred_target.shape} != {gt_target.shape}"
                            # target_position_loss = F.mse_loss(pred_target[:,:3], gt_target[:,:3])
                            # test_target_position_loss.append(target_position_loss)

                            # gripper
                            pred_gripper = action_tokenizer.decode(gripper_logits, soft = True)
                            gt_gripper = batch['gripper_poses'].to(device_id)
                            assert pred_gripper.shape == gt_gripper.shape, f"Gripper shape {pred_gripper.shape} != {gt_gripper.shape}"
                            gripper_position_loss = F.mse_loss(pred_gripper[:,:3], gt_gripper[:,:3])
                            gripper_orientation_loss = F.mse_loss(pred_gripper[:,3:6], gt_gripper[:,3:6])
                            gripper_open_loss = F.mse_loss(pred_gripper[:,6], gt_gripper[:,6])
                            test_gripper_position_loss.append(gripper_position_loss)
                            test_gripper_orientation_loss.append(gripper_orientation_loss)
                            test_gripper_open_loss.append(gripper_open_loss)

                            #action
                            pred_action = action_tokenizer.decode(action_logits, soft = True)
                            gt_action = batch['actions'].to(device_id)
                            assert pred_action.shape == gt_action.shape, f"Action shape {pred_action.shape} != {gt_action.shape}"
                            action_position_loss = F.mse_loss(pred_action[:,:3], gt_action[:,:3])
                            action_orientation_loss = F.mse_loss(pred_action[:,3:6], gt_action[:,3:6])
                            action_open_loss = F.mse_loss(pred_action[:,6], gt_action[:,6])
                            test_action_position_loss.append(action_position_loss)
                            test_action_orientation_loss.append(action_orientation_loss)
                            test_action_open_loss.append(action_open_loss)

                            if test_idx >= cfg.test_limit_length:
                                break

                    test_nll_loss = torch.stack(test_nll_loss).mean()
                    test_object_position_loss = torch.stack(test_object_position_loss).mean()
                    # test_object_orientation_loss = torch.stack(test_object_orientation_loss).mean()
                    # test_target_position_loss = torch.stack(test_target_position_loss).mean()
                    test_gripper_position_loss = torch.stack(test_gripper_position_loss).mean()
                    test_gripper_orientation_loss = torch.stack(test_gripper_orientation_loss).mean()
                    test_gripper_open_loss = torch.stack(test_gripper_open_loss).mean()
                    test_action_position_loss = torch.stack(test_action_position_loss).mean()
                    test_action_orientation_loss = torch.stack(test_action_orientation_loss).mean()
                    test_action_open_loss = torch.stack(test_action_open_loss).mean()

                    wandb.log(
                        {
                            "test/nll_loss": test_nll_loss,
                            "test/object_position_loss": test_object_position_loss,
                            # "test/object_orientation_loss": test_object_orientation_loss,
                            # "test/target_position_loss": test_target_position_loss,
                            "test/gripper_position_loss": test_gripper_position_loss,
                            "test/gripper_orientation_loss": test_gripper_orientation_loss,
                            "test/gripper_open_loss": test_gripper_open_loss,
                            "test/action_position_loss": test_action_position_loss,
                            "test/action_orientation_loss": test_action_orientation_loss,
                            "test/action_open_loss": test_action_open_loss,
                        },
                        step = total_step
                    )

                    test_loss = test_nll_loss

                    if best_test_loss > test_loss:
                        print(f"Saving Model Checkpoint for Step {total_step}")
                        best_test_loss = test_loss
                        save_dir = adapter_dir if cfg.use_lora else run_dir
                        # Save Processor & Weights
                        processor.save_pretrained(run_dir)
                        vla.save_pretrained(save_dir, save_embedding_layers=True, save_adapter=True, save_config=True)
                    
                    torch.cuda.empty_cache()

                    # Block on Main Process Checkpointing
                    # dist.barrier()
                scheduler.step()
                progress.update()
            
if __name__ == "__main__":
    cfg = FinetuneConfig()
    finetune()
