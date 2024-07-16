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

# torchrun --standalone --nnodes 1 --nproc-per-node 1 finetune/finetune_l1.py

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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from finetune_config import FinetuneConfig
from vla.base_prompter import PurePromptBuilder
from vla.utils import PaddedCollatorForActionPrediction
from vla.action_tokenizer import ActionTokenizer, RLbenchActionTokenizer
from vla.dataset import save_dataset_statistics, RLbenchDataset

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}+e{cfg.episode}"
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
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    

    # Create Action Tokenizer
    action_tokenizer = RLbenchActionTokenizer(processor.tokenizer)

    trainset = RLbenchDataset(
        cfg.train_data_path,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    validset = RLbenchDataset(
        cfg.valid_data_path,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    testset = RLbenchDataset(
        cfg.test_data_path,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    # if distributed_state.is_main_process:
    #     save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_dataloader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=2,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    valid_dataloader = DataLoader(
        validset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=2,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    test_dataloader = DataLoader(
        testset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=2,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Train!
    for epoch in range(cfg.episode):
        print(f"Starting Epoch {epoch + 1} of {cfg.episode}")
        with tqdm.tqdm(total=train_dataloader.__len__() , leave=False) as progress:
            optimizer.zero_grad()
            batch_loss = 0
            for step_idx, batch in enumerate(train_dataloader):
                vla.train()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                
                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches:-1]
                action_gt = batch["labels"][:, 1:].to("cuda")
                mask = action_gt >= action_tokenizer.action_token_begin_idx
                masked_logits = action_logits[mask][:,action_tokenizer.action_token_begin_idx:processor.tokenizer.vocab_size].view(2,7,-1)

                # Compute L1 Loss on Predicted (Continuous) Actions
                action_pred = action_tokenizer.output_logit_to_continous_action(masked_logits)
                action_gt = batch["actions"].to("cuda")
                action_l1_loss = torch.nn.functional.l1_loss(action_pred, action_gt)

                total_loss = (0.5*loss + 0.5*action_l1_loss)/cfg.grad_accumulation_steps
                total_loss.backward()

                batch_loss += total_loss.item()

                # Push Metrics to W&B (every 10 steps)
                if distributed_state.is_main_process and step_idx % 10 == 0:
                    wandb.log(
                        {"nll_loss": loss, "l1_loss": action_l1_loss, "total_loss": total_loss}, step= step_idx + epoch * train_dataloader.__len__()
                    )

                # Optimizer Step
                if (step_idx + 1) % cfg.grad_accumulation_steps == 0 or step_idx == train_dataloader.__len__():
                    optimizer.step()
                    optimizer.zero_grad()

                progress.update()

                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                if step_idx > 0 and step_idx % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {step_idx}")

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = adapter_dir if cfg.use_lora else run_dir

                        # Save Processor & Weights
                        processor.save_pretrained(run_dir)
                        vla.save_pretrained(save_dir)

                    # Block on Main Process Checkpointing
                    # dist.barrier()
            scheduler.step()

if __name__ == "__main__":
    cfg = FinetuneConfig()
    finetune()
