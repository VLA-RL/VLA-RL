from transformers import AutoModelForVision2Seq
from peft import PeftModel
from finetune_config import FinetuneConfig
import argparse
import torch

def merge_model(base_model_path, adapter_path, output_path):
    base_model = AutoModelForVision2Seq.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map = "auto")
    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(output_path)

def parse_args():
    cfg = FinetuneConfig()
    parser = argparse.ArgumentParser(description="Merge adapter with base model")
    parser.add_argument('--base_model_path', type=str, default=cfg.vla_path)
    parser.add_argument('--adapter_path', type=str, default=cfg.adapter_dir)
    parser.add_argument('--output_path', type=str, default=f"{cfg.vla_path}/adapter-merged")
    return parser.parse_args()

def main():
    args = parse_args()
    base_model_path = args.base_model_path
    adapter_path = "adapter-tmp/openvla-7b+test1+b2+lr-2e-05+lora-r4+dropout-0.0+q-4bit"
    output_path = args.output_path
    merge_model(base_model_path, adapter_path, output_path)

if __name__ == "__main__":
    main()