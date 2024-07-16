from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from vla.base_vision import ImageTransform
from typing import Any, Dict, Tuple, Type
from vla.base_prompter import PromptBuilder
import torch
from PIL import Image
import json
from matplotlib import pyplot as plt

IGNORE_INDEX = -100

class RLbenchDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.data = torch.load(data_path)
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "rlbench": {
                "action": {"q01": np.array(self.data['actions']).min(0), 
                           "q99": np.array(self.data['actions']).max(0)}
            }
        }

    def __len__(self):
        return len(self.data['actions'])

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(plt.imread(self.data['image_paths'][idx]) * 255.0, dtype=np.uint8))
        action = np.asarray(self.data['actions'][idx], dtype=np.float32)
        instruction = self.data['language_instructions'][idx]

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},#TODO: COT
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)


        # TODO:Check if this is correct
        # input_ids[- (len(action) + 1):] = 32000
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        data = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, actions=torch.tensor(action))

        return data

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    # overwatch.info(f"Saved dataset statistics file at path {out_path}")