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
    
class RLbenchCotDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
    ) -> None:
        self.data = torch.load(data_path)
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        # self.dataset_statistics = (np.array(self.data['gripper_poses']).min(0), np.array(self.data['gripper_poses']).min(0))

    def __len__(self):
        return len(self.data['actions'])
    
    # def get_minmax(self):
    #     return self.dataset_statistics

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.open(self.data['images'][idx])
        instruction = self.data['instructions'][idx]
        gripper = self.data['grippers'][idx]
        item = self.data['items'][idx]
        object_ = self.data['objects'][idx]
        target = self.data['targets'][idx]
        stage = self.data['stages'][idx]
        action = self.data['actions'][idx]
        
        """
        A chat between a curious user and an artificial intelligence assistant.
        The assistant gives helpful, detailed, and polite answers to the user's questions.
        USER: What action should the robot take to place the watermelon on the towel? ASSISTANT:
        """

        # prompt = "In: You are an assistant helping to control a robotic manipulator. The robot performs tasks by following a series of steps to interact with objects in its environment. What the next key pose of gripper should the robot take to {instruction}? Out: Let's think step by step, {cot} </s>"
        # prompt_chat = "You are an assistant helping to control a robotic manipulator. The robot performs tasks by following a series of steps to interact with objects in its environment. The environment includes items like soup cans and baskets, and the robot uses a gripper to pick up and move these items.\n\nInstructions format:\n- 'USER': Describes the task to be performed.\n- 'ASSISTANT': Provides a detailed step-by-step plan for the robot to execute the task.\n\nThe 'ASSISTANT' response includes:\n1. A logical step-by-step plan for the task.\n2. The current positions of relevant objects and the gripper.\n3. The current state of the gripper (whether it has grasped the object or not).\n4. The next key pose of the gripper to achieve the task.\n\nExample:\n\nUSER: What action should the robot take to pick up the soup and place it in the basket?\nASSISTANT: Let's think step by step. The plan is to move the gripper to the soup and pick it up, then move over the basket, and then place the soup in the basket. The soup is located at <object>ĉ‖호 </object>. The basket is located at <target>Ζ‖ご </target>. The gripper pose is <gripper>阳‖素군雅导弘 </gripper>. The gripper hasn't grasped the soup. So the current step is to move the gripper to the soup and pick it up. The next key pose of the gripper is <action>机‖素秀麻방弘 </action>. \n <current conversation> USER: What is the next key pose of the gripper should the robot take to {instruction}? ASSISTANT: Let's think step by step, {cot}</s>"
        # prompt_instruct = "In: You are an assistant helping to control a robotic manipulator. The robot performs tasks by following a series of steps to interact with objects in its environment."

        prompt = """In: You are an assistant helping to control a robotic manipulator. The robot performs tasks by following a series of steps to interact with objects in its environment. 
        What the next key pose of gripper should the robot take to {instruction}? Let's think step by step. 
        Out: <g>{gripper} </g>, <item_{item}>, <o>{object_} </o>, <t>{target} </t>, <stage_{stage}>, <a>{action} </a></s>
        """

        prompt = prompt.format(
            instruction = instruction,
            gripper = self.action_tokenizer(gripper),
            item = item,
            object_ = self.action_tokenizer(object_),
            target = self.action_tokenizer(target),
            stage = stage,
            action = self.action_tokenizer(action)
        )

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
        labels = list(input_ids)


        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        out_idx = (input_ids == 4451).to(torch.int).argmax().item()
        labels[:out_idx] = IGNORE_INDEX

        data = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, 
                    grippers = torch.tensor(gripper), items = torch.tensor(item), objects = torch.tensor(object_),
                    targets = torch.tensor(target), stages = torch.tensor(stage), actions = torch.tensor(action))
                    
        return data
