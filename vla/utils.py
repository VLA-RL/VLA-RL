
"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
From https://github.com/openvla/openvla/blob/main/prismatic/util/data_utils.py
"""
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

@dataclass
class PaddedCollatorForPosePrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        actions = [instance["actions"] for instance in instances]
        target_item_poses = [instance['target_item_poses'] for instance in instances]
        basket_positions = [instance['basket_positions'] for instance in instances]
        gripper_poses = [instance['gripper_poses'] for instance in instances]

        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        actions = torch.stack(actions)
        target_item_poses = torch.stack(target_item_poses)
        basket_positions = torch.stack(basket_positions)
        gripper_poses = torch.stack(gripper_poses)


        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions = actions,
            target_item_poses = target_item_poses,
            basket_positions = basket_positions,
            gripper_poses = gripper_poses
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        actions = [instance["actions"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        actions = torch.stack(actions)

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions = actions
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
    
class runningLoss:
    def __init__(self):
        self.nll_loss = None
        self.object_position_loss = None
        self.object_orientation_loss = None
        self.target_position_loss = None
        self.gripper_position_loss = None
        self.gripper_orientation_loss = None
        self.gripper_open_loss = None
        self.action_position_loss = None
        self.action_orientation_loss = None
        self.action_open_loss = None
        self.alpha = 0.1

    def update(self, nll_loss, object_position_loss, gripper_position_loss, gripper_orientation_loss, gripper_open_loss, action_position_loss, action_orientation_loss, action_open_loss):
        if self.nll_loss is None:
            self.nll_loss = nll_loss.item()
            self.object_position_loss = object_position_loss.item()
            # self.object_orientation_loss = object_orientation_loss.item()
            # self.target_position_loss = target_position_loss.item()
            self.gripper_position_loss = gripper_position_loss.item()
            self.gripper_orientation_loss = gripper_orientation_loss.item()
            self.gripper_open_loss = gripper_open_loss.item()
            self.action_position_loss = action_position_loss.item()
            self.action_orientation_loss = action_orientation_loss.item()
            self.action_open_loss = action_open_loss.item()
        
        else:
            self.nll_loss = (1-self.alpha) * self.nll_loss + self.alpha * nll_loss.item()
            self.object_position_loss = (1-self.alpha) * self.object_position_loss + self.alpha * object_position_loss.item()
            # self.object_orientation_loss = (1-self.alpha) * self.object_orientation_loss + self.alpha * object_orientation_loss.item()
            # self.target_position_loss = (1-self.alpha) * self.target_position_loss + self.alpha * target_position_loss.item()
            self.gripper_position_loss = (1-self.alpha) * self.gripper_position_loss + self.alpha * gripper_position_loss.item()
            self.gripper_orientation_loss = (1-self.alpha) * self.gripper_orientation_loss + self.alpha * gripper_orientation_loss.item()
            self.gripper_open_loss = (1-self.alpha) * self.gripper_open_loss + self.alpha * gripper_open_loss.item()
            self.action_position_loss = (1-self.alpha) * self.action_position_loss + self.alpha * action_position_loss.item()
            self.action_orientation_loss = (1-self.alpha) * self.action_orientation_loss + self.alpha * action_orientation_loss.item()
            self.action_open_loss = (1-self.alpha) * self.action_open_loss + self.alpha * action_open_loss.item()

        
        normalized_loss = {
            'nll_loss': nll_loss/self.nll_loss,
            'object_position_loss': object_position_loss/self.object_position_loss,
            # 'object_orientation_loss': object_orientation_loss/self.object_orientation_loss,
            # 'target_position_loss': target_position_loss/self.target_position_loss,

            'gripper_position_loss': gripper_position_loss/self.gripper_position_loss,
            'gripper_orientation_loss': gripper_orientation_loss/self.gripper_orientation_loss,
            'gripper_open_loss': gripper_open_loss/self.gripper_open_loss,
            
            'action_position_loss': action_position_loss/self.action_position_loss,
            'action_orientation_loss': action_orientation_loss/self.action_orientation_loss,
            'action_open_loss': action_open_loss/self.action_open_loss,
        }

        normalized_loss.update({'total_loss': 
            0.5*normalized_loss['nll_loss'] + 0.1*normalized_loss['object_position_loss'] +
              0.1*normalized_loss['gripper_position_loss'] + 0.1*normalized_loss['gripper_orientation_loss'] +
                0.1*normalized_loss['action_position_loss'] + 0.1*normalized_loss['action_orientation_loss'] })

        return normalized_loss

class AngleLoss(nn.Module):
    def __init__(self):
        super(AngleLoss, self).__init__()

    def forward(self, pred, target):
        loss = torch.abs(pred - target)
        loss = torch.min(loss, 2*torch.pi - loss)
        loss = torch.mean(loss**2)
        return loss

