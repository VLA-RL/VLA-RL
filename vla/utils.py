
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
from torch.utils.data import Sampler, DataLoader, Dataset
import numpy as np
import torch.nn.functional as F


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
        grippers, items, objects,targets, stages, actions = \
            tuple([instance[key] for instance in instances] for key in ("grippers", "items", "objects", "targets", "stages", "actions"))

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
        

        grippers = torch.stack(grippers)
        items = torch.stack(items)
        objects = torch.stack(objects)
        targets = torch.stack(targets)
        stages = torch.stack(stages)
        actions = torch.stack(actions)

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            grippers = grippers,
            items = items,
            objects = objects,
            targets = targets,
            stages = stages,
            actions = actions
        )
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
        # self.gripper_position_loss = None
        # self.gripper_orientation_loss = None
        # self.gripper_open_loss = None
        self.item_loss = None
        self.object_position_loss = None
        self.target_position_loss = None
        self.stage_loss = None
        self.action_position_loss = None
        self.action_orientation_loss = None
        self.action_open_loss = None
        self.alpha = 0.1

    def update(self, loss_dict):
        if self.nll_loss is None:
            self.nll_loss = loss_dict['nll_loss'].item()
            # self.gripper_position_loss = loss_dict['gripper_position_loss'].item()
            # self.gripper_orientation_loss = loss_dict['gripper_orientation_loss'].item()
            # self.gripper_open_loss = loss_dict['gripper_open_loss'].item()
            self.item_loss = loss_dict['item_loss'].item()
            self.object_position_loss = loss_dict['object_position_loss'].item()
            self.target_position_loss = loss_dict['target_position_loss'].item()
            self.stage_loss = loss_dict['stage_loss'].item()
            self.action_position_loss = loss_dict['action_position_loss'].item()
            self.action_orientation_loss = loss_dict['action_orientation_loss'].item()
            self.action_open_loss = loss_dict['action_open_loss'].item()
        
        else:
            self.nll_loss = (1-self.alpha) * self.nll_loss + self.alpha * loss_dict['nll_loss'].item()
            # self.gripper_position_loss = (1-self.alpha) * self.gripper_position_loss + self.alpha * loss_dict['gripper_position_loss'].item()
            # self.gripper_orientation_loss = (1-self.alpha) * self.gripper_orientation_loss + self.alpha * loss_dict['gripper_orientation_loss'].item()
            # self.gripper_open_loss = (1-self.alpha) * self.gripper_open_loss + self.alpha * loss_dict['gripper_open_loss'].item()
            self.item_loss = (1-self.alpha) * self.item_loss + self.alpha * loss_dict['item_loss'].item()
            self.object_position_loss = (1-self.alpha) * self.object_position_loss + self.alpha * loss_dict['object_position_loss'].item()
            self.target_position_loss = (1-self.alpha) * self.target_position_loss + self.alpha * loss_dict['target_position_loss'].item()
            self.stage_loss = (1-self.alpha) * self.stage_loss + self.alpha * loss_dict['stage_loss'].item()
            self.action_position_loss = (1-self.alpha) * self.action_position_loss + self.alpha * loss_dict['action_position_loss'].item()
            self.action_orientation_loss = (1-self.alpha) * self.action_orientation_loss + self.alpha * loss_dict['action_orientation_loss'].item()
            self.action_open_loss = (1-self.alpha) * self.action_open_loss + self.alpha * loss_dict['action_open_loss'].item()
            
        
        normalized_loss = {
            'nll_loss': loss_dict['nll_loss']/self.nll_loss,
            # 'gripper_position_loss': loss_dict['gripper_position_loss']/self.gripper_position_loss,
            # 'gripper_orientation_loss': loss_dict['gripper_orientation_loss']/self.gripper_orientation_loss,
            # 'gripper_open_loss': loss_dict['gripper_open_loss']/self.gripper_open_loss,
            'item_loss': loss_dict['item_loss']/self.item_loss,
            'object_position_loss': loss_dict['object_position_loss']/self.object_position_loss,
            'target_position_loss': loss_dict['target_position_loss']/self.target_position_loss,
            'stage_loss': loss_dict['stage_loss']/self.stage_loss,
            'action_position_loss': loss_dict['action_position_loss']/self.action_position_loss,
            'action_orientation_loss': loss_dict['action_orientation_loss']/self.action_orientation_loss,
            'action_open_loss': loss_dict['action_open_loss']/self.action_open_loss
        }

        normalized_loss.update({'total_loss': 
            0.4*normalized_loss['nll_loss'] + 
            # 0.05*normalized_loss['gripper_position_loss'] + 0.05*normalized_loss['gripper_orientation_loss'] + 0.05*normalized_loss['gripper_open_loss'] +
            #   0.05*normalized_loss['item_loss'] +
                0.2*normalized_loss['object_position_loss'] +
                #   0.1*normalized_loss['target_position_loss'] +
                    # 0.1*normalized_loss['stage_loss'] +
                      0.2*normalized_loss['action_position_loss'] + 0.2*normalized_loss['action_orientation_loss'] #+ 0.05*normalized_loss['action_open_loss']
                      })
        return normalized_loss

class AngleLoss(nn.Module):
    def __init__(self):
        super(AngleLoss, self).__init__()

    def forward(self, pred, target):
        pred_rx = pred[:, 0]
        pred_ry = pred[:, 1]
        pred_rz = pred[:, 2]
        target_rx = target[:, 0]
        target_ry = target[:, 1]
        target_rz = target[:, 2]

        #loss rx
        pred_rx = torch.where(pred_rx > 0, pred_rx - torch.pi, pred_rx + torch.pi)
        target_rx = torch.where(target_rx>0, target_rx - torch.pi, target_rx + torch.pi)
        loss_rx = F.mse_loss(pred_rx, target_rx)

        #loss ry
        loss_ry = F.mse_loss(pred_ry, target_ry)

        #loss rz 
        error_rz = torch.abs(pred_rz - target_rz)
        error_rz = torch.min(error_rz, torch.pi - error_rz)
        loss_rz = torch.mean(error_rz**2)
        loss = (loss_rx + loss_ry + loss_rz)/3
        return loss

#create a sampler for weighted random sampling
class SamplerForPosePrediction(Sampler):
    def __init__(self, group_labels, group1_ratio=0.6):
        self.group_labels = group_labels
        self.group1_ratio = group1_ratio
        self.group0_indices = np.where(group_labels == 0)[0]
        self.group1_indices = np.where(group_labels == 1)[0]
        self.group0_weight = (1 - group1_ratio) / len(self.group0_indices)
        self.group1_weight = group1_ratio / len(self.group1_indices)
        self.weights = np.zeros(len(group_labels))
        self.weights[self.group0_indices] = self.group0_weight
        self.weights[self.group1_indices] = self.group1_weight

    def __iter__(self):
        return iter(torch.multinomial(torch.tensor(self.weights, dtype=torch.double), len(self.weights), replacement=True).tolist())

    def __len__(self):
        return len(self.group_labels)
